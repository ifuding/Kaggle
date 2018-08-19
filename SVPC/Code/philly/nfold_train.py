from sklearn.model_selection import KFold
from lgb import lgbm_train
# import xgboost as xgb
# from functools import reduce
import numpy as np
from keras_train import DNN_Model, VAE_Model
import keras_train
# import gensim
# from RCNN_Keras import get_word2vec, RCNN_Model
# from RNN_Keras import RNN_Model
from tensorflow.python.keras.models import Model
# from xgb import xgb_train
import pandas as pd
from sklearn import metrics

# RNN_PARAMS
RCNN_HIDDEN_UNIT = [128, 64]

def cal_leak_blend_loss(leak_train, model, valide_data, valide_label):
    # print (leak_train.head)
    # leak_train = leak_train.copy()
    pred = pd.Series(model_eval(model[0], model[1], valide_data), index = valide_data.index)
    # leak_valide_part = leak_train.loc[valide_data.index]
    # print(leak_valide_part.shape)
    # print ('valide_data: ', valide_data.head())
    blend_leak_target = leak_train.loc[valide_data.index, 'leak_target']
    # loss = np.sqrt(metrics.mean_squared_error(valide_label, blend_leak_target.values))
    # print ('before blend leak_blend_loss: ', loss)
    blend_leak_target[blend_leak_target == 0] = pred[blend_leak_target == 0]
    # print(blend_leak_target[blend_leak_target == np.nan])
    loss = np.sqrt(metrics.mean_squared_error(valide_label, pred.values))
    blend_leak_loss = np.sqrt(metrics.mean_squared_error(valide_label, blend_leak_target.values))
    print ('loss: ', loss, 'leak_blend_loss: ', blend_leak_loss)
    return loss, blend_leak_loss

def nfold_train(train_data, train_label, model_types = None,
            stacking = False, valide_data = None, valide_label = None,
            test_data = None, train_weight = None, valide_weight = None,
            flags = None ,tokenizer = None, scores = None, emb_weight = None, cat_max = None):
    """
    nfold Training
    """
    print("Over all training size:")
    print(train_data.shape)
    print("Over all label size:")
    print(train_label.shape)

    fold = flags.nfold
    kf = KFold(n_splits=fold, shuffle=False)
    # wv_model = gensim.models.Word2Vec.load("wv_model_norm.gensim")
    stacking = flags.stacking
    stacking_data = None
    stacking_label = None
    test_preds = None
    num_fold = 0
    models = []
    losses = []
    leak_train = pd.read_csv(flags.input_training_data_path + '/train_target_leaktarget_38_2_2018_08_19_06_02_12.csv', index_col = 'ID').apply(np.log1p)
    for train_index, test_index in kf.split(train_data):
        # print(test_index[:100])
        # exit(0)
        if valide_label is None:
            train_part = train_data.iloc[train_index]
            train_part_label = None
            if model_types[0] != 'v' and model_types[0] != 'r':
                train_part_label = train_label[train_index]
            valide_part = train_data.iloc[test_index]
            valide_part_label = None
            if model_types[0] != 'v' and model_types[0] != 'r':
                valide_part_label = train_label[test_index]
            if train_weight is not None:
                train_part_weight = train_weight[train_index]
                valide_part_weight = train_weight[test_index]
        else:
            train_part = train_data
            train_part_label = train_label
            valide_part = valide_data
            valide_part_label = valide_label
            if train_weight is not None:
                train_part_weight, valide_part_weight = train_weight, valide_weight
        print('fold: %d th train :-)' % (num_fold))
        print('Train size: {} Valide size: {}'.format(train_part.shape[0], valide_part.shape[0]))
        onefold_models = []
        for model_type in model_types:
            if model_type == 'k' or model_type == 'r':
                # with tf.device('/cpu:0'):
                model = DNN_Model(scores = scores, cat_max = cat_max, flags = flags, emb_weight = emb_weight, model_type = model_type)
                if num_fold == 0:
                    print(model.model.summary())
                model.train(train_part, train_part_label, valide_part, valide_part_label)
                # if stacking:
                #     model = Model(inputs = model.model.inputs, outputs = model.model.get_layer(name = 'merge_sparse_emb').output)
                onefold_models.append((model, model_type))
            elif model_type == 'v':
                # with tf.device('/cpu:0'):
                model = VAE_Model(flags = flags)
                if num_fold == 0:
                    print(model.model.summary())
                model.train(train_part, train_part_label, valide_part, valide_part_label)
                model = Model(inputs = model.model.inputs, outputs = model.model.get_layer(name = 'z').output)
                # if stacking:
                #     model = Model(inputs = model.model.inputs, outputs = model.model.get_layer(name = 'merge_sparse_emb').output)
                onefold_models.append((model, 'v'))
                stacking_data = model_eval(model, 'v', train_data) # for model in onefold_models]
                # stacking_data = reduce((lambda x, y: np.c_[x, y]), stacking_data)
                print('stacking_data shape: {0}'.format(stacking_data.shape))
            elif model_type == 'x':
                model = xgb_train(train_part, train_part_label, valide_part, valide_part_label, num_fold)
                onefold_models.append((model, 'x'))
            elif model_type == 'l':
                model = lgbm_train(train_part, train_part_label, valide_part, valide_part_label, num_fold,
                        fold, flags = flags)
                onefold_models.append((model, 'l'))
                # print (leak_train.head)
        losses.append(cal_leak_blend_loss(leak_train, onefold_models[0], valide_part, valide_part_label))
        # if stacking:
        #     valide_pred = [model_eval(model[0], model[1], valide_part) for model in onefold_models]
        #     valide_pred = reduce((lambda x, y: np.c_[x, y]), valide_pred)
        #     test_pred = [model_eval(model[0], model[1], test_data) for model in onefold_models]
        #     test_pred = reduce((lambda x, y: np.c_[x, y]), test_pred)
        #     if stacking_data is None:
        #         stacking_data = valide_pred #np.c_[valide_part, valide_pred]
        #         stacking_label = valide_part_label
        #         test_preds = test_pred
        #     else:
        #         stacking_data = np.append(stacking_data, valide_pred, axis = 0) #np.append(stacking_data, np.c_[valide_part, valide_pred], axis = 0)
        #         stacking_label = np.append(stacking_label, valide_part_label, axis = 0)
        #         test_preds += test_pred
        #     print('stacking_data shape: {0}'.format(stacking_data.shape))
        #     print('stacking_label shape: {0}'.format(stacking_label.shape))
        #     print('stacking test data shape: {0}'.format(test_preds.shape))
        models.append(onefold_models[0])
        num_fold += 1
        if num_fold == flags.ensemble_nfold:
            break
    mean_loss = np.array(losses).mean(axis = 0)
    print ('Mean loss: ', mean_loss[0], "Mean blend loss: ", mean_loss[1])
    # if stacking:
    #     test_preds /= flags.ensemble_nfold
        # test_data = np.c_[test_data, test_preds]
    return models, stacking_data, stacking_label, test_preds


def model_eval(model, model_type, data_frame):
    """
    """
    if model_type == 'l':
        preds = model.predict(data_frame[keras_train.USED_FEATURE_LIST].values)
    elif model_type == 'k' or model_type == 'LR' or model_type == 'DNN' or model_type == 'rcnn' \
        or model_type == 'r' or model_type == 'cnn':
        preds = model.predict(data_frame, verbose = 2)
    elif model_type == 'v':
        preds = model.predict(data_frame[keras_train.USED_FEATURE_LIST].values, verbose = 2)
    elif model_type == 't':
        print("ToDO")
    elif model_type == 'x':
        preds = model.predict(xgb.DMatrix(data_frame), ntree_limit=model.best_ntree_limit)
    return preds #.reshape((data_frame.shape[0], -1))

def models_eval(models, data):
    preds = None
    for (model, model_type) in models:
        pred = model_eval(model, model_type, data)
        if preds is None:
            preds = pred.copy()
        else:
            preds += pred
    preds /= len(models)
    return preds