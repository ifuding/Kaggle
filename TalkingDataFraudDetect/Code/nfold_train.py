from sklearn.model_selection import KFold
from lgb import lgbm_train
# import xgboost as xgb
# from functools import reduce
import numpy as np
from keras_train import DNN_Model
# import gensim
# from RCNN_Keras import get_word2vec, RCNN_Model
# from RNN_Keras import RNN_Model
from tensorflow.python.keras.models import Model

# RNN_PARAMS
RCNN_HIDDEN_UNIT = [128, 64]


def nfold_train(train_data, train_label, model_types = None,
            stacking = False, valide_data = None, valide_label = None,
            test_data = None, train_weight = None, valide_weight = None, 
            flags = None ,tokenizer = None, scores = None, emb_weight = None):
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
    for train_index, test_index in kf.split(train_data):
        # print(test_index[:100])
        # exit(0)
        if valide_label is None:
            train_part = train_data[train_index]
            train_part_label = train_label[train_index]
            valide_part = train_data[test_index]
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
        print('Train size: {} Valide size: {}'.format(train_part_label.shape[0], valide_part_label.shape[0]))
        onefold_models = []
        for model_type in model_types:
            if model_type == 'k':
                # with tf.device('/cpu:0'):
                if flags.load_only_singleCnt:
                    dense_input_len = train_part.shape[1]
                model = DNN_Model(hidden_dim = [int(hn.strip()) for hn in flags.full_connect_hn.strip().split(',')], \
                    batch_size = flags.batch_size, epochs = flags.epochs, \
                    batch_interval = flags.batch_interval, emb_dropout = flags.emb_dropout, \
                    full_connect_dropout = flags.full_connect_dropout, scores = scores, \
                    emb_dim = [int(e.strip()) for e in flags.emb_dim.strip().split(',')], \
                    load_only_singleCnt = flags.load_only_singleCnt, dense_input_len = dense_input_len)
                if num_fold == 0:
                    print(model.model.summary())
                if flags.load_only_singleCnt:
                    model.train(train_part, train_part_label, valide_part, valide_part_label)
                else:
                    model.train(list(train_part.transpose()), train_part_label, \
                        list(valide_part.transpose()), valide_part_label)
                if stacking:
                    model = Model(inputs = model.model.inputs, outputs = model.model.get_layer(name = 'RCNN_CONC').output)
                onefold_models.append((model, 'k'))
            elif model_type == 'x':
                pass
                # model = xgb_train(train_part, train_part_label, valide_part, valide_part_label, num_fold)
                # onefold_models.append((model, 'x'))
            elif model_type == 'l':
                model = lgbm_train(train_part, train_part_label, valide_part, valide_part_label, num_fold,
                        fold)
                onefold_models.append((model, 'l'))
        if stacking:
            valide_pred = [model_eval(model[0], model[1], valide_part) for model in onefold_models]
            valide_pred = reduce((lambda x, y: np.c_[x, y]), valide_pred)
            test_pred = [model_eval(model[0], model[1], test_data) for model in onefold_models]
            test_pred = reduce((lambda x, y: np.c_[x, y]), test_pred)
            if stacking_data is None:
                stacking_data = valide_pred #np.c_[valide_part, valide_pred]
                stacking_label = valide_part_label
                test_preds = test_pred
            else:
                stacking_data = np.append(stacking_data, valide_pred, axis = 0) #np.append(stacking_data, np.c_[valide_part, valide_pred], axis = 0)
                stacking_label = np.append(stacking_label, valide_part_label, axis = 0)
                test_preds += test_pred
            print('stacking_data shape: {0}'.format(stacking_data.shape))
            print('stacking_label shape: {0}'.format(stacking_label.shape))
            print('stacking test data shape: {0}'.format(test_preds.shape))
        models.append(onefold_models[0])
        num_fold += 1
        if num_fold == flags.ensemble_nfold:
            break
    if stacking:
        test_preds /= flags.ensemble_nfold
        # test_data = np.c_[test_data, test_preds]
    return models, stacking_data, stacking_label, test_preds


def model_eval(model, model_type, data_frame):
    """
    """
    if model_type == 'l':
        preds = model.predict(data_frame)
    elif model_type == 'k' or model_type == 'LR' or model_type == 'DNN' or model_type == 'rcnn' \
        or model_type == 'rnn' or model_type == 'cnn':
        preds = model.predict(data_frame, verbose = 2)
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