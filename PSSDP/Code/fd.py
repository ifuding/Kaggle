from sklearn import metrics, preprocessing, pipeline, \
    feature_extraction, decomposition, model_selection
import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from time import gmtime, strftime
import numpy.random as rng
# from multiprocessing.dummy import Pool
# import h5py
# import concurrent.futures
import tensorflow as tf
# import multiprocessing as mp
# import gensim
import os
from lcc_sample import lcc_sample
from scipy.special import logit
from scipy.special import expit as sigmoid

from sklearn.cross_validation import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers import Input, concatenate, merge, LSTM, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from resnet import res_net, create_dnn

DNN_EPOCHS = 30
BATCH_SIZE = 125
DNN_BN = True
HIDDEN_UNITS = [32, 16, 8]
DROPOUT_RATE = 0
LOAD_DATA = True

if LOAD_DATA:
    data_folder = '../Data/'
    train = pd.read_csv(data_folder + 'train.csv')#.iloc[:1000]
    train_len = train.index.size
    continus_columns = train.columns.to_series().select(lambda s: not s.endswith(('_cat', 'id', 'target', '_bin'))).values
    #print('continus_columns: {}\n{}'.format(len(continus_columns), continus_columns))
    category_columns = train.columns.to_series().select(lambda s: s.endswith(('_cat'))).values
    #print('category_columns: {}\n{}'.format(len(category_columns), category_columns))
    binary_columns = train.columns.to_series().select(lambda s: s.endswith(('_bin'))).values
    #print('binary_columns: {}\n{}'.format(len(binary_columns), binary_columns))
    category_nunique = train[category_columns].nunique().values
    continus_binary_columns = list(continus_columns) + list(binary_columns)
    train_label = train['target'].astype(np.int8)
    # train_category = train[category_columns].values
    train = train.drop(['target', 'id'], axis = 1)
    train_columns = train.columns.values
    category_indice = [i for i in range(len(train_columns)) if train_columns[i] in category_columns]
    continus_binary_indice = [i for i in range(len(train_columns)) if train_columns[i] not in category_columns]
    #print ('category_indice: {}'.format(category_indice))
    #print ('continus_binary_indice: {}'.format(continus_binary_indice))

    test = pd.read_csv(data_folder + 'test.csv')#.iloc[:1000]
    test_id = test['id'].astype(np.int32).values
    # test_category = test[category_columns]
    test = test.drop(['id'], axis = 1)

    df_all = pd.concat([train, test])
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoder = np.array([label_encoder.fit_transform(df_all[c].values) for c in category_columns]).T
    category_onehot = onehot_encoder.fit_transform(integer_encoder)
    # df_array = np.c_[df_all[continus_binary_columns].values, category_onehot]
    df_array = np.c_[df_all.values, category_onehot]
    feature_name = list(df_all.columns.values)
    feature_name += ['cat_one_hot_' + str(i) for i in range(category_onehot.shape[1])]
    train = df_array[:train_len, :]
    test = df_array[train_len:, :]
    # return train, train_label, test, test_id

def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


def gini_lgbm(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return 'gini', gini_score, True


def xgb_train(train_part, train_part_label, valide_part, valide_part_label, fold_seed):
    """
    """
    print("-----xgb training-----")
    params = {
            'eta': 0.03,
            'max_depth': 6,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': fold_seed,
            'silent': True
        }
    watchlist = [(xgb.DMatrix(train_part, train_part_label), 'train'), (xgb.DMatrix(valide_part, valide_part_label), 'valid')]
    model = xgb.train(params, xgb.DMatrix(train_part, train_part_label), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    #cv_result = xgb.cv(params, xgb.DMatrix(x1, y1), 1000, nfold = fold, verbose_eval=50, early_stopping_rounds = 100)
    #cv_result.to_csv('xgb_cv_result', index = False)
    #exit(0)
    return model


def keras_train(train_part, train_part_label, valide_part, valide_part_label, fold_seed):
    """
    Keras Training
    """
    print("-----Keras training-----")

    model = res_net((train_part.shape[1],))
    # model = create_dnn(train_part.shape[1])
    # model = create_embedding_model()

    callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
            ]

    #model.fit([train_part[:, continus_binary_indice]] + [train_part[:, i] for i in category_indice],
    #        train_part_label, batch_size=BATCH_SIZE, epochs=DNN_EPOCHS,
    #        shuffle=True, verbose=2,
    #        validation_data=([valide_part[:, continus_binary_indice]] + [valide_part[:, i] for i in category_indice], valide_part_label)
    #        , callbacks=callbacks)
    model.fit(train_part, train_part_label, batch_size=BATCH_SIZE, epochs=DNN_EPOCHS,
                shuffle=True, verbose=2,
                validation_data=(valide_part, valide_part_label)
                , callbacks=callbacks)

    return model


def lgbm_train(train_part, train_part_label, valide_part, valide_part_label, fold_seed,
        fold = 5, train_weight = None, valide_weight = None):
    """
    LGBM Training
    """
    print("-----LGBM training-----")

    d_train = lgb.Dataset(train_part, train_part_label, weight = train_weight)
    d_valide = lgb.Dataset(valide_part, valide_part_label, weight = valide_weight)
    params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss'},
          #  'feature_fraction': 0.9,
          #  'bagging_fraction': 0.95,
          #  'bagging_freq': 5,
            'num_leaves': 60, #60, #40, # 60,
          #  'min_sum_hessian_in_leaf': 20,
            'max_depth': 10,#12, #6, # 10,
            'learning_rate': 0.028, # 0.025,
           'feature_fraction': 0.5,#0.35, # 0.6
            'verbose': 0,
          #   'valid_sets': [d_valide],
            'num_boost_round': 400, #361,
            'feature_fraction_seed': fold_seed,
            #'bagging_fraction': 0.9,
            # 'bagging_freq': 15,
            #'bagging_seed': fold_seed,
            # 'early_stopping_round': 10,
            # 'random_state': 10
            # 'verbose_eval': 20
            #'min_data_in_leaf': 665
        }

    #bst = lgb.train(
    #                params ,
    #                d_train,
    #                verbose_eval = 50,
    #                valid_sets = [d_train, d_valide],
    #                feature_name=['f' + str(i + 1) for i in range(train_part.shape[1])],
    #                #feval = gini_lgbm
    #                #num_boost_round = 1
    #                )
    #feature_imp = bst.feature_importance(importance_type = 'gain')
    #print (feature_name[np.argsort(feature_imp)])
    #exit(0)
    cv_result = lgb.cv(params, d_train, nfold=fold) #, feval = gini_lgbm)
    pd.DataFrame(cv_result).to_csv('cv_result', index = False)
    exit(0)
    return bst


def nfold_train(train_data, train_label, fold = 5, model_types = None,
            stacking = False, valide_data = None, valide_label = None,
            test_data = None, train_weight = None, valide_weight = None):
    """
    nfold Training
    """
    print("Over all training size:")
    print(train_data.shape)

    kf = KFold(len(train_label), n_folds=fold, shuffle=True)

    stacking_data = None
    stacking_label = None
    test_preds = None
    num_fold = 0
    models = []
    for train_index, test_index in kf:
        print('fold: %d th train :-)' % (num_fold))
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
        onefold_models = []
        for model_type in model_types:
            if model_type == 'k':
                model = keras_train(train_part, train_part_label, valide_part, valide_part_label, num_fold)
                block0 = Model(inputs = model.input, \
                            outputs = model.get_layer('block0').output)
                onefold_models.append((block0, 'k'))
                block1 = Model(inputs = model.input, \
                            outputs = model.get_layer('block1').output)
                onefold_models.append((block1, 'k'))
                block2 = Model(inputs = model.input, \
                            outputs = model.get_layer('block2').output)
                onefold_models.append((block2, 'k'))
            elif model_type == 'x':
                model = xgb_train(train_part, train_part_label, valide_part, valide_part_label, num_fold)
                onefold_models.append((model, 'x'))
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
                stacking_data = np.c_[valide_part, valide_pred]
                stacking_label = valide_part_label
                test_preds = test_pred
            else:
                stacking_data = np.append(stacking_data, np.c_[valide_part, valide_pred], axis = 0)
                stacking_label = np.append(stacking_label, valide_part_label, axis = 0)
                test_preds += test_pred
            print('stacking_data shape: {0}'.format(stacking_data.shape))
            print('stacking_label shape: {0}'.format(stacking_label.shape))
        models.append(onefold_models[0])
        num_fold += 1
    if stacking:
        test_preds /= fold
        test_data = np.c_[test_data, test_preds]
    return models, stacking_data, stacking_label, test_data


def model_eval(model, model_type, data_frame):
    """
    """
    if model_type == 'l':
        preds = model.predict(data_frame)
    elif model_type == 'k' or model_type == 'LR' or model_type == 'DNN' or model_type == 'RCNN':
        preds = model.predict(data_frame, batch_size=BATCH_SIZE, verbose=2)
    elif model_type == 't':
        print("ToDO")
    elif model_type == 'x':
        preds = model.predict(xgb.DMatrix(data_frame), ntree_limit=model.best_ntree_limit)
    return preds.reshape((data_frame.shape[0], -1))


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


def lcc_ensemble(pilot_models, sample_models, data):
    pilot_preds = models_eval(pilot_models, data)
    sample_preds = models_eval(sample_models, data)
    preds = sigmoid(logit(pilot_preds) + logit(sample_preds))
    return preds


def gen_sub(models, merge_features, test_id, preds = None):
    """
    Evaluate single Type model
    """
    print('Start generate submission!')
    if preds is None:
        preds = models_eval(models, merge_features)
    submission = pd.DataFrame(np.c_[test_id, preds], columns=['id', 'target'])
    submission['id'] = test_id
    sub_name = "submission" + strftime('_%Y_%m_%d_%H_%M_%S', gmtime()) + ".csv"
    print('Output to ' + sub_name)
    submission.to_csv(sub_name, index=False)

if __name__ == "__main__":
    # train, train_label, test, test_id = load_data()
    # model_l, stacking_data, stacking_label = lgbm_train(train, train_label, 5, False, train, train_label)
    #np.save('stacking_data', stacking_data)
    #np.save('stacking_label', stacking_label)
    #model_x, stacking_data, stacking_label, test = xgb_train(train, train_label, 5, True, None, None, test)
    #model_l, stacking_data, stacking_label = lgbm_train(stacking_data, stacking_label, 5, False, stacking_data, stacking_label)
    models, stacking_data, stacking_label, test = nfold_train(train, train_label, 5, ['k'], True, None, None, test)
    np.save('stacking_data_keras', stacking_data)
    np.save('stacking_label_keras', stacking_label)
    np.save('stacking_test_data_keras', test)
    #stacking_data = np.load('stacking_data_keras.npy')
    #stacking_label = np.load('stacking_label_keras.npy')
    #test = np.load('stacking_test_data_keras.npy')
    feature_name += ['dnn_emb_' + str(i) for i in range(stacking_data.shape[1] - train.shape[1])]
    feature_name = np.array(feature_name)
    # print('Before shuffle feature name: {}'.format(feature_name))
    feature_ind = np.array(range(stacking_data.shape[1]))
    np.random.shuffle(feature_ind)
    stacking_data = stacking_data[:, feature_ind]
    test = test[:, feature_ind]
    feature_name = feature_name[feature_ind]
    # print('After shuffle feature name: {}'.format(feature_name))
    #pilot_preds = models_eval(pilot_models, train)
    #sample_data, sample_label, weight = lcc_sample(train_label, pilot_preds, train, 2)
    #np.save('lcc_sample_data', sample_data)
    #np.save('lcc_sample_label', sample_label)
    #np.save('lcc_sample_weight', weight)
    #sample_data = np.load('lcc_sample_data.npy')
    #sample_label = np.load('lcc_sample_label.npy')
    #weight = np.load('lcc_sample_weight.npy')
    models, stacking_data, stacking_label, test = nfold_train(stacking_data, stacking_label, 5, ['l'], False, stacking_data, stacking_label, test)
    # lcc_preds = lcc_ensemble(pilot_models, sample_models, test)
    #np.save('stacking_data_kx', stacking_data)
    #np.save('stacking_label_kx', stacking_label)
    #np.save('stacking_test_data_kx', test)
    #gen_sub(models, test, test_id)
