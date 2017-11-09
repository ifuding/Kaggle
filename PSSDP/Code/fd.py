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
import h5py
# import concurrent.futures
import tensorflow as tf
# import multiprocessing as mp
import gensim
import os

from sklearn.cross_validation import KFold
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

DNN_EPOCHS = 10
BATCH_SIZE = 50
DNN_BN = False
HIDDEN_UNITS = [40, 20, 5]
DROPOUT_RATE = 0
LOAD_DATA = True

if LOAD_DATA:
    data_folder = '../Data/'
    train = pd.read_csv(data_folder + 'train.csv')
    continus_columns = train.columns.to_series().select(lambda s: not s.endswith(('_cat', 'id', 'target', '_bin'))).values
    print('continus_columns: {}\n{}'.format(len(continus_columns), continus_columns))
    category_columns = train.columns.to_series().select(lambda s: s.endswith(('_cat'))).values
    print('category_columns: {}\n{}'.format(len(category_columns), category_columns))
    binary_columns = train.columns.to_series().select(lambda s: s.endswith(('_bin'))).values
    print('binary_columns: {}\n{}'.format(len(binary_columns), binary_columns))
    category_nunique = train[category_columns].nunique().values
    feature_columns = list(continus_columns) + list(binary_columns)
    train_label = train['target'].astype(np.int8)
    train_category = train[category_columns].values
    train = train.drop(['target', 'id'], axis = 1)[feature_columns].values
    test = pd.read_csv(data_folder + 'test.csv')
    test_id = test['id'].astype(np.int32).values
    test_category = test[category_columns]
    test = test.drop(['id'], axis = 1)[feature_columns].values
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


def xgb_train(train_data, train_label, fold = 5, stacking = False, valide_data = None, valide_label = None, test_data = None):
    """
    """
    denom = 0
    # fold = 5 #Change to 5, 1 for Kaggle Limits
    models = []
    kf = KFold(len(train_label), n_folds=fold, shuffle=True)

    stacking_data = None
    stacking_label = None
    test_preds = None

    i = 0
    for train_index, test_index in kf:
        if valide_label is None:
            train_part = train_data[train_index]
            train_part_label = train_label[train_index]
            valide_part = train_data[test_index]
            valide_part_label = train_label[test_index]
        else:
            train_part = train_data
            train_part_label = train_label
            valide_part = valide_data
            valide_part_label = valide_label

        params = {
            'eta': 0.03,
            'max_depth': 6,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': i,
            'silent': True
        }
        #if valide_data is None:
        #    x1, x2, y1, y2 = model_selection.train_test_split(train_data, train_label, test_size=1./fold, random_state=i)
        #else:
        #    x1, x2, y1, y2 = train_data, valide_data, train_label, valide_label
        watchlist = [(xgb.DMatrix(train_part, train_part_label), 'train'), (xgb.DMatrix(valide_part, valide_part_label), 'valid')]
        model = xgb.train(params, xgb.DMatrix(train_part, train_part_label), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
        #cv_result = xgb.cv(params, xgb.DMatrix(x1, y1), 1000, nfold = fold, verbose_eval=50, early_stopping_rounds = 100)
        #cv_result.to_csv('xgb_cv_result', index = False)
        #exit(0)
        if stacking:
            valide_pred = model_eval(model, 'x', valide_part)
            test_pred = model_eval(model, 'x', test_data)
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
        models.append((model, 'x'))
        i += 1
    test_preds /= fold
    test_data = np.c_[test_data, test_preds]

    return models, stacking_data, stacking_label, test_data

def create_embedding_layer():
    input = Input(shape=(), dtype='int8')
    x_ohe = Lambda(K.one_hot, arguments={'nb_classes': nb_classes}, output_shape=output_shape)(input)


def create_dnn(input_len):
    model = Sequential()
    # First HN
    model.add(Dense(HIDDEN_UNITS[0], activation='sigmoid', input_dim = input_len))
    if DNN_BN:
        model.add(BatchNormalization())
    if DROPOUT_RATE > 0:
        model.add(Dropout(DROPOUT_RATE))
    # Second HN
    model.add(Dense(HIDDEN_UNITS[1], activation='sigmoid'))
    if DNN_BN:
        model.add(BatchNormalization())
    if DROPOUT_RATE > 0:
        model.add(Dropout(DROPOUT_RATE))
    # Third HN
    model.add(Dense(HIDDEN_UNITS[2], activation='sigmoid'))
    if DNN_BN:
        model.add(BatchNormalization())
    if DROPOUT_RATE > 0:
        model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1, activation='sigmoid'))

    # optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = RMSprop(lr=1e-3, rho = 0.9, epsilon = 1e-8)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model

def keras_train(train_data, train_label, fold = 5, stacking = False, valide_data = None, valide_label = None):
    """
    LGB Training
    """
    print("Over all training size:")
    print(train_data.shape)

    kf = KFold(len(train_label), n_folds=fold, shuffle=True)

    stacking_data = None
    stacking_label = None
    num_fold = 0
    models = []
    for train_index, test_index in kf:
        if valide_label is None:
            train_part = train_data[train_index]
            train_part_label = train_label[train_index]
            valide_part = train_data[test_index]
            valide_part_label = train_label[test_index]
        else:
            train_part = train_data
            train_part_label = train_label
            valide_part = valide_data
            valide_part_label = valide_label

        model = create_dnn(train_data.shape[1])

        print('fold: %d th keras train :-)' % (num_fold))
        num_fold += 1
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
        model.fit(train_part,
                train_part_label, batch_size=BATCH_SIZE, epochs=DNN_EPOCHS,
                shuffle=True, verbose=2,
                validation_data=(valide_part, valide_part_label)
                , callbacks=callbacks)

        if stacking:
            if stacking_data is None:
                valide_pred = model_eval(model, 'k', valide_part)
                stacking_data = np.c_[valide_part, valide_pred]
                stacking_label = valide_part_label
            else:
                valide_pred = model_eval(model, 'k', valide_part)
                stacking_data = np.append(stacking_data, np.c_[valide_part, valide_pred], axis = 0)
                stacking_label = np.append(stacking_label, valide_part_label, axis = 0)
            print('stacking_data shape: {0}'.format(stacking_data.shape))
            print('stacking_label shape: {0}'.format(stacking_label.shape))
        models.append((model, 'k'))

    return models, stacking_data, stacking_label


def lgbm_train(train_data, train_label, fold = 5, stacking = False, valide_data = None, valide_label = None):
    """
    LGB Training
    """
    print("Over all training size:")
    print(train_data.shape)

    kf = KFold(len(train_label), n_folds=fold, shuffle=True)

    stacking_data = None
    stacking_label = None
    num_fold = 0
    models = []
    for train_index, test_index in kf:
        if valide_label is None:
            train_part = train_data[train_index]
            train_part_label = train_label[train_index]
            valide_part = train_data[test_index]
            valide_part_label = train_label[test_index]
        else:
            train_part = train_data
            train_part_label = train_label
            valide_part = valide_data
            valide_part_label = valide_label

        d_train = lgb.Dataset(train_part, train_part_label)
        d_valide = lgb.Dataset(valide_part, valide_part_label)
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss'},
          #  'feature_fraction': 0.9,
          #  'bagging_fraction': 0.95,
          #  'bagging_freq': 5,
            'num_leaves': 40, # 60,
          #  'min_sum_hessian_in_leaf': 20,
            'max_depth': 6, # 10,
            'learning_rate': 0.028, # 0.025,
           'feature_fraction': 0.35, # 0.6
            'verbose': 0,
          #   'valid_sets': [d_valide],
            'num_boost_round': 400, #381,
            'feature_fraction_seed': num_fold,
            # 'bagging_fraction': 0.9,
            # 'bagging_freq': 15,
            # 'bagging_seed': i,
            # 'early_stopping_round': 10,
            # 'random_state': 10
            # 'verbose_eval': 20
            #'min_data_in_leaf': 665
        }

        print('fold: %d th light GBM train :-)' % (num_fold))
        num_fold += 1
        bst = lgb.train(
                        params ,
                        d_train,
                        verbose_eval = 50,
                        valid_sets = [d_train, d_valide],
                        feval = gini_lgbm
                        #num_boost_round = 1
                        )
        if stacking:
            if stacking_data is None:
                valide_pred = model_eval(bst, 'l', valide_part)
                stacking_data = np.c_[valide_part, valide_pred]
                stacking_label = valide_part_label
            else:
                valide_pred = model_eval(bst, 'l', valide_part)
                stacking_data = np.append(stacking_data, np.c_[valide_part, valide_pred], axis = 0)
                stacking_label = np.append(stacking_label, valide_part_label, axis = 0)
            print('stacking_data shape: {0}'.format(stacking_data.shape))
            print('stacking_label shape: {0}'.format(stacking_label.shape))
        #cv_result = lgb.cv(params, d_train, nfold=fold) #, feval = gini_lgbm)
        #pd.DataFrame(cv_result).to_csv('cv_result', index = False)
        #exit(0)
        models.append((bst, 'l'))

    return models, stacking_data, stacking_label


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
    return preds


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
    train, train_label, test, test_id = load_data()
    #model_l, stacking_data, stacking_label = lgbm_train(train, train_label, 5, True)
    #np.save('stacking_data', stacking_data)
    #np.save('stacking_label', stacking_label)
    #stacking_data = np.load('stacking_data_xgb.npy')
    #stacking_label = np.load('stacking_label_xgb.npy')
    #test = np.load('stacking_test_data_xgb.npy')
    #model_x, stacking_data, stacking_label, test = xgb_train(train, train_label, 5, True, None, None, test)
    #np.save('stacking_data_xgb', stacking_data)
    #np.save('stacking_label_xgb', stacking_label)
    #np.save('stacking_test_data_xgb', test)
    # model_l, stacking_data, stacking_label = lgbm_train(stacking_data, stacking_label, 5, False)
    model_k, stacking_data, stacking_label = keras_train(train, train_label, 5, False)
    #gen_sub(model_l, test, test_id)
