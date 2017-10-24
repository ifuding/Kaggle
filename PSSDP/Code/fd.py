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
from keras.layers import Input, concatenate, merge, LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load_data():
    data_folder = '../Data/'
    train = pd.read_csv(data_folder + 'train.csv')
    train_label = train['target'].astype(np.int8)
    train = train.drop(['target', 'id'], axis = 1).values
    test = pd.read_csv(data_folder + 'test.csv')
    test_id = test['id'].astype(np.int32).values
    test = test.drop(['id'], axis = 1).values
    return train, train_label, test, test_id


def xgb_train(train_data, train_label, fold = 5, valide_data = None, valide_label = None):
    """
    """
    denom = 0
    # fold = 5 #Change to 5, 1 for Kaggle Limits
    models = []
    for i in range(fold):
        params = {
            'eta': 0.03,
            'max_depth': 6,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': i,
            'silent': True
        }
        if valide_data is None:
            x1, x2, y1, y2 = model_selection.train_test_split(train_data, train_label, test_size=1./fold, random_state=i)
        else:
            x1, x2, y1, y2 = train_data, valide_data, train_label, valide_label
        watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
        # model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
        #score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
        #print(score1)
        cv_result = xgb.cv(params, xgb.DMatrix(x1, y1), 1000, nfold = fold, verbose_eval=50, early_stopping_rounds = 100)
        cv_result.to_csv('xgb_cv_result', index = False)
        exit(0)
        models.append((model, 'x'))

    return models


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
            'learning_rate': 0.03, # 0.025,
           'feature_fraction': 0.6, # 0.6
            'verbose': 0,
          #   'valid_sets': [d_valide],
            'num_boost_round': 360, #381,
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
                        valid_sets = [d_train, d_valide]
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
        #cv_result = lgb.cv(params, d_train, nfold=fold)
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
    # submission['id'] = test_id
    sub_name = "submission" + strftime('_%Y_%m_%d_%H_%M_%S', gmtime()) + ".csv"
    print('Output to ' + sub_name)
    submission.to_csv(sub_name, index=False)

if __name__ == "__main__":
    #train, train_label, test, test_id = load_data()
    #model_l, stacking_data, stacking_label = lgbm_train(train, train_label, 5, True)
    #np.save('stacking_data', stacking_data)
    #np.save('stacking_label', stacking_label)
    stacking_data = np.load('stacking_data.npy')# [:, :-1]
    stacking_label = np.load('stacking_label.npy')
    model_x = xgb_train(stacking_data, stacking_label, 5)
    # gen_sub(model_l, test, test_id)
