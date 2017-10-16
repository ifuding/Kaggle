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
    train = pd.read_csv(data_folder + 'train.csv').iloc[:10000]
    train_label = train['target'].astype(np.int8)
    train = train.drop(['target', 'id'], axis = 1).values
    test = pd.read_csv(data_folder + 'test.csv').iloc[:1000]
    test_id = test['id'].astype(np.int32).values
    test = test.drop(['id'], axis = 1).values
    return train, train_label, test, test_id


def lgbm_train(train_data, train_label, fold = 5, valide_data = None, valide_label = None):
    """
    LGB Training
    """
    print("Over all training size:")
    print(train_data.shape)

    kf = KFold(len(train_label), n_folds=fold, shuffle=True)

    num_fold = 0
    models = []
    for train_index, test_index in kf:
        if valide_label is None:
            d_train = lgb.Dataset(train_data[train_index], train_label[train_index]) #, categorical_feature = SPARCE_INDICES)
            d_valide = lgb.Dataset(train_data[test_index], train_label[test_index])
        else:
            d_train = lgb.Dataset(train_data, train_label) #, categorical_feature = SPARCE_INDICES)
            d_valide = lgb.Dataset(valide_data, valide_label)

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
        #cv_result = lgb.cv(params, d_train, nfold=fold)
        #pd.DataFrame(cv_result).to_csv('cv_result', index = False)
        #exit(0)
        models.append((bst, 'l'))

    return models


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
    train, train_label, test, test_id = load_data()
    model_l = lgbm_train(train, train_label, 2, train, train_label)
    gen_sub(model_l, test, test_id)
