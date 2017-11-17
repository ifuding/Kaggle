
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
from lcc_sample import lcc_sample

from sklearn.cross_validation import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers import Input, concatenate, merge, LSTM, Lambda, Add, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def dense_bn_layer(input_tensor, hn_num, name = None):
    """
    """
    x = Dense(hn_num)(input_tensor)
    x = BatchNormalization(name = name)(x)
    return x


def dense_bn_act_layer(input_tensor, hn_num, name = None, act = 'relu'):
    """
    """
    x = Dense(hn_num)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(act, name = name)(x)
    return x


def identity_block(input_tensor, hn_num, name = None):
    """
    """
    adjust_layer = dense_bn_layer(input_tensor, hn_num)
    x = Activation('relu')(adjust_layer)
    x = dense_bn_act_layer(x, hn_num * 3 / 2)
    x = dense_bn_layer(x, hn_num)
    x = Add()([x, adjust_layer])
    x = Activation('relu', name = name)(x)
    return x


def res_net(input_shape, hns = [8, 6, 4, 7], classes = 2):
    """
    """
    inputs = Input(shape=input_shape)
    x = identity_block(inputs, hns[0], name = 'block0')
    x = identity_block(x, hns[1], name = 'block1')
    x = identity_block(x, hns[2], name = 'block2')
    # x = identity_block(x, hns[3])
    if classes == 2:
        x = Dense(1, activation='sigmoid')(x)
    else:
        x = Dense(classes, activation='softmax')(x)
    model = Model(inputs, x)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model


def create_embedding_layer():
    input_list = []
    embedding_list = []
    for nunique in category_nunique:
        input_ = Input(shape=(1, ), dtype='int32')
        # x_ohe = Lambda(K.one_hot, arguments={'num_classes': nunique})(input_)
        x_ohe = Lambda(one_hot, arguments={'num_classes': nunique})(input_)
        # x_ohe = K.one_hot(input_, nunique)
        input_list.append(input_)
        embedding_list.append(x_ohe)
    return input_list, concatenate(embedding_list, axis = 2)


def create_dnn(input_len, HIDDEN_UNITS = [20, 10, 4], DNN_BN = False, DROPOUT_RATE = 0):
    inputs = Input(shape=(input_len,))
    x = dense_bn_act_layer(inputs, HIDDEN_UNITS[0], name = 'hn1')
    x = dense_bn_act_layer(x, HIDDEN_UNITS[1], name = 'hn2')
    # x = dense_bn_act_layer(x, HIDDEN_UNITS[2], name = 'hn3')
    x = dense_bn_act_layer(x, 1, name = 'prob', act = 'sigmoid')
    ## First HN
    #model.add(Dense(HIDDEN_UNITS[0], activation='relu', input_dim = input_len))
    #if DNN_BN:
    #    model.add(BatchNormalization())
    #if DROPOUT_RATE > 0:
    #    model.add(Dropout(DROPOUT_RATE))

    # optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = RMSprop(lr=1e-3, rho = 0.9, epsilon = 1e-8)
    model = Model(inputs, x)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model


def create_embedding_model():
    """
    """
    dense_input = Input(shape=(len(continus_binary_indice),))
    input_list, embedding_layer = create_embedding_layer()
    embedding_layer = Flatten()(embedding_layer)
    merge_input = concatenate([dense_input, embedding_layer], axis = 1)

    merge_len = len(continus_binary_indice) + sum(category_nunique)
    output = create_dnn(merge_len)(merge_input)

    model = Model([dense_input] + input_list, output)
    # optimizer = RMSprop(lr=1e-3, rho = 0.9, epsilon = 1e-8)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model
