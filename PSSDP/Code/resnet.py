
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


def dense_bn_layer(input_tensor, hn_num):
    """
    """
    x = Dense(hn_num)(input_tensor)
    x = BatchNormalization()(x)
    return x


def dense_bn_act_layer(input_tensor, hn_num, act = 'relu'):
    """
    """
    x = Dense(hn_num)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    return x


def identity_block(input_tensor, hn_num):
    """
    """
    adjust_layer = dense_bn_layer(input_tensor, hn_num)
    x = Activation('relu')(adjust_layer)
    x = dense_bn_act_layer(x, hn_num * 3 / 2)
    x = dense_bn_layer(x, hn_num)
    x = Add()([x, adjust_layer])
    x = Activation('relu')(x)
    return x


def res_net(input_shape, hns = [16, 8, 4, 7], classes = 2):
    """
    """
    inputs = Input(shape=input_shape)
    x = identity_block(inputs, hns[0])
    x = identity_block(x, hns[1])
    x = identity_block(x, hns[2])
    # x = identity_block(x, hns[3])
    if classes == 2:
        x = Dense(1, activation='sigmoid')(x)
    else:
        x = Dense(classes, activation='softmax')(x)
    model = Model(inputs, x)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model
