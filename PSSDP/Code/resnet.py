
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


def identity_block(input_tensor, hn_num):
    """
    """
    x = Dense(hn_num * 3 / 2)(input_tensor)
    x = BatchNormalization()(x)
    x = Dense(hn_num)(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def res_net(input_shape, hns = [16, 8, 4], classes = 2):
    """
    """
    inputs = Input(shape=input_shape)
    x = Dense(hns[0], activation='relu')(inputs)
    x = identity_block(x, hns[0])
    x = Dense(hns[1], activation='relu')(inputs)
    x = identity_block(x, hns[1])
    x = Dense(hns[2], activation='relu')(inputs)
    x = identity_block(x, hns[2])
    if classes == 2:
        x = Dense(1, activation='sigmoid')(x)
    else:
        x = Dense(classes, activation='softmax')(x)
    model = Model(inputs, x)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model
