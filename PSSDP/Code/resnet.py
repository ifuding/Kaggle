
from sklearn import metrics, preprocessing, pipeline, \
    feature_extraction, decomposition, model_selection
import sklearn
import pandas as pd
import numpy as np
from time import gmtime, strftime
import numpy.random as rng
# from multiprocessing.dummy import Pool
# import concurrent.futures
import tensorflow as tf
# import multiprocessing as mp
import os
from lcc_sample import lcc_sample
from eval import min_pred

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers import Input, concatenate, merge, LSTM, Lambda, Add, Activation, Subtract
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l1, l2
from sklearn.metrics import log_loss
from keras import __version__ as keras_version


RANK_SCALE = 1
DROPOUT_RATE = 0.8 #0.35
EPSILON = 1e-7
L2_NORM = 0
R_RANK_GAMMA = 0.1
R_RANK_P = 1

def dense_bn_layer(input_tensor, hn_num, name = None, dropout = True, bn = True):
    """
    """
    hn_num = int(hn_num)
    x = Dense(hn_num, kernel_regularizer = l2(L2_NORM))(input_tensor)
    if bn:
        x = BatchNormalization(name = name)(x)
    if dropout:
        x = Dropout(DROPOUT_RATE)(x)
    return x


def dense_bn_act_layer(input_tensor, hn_num, name = None, act = 'relu', dropout = True, bn = True):
    """
    """
    hn_num = int(hn_num)
    x = Dense(hn_num, kernel_regularizer = l2(L2_NORM))(input_tensor)
    if bn:
        x = BatchNormalization()(x)
    if dropout:
        x = Dropout(DROPOUT_RATE)(x)
    x = Activation(act, name = name)(x)
    return x


def identity_block(input_tensor, hn_num, name = None, dropout = True):
    """
    """
    adjust_layer = dense_bn_layer(input_tensor, hn_num, dropout = dropout)
    x = Activation('relu')(adjust_layer)
    # x = dense_bn_act_layer(x, hn_num * 3 / 2, dropout = dropout)
    x = dense_bn_act_layer(x, hn_num, dropout = dropout)
    x = dense_bn_layer(x, hn_num, dropout = dropout)
    x = Add()([x, adjust_layer])
    x = Activation('relu', name = name)(x)
    return x


def boosting_identity_block(input_tensor, hn_num, name = None):
    """
    """
    boost_input = Input(shape=(1,))
    adjust_layer = dense_bn_layer(input_tensor, hn_num)
    x = Activation('relu')(adjust_layer)
    x = dense_bn_act_layer(x, hn_num * 3 / 2)
    x = dense_bn_layer(x, hn_num)
    x = Add()([x, adjust_layer])
    x = Activation('relu', name = name)(x)
    return x


def res_net(input_shape, hns = [8, 6, 4, 4], classes = 2):
    """
    """
    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)
    x = identity_block(x, hns[0], name = 'block0', dropout = True)
    x = identity_block(x, hns[1], name = 'block1', dropout = True)
    x = identity_block(x, hns[2], name = 'block2', dropout = True)
    #x = identity_block(x, hns[3], name = 'block3', dropout = True)
    x = Dense(1, name = 'pre_sigmoid')(x)
    x = BatchNormalization()(x)
    proba = Activation('sigmoid')(x)
    model = Model(inputs, x)
    model.compile(optimizer=Nadam(lr = 0.001), loss='binary_crossentropy')

    return model


def boosting_dnn(input_shape, hns = [8, 6, 4, 7], classes = 2):
    """
    """
    inputs = Input(input_shape)
    boost_input = Lambda(lambda x: x[:, -1])(inputs)
    # dnn_input = Lambda(lambda x: x[:, :-1])(inputs)
    dnn_input = inputs
    #dnn_module
    # dnn_model = create_dnn((input_shape[0] - 1,), hns)
    dnn_model = create_dnn((input_shape[0],), hns)
    dnn_pre_sigmoid = Model(dnn_model.input, dnn_model.get_layer('pre_sigmoid').output)(dnn_input)
    # boost
    pre_sigmoid = Add(name = 'pre_sigmoid')([dnn_pre_sigmoid, boost_input])
    proba = Activation('sigmoid')(pre_sigmoid)

    model = Model(inputs, proba)
    model.compile(optimizer=Nadam(lr = 0.001), loss='binary_crossentropy')

    return model


def boosting_res_net(input_shape, hns = [128, 64, 16, 4], classes = 2, out_layer_name = None):
    """
    """
    inputs = Input(input_shape)
    boost_input = Lambda(lambda x: x[:, -1])(inputs)
    # res_module
    res_inputs = Lambda(lambda x: x[:, :-1])(inputs)
    res_model = res_net((input_shape[0] - 1, ), hns)
    #res_inputs = inputs
    #res_model = res_net(input_shape, hns)
    res_pre_sigmoid = Model(res_model.input, res_model.get_layer('pre_sigmoid').output)(res_inputs)
    # boost
    pre_sigmoid = Add(name = 'pre_sigmoid')([res_pre_sigmoid, boost_input])
    proba = Activation('sigmoid', name = out_layer_name)(pre_sigmoid)

    model = Model(inputs, proba)
    model.compile(optimizer=Nadam(lr = 0.001), loss='binary_crossentropy')

    return model


def rank_net(input_shape, hns = [6, 4, 4, 4], classes = 2):
    """
    """
    res_model = res_net((input_shape[1],), hns)
    res_model = Model(res_model.input, res_model.get_layer('pre_sigmoid').output)

    inputs = Input(input_shape)
    minor_inputs = Lambda(lambda x: x[:, 0], name = 'minor_input')(inputs)
    pred_minor = res_model(minor_inputs)
    minor_out_proba = Lambda(lambda x: x, name = 'minor_out_proba')(pred_minor)
    major_inputs = Lambda(lambda x: x[:, 1], name = 'major_input')(inputs)
    pred_major = res_model(major_inputs)
    major_out_proba = Lambda(lambda x: x, name = 'major_out_proba')(pred_major)

    sub = Subtract()([major_out_proba, minor_out_proba])
    sub = Lambda(lambda x: x * RANK_SCALE, name = 'rank_scale_layer')(sub)
    proba = Activation('sigmoid')(sub)

    model = Model(inputs, proba)
    # model.compile(optimizer=Nadam(lr = 0.0005), loss=min_pred)
    model.compile(optimizer=Nadam(lr = 0.001), loss='binary_crossentropy')

    return model


def ll_rank_net(input_shape, hns = [128, 64, 4, 4], classes = 2):
    """
    """
    # res_model = boosting_dnn((input_shape[1],), hns)
    res_model = create_dnn((input_shape[1],), hns)
    # res_model = boosting_res_net((input_shape[1],), hns)
    # res_model = create_dnn((input_shape[1],), hns)
    # res_model = res_net((input_shape[1],), hns)
    res_model = Model(res_model.input, res_model.get_layer('pre_sigmoid').output)
    inputs = Input(input_shape)

    minor_inputs = Lambda(lambda x: x[:, 0], name = 'minor_input')(inputs)
    pred_minor = res_model(minor_inputs)
    minor_pre_sigmoid = Lambda(lambda x: x, name = 'minor_pre_sigmoid')(pred_minor)
    minor_proba = Activation('sigmoid', name = 'minor_pred')(minor_pre_sigmoid)
    # minor_pred_loss = Lambda(lambda x: -1 * K.log(K.clip(x, EPSILON, 1)), name = 'minor_loss')(minor_proba)
    minor_pred_loss = Lambda(lambda x: 1 * (1 - x), name = 'minor_loss')(minor_proba)

    major_inputs = Lambda(lambda x: x[:, 1], name = 'major_input')(inputs)
    pred_major = res_model(major_inputs)
    major_pre_sigmoid = Lambda(lambda x: x, name = 'major_pre_sigmoid')(pred_major)
    major_proba = Activation('sigmoid', name = 'major_pred')(major_pre_sigmoid)
    # major_pred_loss = Lambda(lambda x: -1 * K.log(K.clip(1 - x, EPSILON, 1)), name = 'major_loss')(major_proba)
    major_pred_loss = Lambda(lambda x: 1 * x, name = 'major_loss')(major_proba)

    # sub = Subtract()([minor_pre_sigmoid, major_pre_sigmoid])
    sub = Subtract()([minor_proba, major_proba])
    sub = Lambda(lambda x: x * RANK_SCALE, name = 'rank_scale_layer')(sub)
    sub = Lambda(lambda x: -1 * (x - R_RANK_GAMMA), name = 'r_rank_gamma_layer')(sub)
    # rank_proba = Activation('sigmoid')(sub)
    rank_proba = Activation('relu')(sub)
    rank_loss = Lambda(lambda x: x ** R_RANK_P, name = 'rank_loss')(rank_proba)
    # rank_loss = Activation('tanh')(rank_proba)
    # rank_loss = Lambda(lambda x: -1 * K.log(K.clip(x, EPSILON, 1)), name = 'rank_loss')(rank_proba)

    loss = Add()([minor_pred_loss, rank_loss, major_pred_loss])
    model = Model(inputs, rank_loss)
    model.compile(optimizer=Nadam(), loss=min_pred)
    # model.compile(optimizer=Nadam(lr = 0.001), loss='binary_crossentropy')

    return model


def boosting_rank_net(input_shape, hns = [8, 6, 4, 4], classes = 2):
    """
    """
    res_model = boosting_res_net((input_shape[1],), hns, out_layer_name = 'proba')
    res_model = Model(res_model.input, res_model.get_layer('pre_sigmoid').output)

    inputs = Input(input_shape)
    minor_inputs = Lambda(lambda x: x[:, 0], name = 'minor_input')(inputs)
    pred_minor = res_model(minor_inputs)
    minor_out_proba = Lambda(lambda x: x, name = 'minor_out_proba')(pred_minor)
    major_inputs = Lambda(lambda x: x[:, 1], name = 'major_input')(inputs)
    pred_major = res_model(major_inputs)
    major_out_proba = Lambda(lambda x: x, name = 'major_out_proba')(pred_major)

    sub = Subtract()([major_out_proba, minor_out_proba])
    sub = Lambda(lambda x: x * RANK_SCALE, name = 'rank_scale_layer')(sub)
    proba = Activation('sigmoid')(sub)

    model = Model(inputs, proba)
    model.compile(optimizer=Nadam(lr = 0.001), loss=min_pred)

    return model


def boosting_parallel_res_net(input_shape, hns = [8, 6, 4, 7], classes = 2):
    """
    """
    boost_input = Input(shape=(1,))
    # res_module
    res_shape = (input_shape[0] - 1,)
    boost_res_net_model = boosting_res_net(input_shape)
    res_inputs = Input(shape = res_shape)

    boost_res_net_out_list = [boost_res_net_model([res_inputs, boost_input]) for i in range(8)]
    boost_res_net_out = concatenate(boost_res_net_out_list, axis = 1)

    x = Dense(4, activation = 'sigmoid')(boost_res_net_out)
    proba = Dense(1, activation = 'sigmoid')(x)
    model = Model([res_inputs, boost_input], proba)
    model.compile(optimizer=Nadam(lr = 0.001), loss='binary_crossentropy')

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


def create_dnn(input_shape, HIDDEN_UNITS = [16, 8, 4], DNN_BN = False, DROPOUT_RATE = 0):
    inputs = Input(input_shape)
    x = BatchNormalization()(inputs)
    x = dense_bn_act_layer(x, HIDDEN_UNITS[0], name = 'hn0', dropout = True)
    x = dense_bn_act_layer(x, HIDDEN_UNITS[1], name = 'hn1', dropout = True)
    # x = dense_bn_act_layer(x, HIDDEN_UNITS[2], name = 'hn2', dropout = True)
    x = Dense(1, name = 'pre_sigmoid')(x)
    proba = Activation('sigmoid')(x)
    model = Model(inputs, x)
    model.compile(optimizer=Nadam(), loss='binary_crossentropy')

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
