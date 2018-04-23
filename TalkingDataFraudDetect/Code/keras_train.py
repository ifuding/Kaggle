
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

from sklearn.cross_validation import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss

from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed, SimpleRNN, \
        GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Activation, \
        SpatialDropout1D, Conv2D, Conv1D, Reshape, Flatten, AveragePooling2D, MaxPooling2D, Dropout, \
        MaxPooling1D, AveragePooling1D, Embedding, Concatenate, BatchNormalization
# from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, Callback
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam, Nadam


RANK_SCALE = 1
DROPOUT_RATE = 0.5 #0.35
EPSILON = 1e-7
L2_NORM = 0
R_RANK_GAMMA = 0.1
R_RANK_P = 1
HIDDEN_UNITS = [4, 150, 100]
DNN_EPOCHS = 40
BATCH_SIZE = 64

SPARSE_FEATURES = {"app": {"max": 768, "emb": 5},
                   "channel": {"max": 500, "emb": 5},
                   "device": {"max": 4227, "emb": 5},
                   "ip": {"max": 364778, "emb": 5},
                   "os": {"max": 956, "emb": 5},
                   "hour": {"max": 23, "emb": 5},
                   # "day": {"max": 10, "emb": 4}
                   }
SPARSE_FEATURE_LIST = list(SPARSE_FEATURES.keys())
print ("SPARSE_FEATURE_LIST: {0}".format(SPARSE_FEATURE_LIST))

CATEGORY_FEATURES = ['app','device','os','channel', 'hour']

DENSE_FEATURE_LIST = [
'appAttrOverCnt','appchannelAttrOverCnt','appchannelhourAttrOverCnt','appdeviceAttrOverCnt','appdevicechannelAttrOverCnt',
'appdevicehourAttrOverCnt','appdeviceosAttrOverCnt','apphourAttrOverCnt','apposAttrOverCnt','apposchannelAttrOverCnt',
'apposhourAttrOverCnt','channelAttrOverCnt','deviceAttrOverCnt','deviceosAttrOverCnt','deviceoschannelAttrOverCnt',
'deviceoshourAttrOverCnt','hourAttrOverCnt','ipAttrOverCnt','ipappAttrOverCnt','ipappdeviceAttrOverCnt',
'ipapphourAttrOverCnt','ipapposAttrOverCnt','ipdeviceAttrOverCnt','ipdevicechannelAttrOverCnt','ipdevicehourAttrOverCnt',
'ipdeviceosAttrOverCnt','iphourAttrOverCnt','iposAttrOverCnt','osAttrOverCnt','oschannelAttrOverCnt','oshourAttrOverCnt',
'ipdayhour_channelCount','ipapp_channelCount','ipappos_channelCount','ipdaychannel_hourVar','ipappos_hourVar',
'ipappchannel_dayVar','ipappchannel_hourVar'
    ]
print ("DENSE_FEATURE_LIST: {0} {1}".format(len(DENSE_FEATURE_LIST), DENSE_FEATURE_LIST))
DATA_HEADER = ['ip'] + CATEGORY_FEATURES + ['day'] + DENSE_FEATURE_LIST
USED_FEATURE_LIST = CATEGORY_FEATURES + DENSE_FEATURE_LIST

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1, batch_interval = 1000000, verbose = 2, \
            scores = []):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        # print("y_val shape:{0}".format(self.y_val.shape))
        self.batch_interval = batch_interval
        self.verbose = verbose
        self.scores = scores

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = metrics.roc_auc_score(self.y_val, y_pred)
            self.scores.append("epoch:{0} {1}".format(epoch + 1, score))
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
    
    def on_batch_end(self, batch, logs={}):
        if(self.verbose >= 2) and (batch % self.batch_interval == 0):
            # y_pred = self.model.predict(self.X_val, verbose=0)
            # loss = metrics.log_loss(self.y_val, y_pred)
            print("Hi! on_batch_end() , batch=",batch,",logs:",logs)
            # print("Valide size=",y_pred.shape[0], "  Valide loss=",loss)

class DNN_Model:
    """
    """
    def __init__(self, hidden_dim, batch_size, epochs, batch_interval, emb_dropout, \
                full_connect_dropout, scores, emb_dim, load_only_singleCnt, dense_input_len):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.batch_interval = batch_interval
        self.emb_dropout = emb_dropout
        self.full_connect_dropout = full_connect_dropout
        self.scores = scores
        self.emb_dim = emb_dim
        self.dense_input_len = dense_input_len
        self.load_only_singleCnt = load_only_singleCnt
        self.model = self.create_model()


    def act_blend(self, linear_input):
        full_conv_relu = Activation('relu')(linear_input)
        return full_conv_relu
        full_conv_sigmoid = Activation('sigmoid')(linear_input)
        full_conv = concatenate([full_conv_relu, full_conv_sigmoid], axis = 1)
        return full_conv


    def full_connect_layer(self, input):
        full_connect = input
        for hn in self.hidden_dim:
            full_connect = Dense(hn, activation = 'relu')(full_connect)
            if self.full_connect_dropout > 0:
                full_connect = Dropout(self.full_connect_dropout)(full_connect)
        return full_connect


    def train(self, train_part, train_part_label, valide_part, valide_part_label):
        """
        Keras Training
        """
        print("-----DNN training-----")

        callbacks = [
                EarlyStopping(monitor='val_loss', patience=3, verbose=0),
                RocAucEvaluation(validation_data=(valide_part, valide_part_label), interval=1, \
                    batch_interval = self.batch_interval, scores = self.scores)
                ]

        self.model.fit(train_part, train_part_label, batch_size=self.batch_size, epochs=self.epochs,
                    shuffle=True, verbose=2,
                    validation_data=(valide_part, valide_part_label)
                    , callbacks=callbacks)
        return self.model


    def predict(self, test_part, batch_size = 1024, verbose=2):
        """
        Keras Training
        """
        print("-----DNN Test-----")
        pred = self.model.predict(test_part, batch_size=1024, verbose=verbose)
        return pred


    def create_dense_model(self):
        """
        """
        dense_input = Input(shape=(self.dense_input_len,))
        norm_dense_input = BatchNormalization()(dense_input)
        dense_output = self.full_connect_layer(norm_dense_input)
        proba = Dense(1, activation = 'sigmoid')(dense_output)

        model = Model(dense_input, proba) 
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

        return model


    def create_embedding_model(self):
        """
        """
        sparse_emb_list = []
        sparse_input_list = []
        merge_input_len = 0
        i = 0
        for sparse_feature in SPARSE_FEATURES:
            sparse_input = Input(shape=(1,))
            sparse_input_list.append(sparse_input)
            max_id = SPARSE_FEATURES[sparse_feature]["max"]
            emb_dim = self.emb_dim[i] #SPARSE_FEATURES[sparse_feature]["emb"]
            i += 1
            sparse_embedding = Embedding(max_id + 1, emb_dim, input_length = 1)(sparse_input)
            sparse_embedding = Reshape((emb_dim,))(sparse_embedding)
            sparse_emb_list.append(sparse_embedding)
            merge_input_len += emb_dim
        
        # dense_input = Input(shape=(len(CONTINUOUS_COLUMNS),))
        merge_input = Concatenate()(sparse_emb_list)
        dense_output = self.full_connect_layer(merge_input)
        proba = Dense(1, activation = 'sigmoid')(dense_output)

        model = Model(sparse_input_list, proba) 
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

        return model

    def create_model(self):
        """
        """
        if self.load_only_singleCnt:
            return self.create_dense_model()
        else:
            return self.create_embedding_model()


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
    x = identity_block(x, hns[0], name = 'block0', dropout = False)
    x = identity_block(x, hns[1], name = 'block1', dropout = False)
    x = identity_block(x, hns[2], name = 'block2', dropout = False)
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


def create_dnn(input_shape, HIDDEN_UNITS = [16, 8, 4], DNN_BN = False, DROPOUT_RATE = 0):
    inputs = Input(input_shape)
    x = BatchNormalization()(inputs)
    x = dense_bn_act_layer(x, HIDDEN_UNITS[0], name = 'hn0', dropout = True)
    # x = dense_bn_act_layer(x, HIDDEN_UNITS[1], name = 'hn1', dropout = True)
    # x = dense_bn_act_layer(x, HIDDEN_UNITS[2], name = 'hn2', dropout = True)
    x = Dense(1, name = 'pre_sigmoid')(x)
    proba = Activation('sigmoid')(x)
    model = Model(inputs, proba)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model