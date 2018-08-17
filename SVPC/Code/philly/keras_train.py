
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

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed, SimpleRNN, \
        GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Activation, \
        SpatialDropout1D, Conv2D, Conv1D, Reshape, Flatten, AveragePooling2D, MaxPooling2D, Dropout, AlphaDropout, \
        MaxPooling1D, AveragePooling1D, Embedding, Concatenate, BatchNormalization, Multiply, Add
# from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import EarlyStopping, Callback
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam, Nadam
from tensorflow.python.keras.losses import mean_squared_error, binary_crossentropy

CATEGORY_FEATURES = [
    "user_id",
    "region","city","parent_category_name","category_name","user_type","image_top_1",
"param_1","param_2","param_3", "Weekday", "WeekdOfYear", "DayOfMonth", 
"item_seq_number", "desc_len", "title_len"
]

DENSE_FEATURES = [
"price"
    ]
USED_FEATURE_LIST = []

class RmseEvaluation(Callback):
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
            y_pred = self.model.predict(self.X_val, verbose=0, batch_size=10240)
            score = np.sqrt(metrics.mean_squared_error(self.y_val, y_pred))
            self.scores.append("epoch:{0} {1}".format(epoch + 1, score))
            print("\n RMSE - epoch: %d - score: %.6f \n" % (epoch+1, score))
    
    def on_batch_end(self, batch, logs={}):
        return
        if(self.verbose >= 2) and (batch % self.batch_interval == 0):
            # y_pred = self.model.predict(self.X_val, verbose=0)
            # loss = metrics.log_loss(self.y_val, y_pred)
            print("Hi! on_batch_end() , batch=",batch,",logs:",logs)
            # print("Valide size=",y_pred.shape[0], "  Valide loss=",loss)

class DNN_Model:
    """
    """
    def __init__(self, scores, cat_max, flags, emb_weight, model_type):
        self.hidden_dim = [int(hn.strip()) for hn in flags.full_connect_hn.strip().split(',')]
        self.batch_size = flags.batch_size
        self.epochs = flags.epochs
        self.batch_interval = flags.batch_interval
        self.emb_dropout = flags.emb_dropout
        self.full_connect_dropout = flags.full_connect_dropout
        self.emb_dim = [int(e.strip()) for e in flags.emb_dim.strip().split(',')]
        self.dense_input_len = len(USED_FEATURE_LIST)
        self.load_only_singleCnt = flags.load_only_singleCnt
        self.max_token = flags.vocab_size
        self.embedding_dim = flags.gram_embedding_dim
        self.fix_wv_model = flags.fix_wv_model
        self.filter_size = [int(hn.strip()) for hn in flags.filter_size.strip().split(',')]
        self.kernel_size_list = [int(kernel.strip()) for kernel in flags.kernel_size_list.strip().split(',')]
        self.rnn_units = [int(hn.strip()) for hn in flags.rnn_units.strip().split(',')]
        self.rnn_input_dropout = 0
        self.rnn_state_dropout = 0
        self.max_len = flags.max_len
        self.lgb_boost_dnn = flags.lgb_boost_dnn
        # self.max_title_len = flags.max_title_len
        self.top_k = 1

        self.scores = scores
        self.cat_max = cat_max
        self.emb_weight = emb_weight
        self.model_type = model_type
        self.model = self.create_model()


    def act_blend(self, linear_input):
        full_conv_selu = Activation('selu')(linear_input)
        full_conv_relu = Activation('relu')(linear_input)
        # return full_conv_relu
        full_conv_sigmoid = Activation('sigmoid')(linear_input)
        full_conv_tanh = Activation('tanh')(linear_input)
        full_conv = Concatenate()([full_conv_sigmoid, full_conv_relu, full_conv_selu])
        return full_conv


    def full_connect_layer(self, input):
        full_connect = input
        for hn in self.hidden_dim:
            fc_in = full_connect
            # full_connect = Dense(hn, kernel_regularizer = l2(0.001), activity_regularizer = l1(0.001))(full_connect)
            full_connect = Concatenate()([Dense(hn, kernel_initializer='lecun_uniform', activation = 'relu')(full_connect), 
                Dense(hn, kernel_initializer='lecun_uniform', activation = 'selu')(full_connect)])
            # full_connect = BatchNormalization()(full_connect)
            # full_connect = self.act_blend(full_connect)
            if self.full_connect_dropout > 0:
                full_connect = Dropout(self.full_connect_dropout)(full_connect) #Dropout(self.full_connect_dropout)(full_connect)
            # full_connect = Concatenate()([fc_in, full_connect])
        return full_connect


    def DNN_DataSet(self, data, sparse = True, dense = True):
        """
        input shape: batch * n_feature
        output shape: batch * [sparse0, spare1, ..., sparsen, dense_features]
        """
        if self.model_type == 'r':
            return np.reshape(data.values, (-1, self.dense_input_len, 1))
        else:
            return data.values

    def RNN_Target(self, data, label):
        """
        """
        return np.c_[data.values[:, 1:], label]

    def train(self, train_part, train_part_label, valide_part, valide_part_label):
        """
        Keras Training
        """
        print("-----DNN training-----")

        DNN_Train_Data = self.DNN_DataSet(train_part, sparse = False, dense = True)
        DNN_Valide_Data = self.DNN_DataSet(valide_part, sparse = False, dense = True)
        if self.model_type == 'r':
            train_part_label = None #self.RNN_Target(train_part, train_part_label)
            valide_part_label = None #self.RNN_Target(valide_part, valide_part_label)
        callbacks = [
                EarlyStopping(monitor='val_loss', patience=50, verbose=0),
                RmseEvaluation(validation_data=(DNN_Valide_Data, valide_part_label), interval=1, \
                    batch_interval = self.batch_interval, scores = self.scores)
                ]

        self.model.fit(DNN_Train_Data, train_part_label, batch_size=self.batch_size, epochs=self.epochs,
                    shuffle=True, verbose=2,
                    validation_data=(DNN_Valide_Data, valide_part_label)
                    , callbacks=callbacks
                    # , class_weight = {0: 1., 1: 5.}
                    )
        # print(self.model.get_weights())
        # exit(0)
        return self.model


    def predict(self, test_part, batch_size = 1024, verbose=2):
        """
        Keras Training
        """
        print("-----DNN Test-----")
        pred = self.model.predict(self.DNN_DataSet(test_part, sparse = False, dense = True), batch_size=10240, verbose=verbose)
        if self.model_type == 'r':
            pred = pred[:, -1]
        return pred


    def pooling_blend(self, input):
        avg_pool = GlobalAveragePooling1D()(input)
        if self.top_k > 1:
            max_pool = Lambda(self._top_k)(input)
        else:
            max_pool = GlobalMaxPooling1D()(input)
        conc = Concatenate()([avg_pool, max_pool])
        return conc


    def create_dense_model(self):
        """
        """
        dense_input = Input(shape=(self.dense_input_len,))
        # drop_dense_input = Dropout(self.full_connect_dropout)(dense_input)
        norm_dense_input = BatchNormalization()(dense_input)
        dense_output = self.full_connect_layer(norm_dense_input)
        proba = Dense(1, activation = 'softplus')(dense_output)

        model = Model(dense_input, proba) 
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model
    
    def create_rnn_model(self):
        """
        """
        seq_input = Input(shape=(self.dense_input_len, 1))
        seq_output = Input(shape=(self.dense_input_len, 1))
        # norm_seq_input = BatchNormalization(name = 'Dense_BN_trainable')(seq_input)
        rnn_out = Bidirectional(LSTM(self.rnn_units[0], return_sequences = True, activation = 'relu'))(seq_input)
        rnn_out = Bidirectional(LSTM(self.rnn_units[1], return_sequences = True, activation = 'relu'))(rnn_out)
        seq_pred = TimeDistributed(Dense(self.hidden_dim[0], activation = 'relu'))(rnn_out)
        seq_pred = TimeDistributed(Dense(1, activation = 'relu'))(seq_pred)
        # seq_pred = Dense(1, activation = 'relu')(rnn_out)
        seq_pred = Reshape((self.dense_input_len,))(seq_pred)
        seq_input_reshape = Reshape((self.dense_input_len,))(seq_input)

        model = Model(seq_input, seq_pred)
        loss = K.mean(mean_squared_error(seq_input_reshape[:, 1:], seq_pred[:, :-1]))
        model.add_loss(loss)

        # def _mean_squared_error(y_true, y_pred):
        #     return K.mean(K.square(y_pred - y_true))
        model.compile(optimizer='adam', loss = None) #_mean_squared_error)

        return model


    def create_boost_model(self):
        """
        """
        dense_input = Input(shape=(1,))
        norm_dense_input = BatchNormalization(name = 'Dense_BN_trainable')(dense_input)

        desc_seq = Input(shape=(self.max_len[0],))
        desc_cnn_conc = self.Create_CNN(desc_seq, name_suffix = '_desc')
        title_seq = Input(shape=(self.max_len[1],))
        title_cnn_conc = self.Create_CNN(title_seq, name_suffix = '_title')

        merge_input = Concatenate(name = 'merge_input_trainable')([norm_dense_input, \
            desc_cnn_conc, title_cnn_conc
        ])
        dense_output = self.full_connect_layer(merge_input)
        deep_pre_sigmoid = Dense(1, name = 'deep_pre_sigmoid_trainable')(dense_output)

        proba = Activation('sigmoid', name = 'proba_trainable')(deep_pre_sigmoid) #Add()([wide_pre_sigmoid, deep_pre_sigmoid]))

        model = Model([dense_input, desc_seq, title_seq], proba) 
        model.compile(optimizer='sgd', loss='mean_squared_error') #, metrics = ['accuracy'])

        # k_model = load_model('../Data/model_allSparse_09763.h5')
        # print (k_model.summary())
        # model.load_weights('../Data/model_allSparse_09763.h5', by_name=True)

        return model

    def create_model(self):
        """
        """
        # if self.load_only_singleCnt:
        #     return self.create_dense_model()
        # else:
        if self.lgb_boost_dnn:
            return self.create_boost_model()
        elif self.model_type == 'r':
            return self.create_rnn_model()
        else:
            return self.create_dense_model()


class VAE_Model:
    """
    """
    def __init__(self, flags):
        self.hidden_dim = [int(hn.strip()) for hn in flags.full_connect_hn.strip().split(',')]
        self.batch_size = flags.batch_size
        self.epochs = flags.epochs
        self.batch_interval = flags.batch_interval
        self.emb_dropout = flags.emb_dropout
        self.full_connect_dropout = flags.full_connect_dropout
        self.mse = flags.vae_mse
        self.original_dim = len(USED_FEATURE_LIST)
        self.intermediate_dim = flags.vae_intermediate_dim
        self.latent_dim = flags.vae_latent_dim
        self.model = self.create_model()
    
    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    def sampling(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def create_model(self):
        """
        """
        # VAE model = encoder + decoder
        # build encoder model
        input_shape = (self.original_dim, )
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(self.intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, name='z')([z_mean, z_log_var])

        # instantiate encoder model
        # encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        # print(encoder.summary())
        # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        # latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        # x = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        x = Dense(self.intermediate_dim, activation='relu')(z)
        outputs = Dense(self.original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        # decoder = Model(latent_inputs, outputs, name='decoder')
        # print(decoder.summary())
        # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        # outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        # VAE loss = mse_loss or xent_loss + kl_loss
        if self.mse:
            reconstruction_loss = mean_squared_error(inputs, outputs)
        else:
            reconstruction_loss = binary_crossentropy(inputs,
                                                    outputs)

        reconstruction_loss *= self.original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam', loss = None)
        # print (vae.summary())

        return vae

    def DNN_DataSet(self, data):
        """
        """
        return data.values


    def train(self, train_part, train_part_label, valide_part, valide_part_label):
        """
        Keras Training
        """
        print("-----DNN training-----")

        DNN_Train_Data = self.DNN_DataSet(train_part)
        DNN_Valide_Data = self.DNN_DataSet(valide_part)
        callbacks = [
                EarlyStopping(monitor='val_loss', patience=3, verbose=0),
                # RmseEvaluation(validation_data=(DNN_Valide_Data, valide_part_label), interval=1, \
                #     batch_interval = self.batch_interval, scores = self.scores)
                ]

        self.model.fit(DNN_Train_Data, batch_size=self.batch_size, epochs=self.epochs,
                    shuffle=True, verbose=2,
                    validation_data=(DNN_Valide_Data, None)
                    , callbacks=callbacks
                    # , class_weight = {0: 1., 1: 5.}
                    )
        # print(self.model.get_weights())
        # exit(0)
        return self.model


    def predict(self, test_part, batch_size = 1024, verbose=2):
        """
        Keras Training
        """
        print("-----DNN Test-----")
        pred = self.model.predict(self.DNN_DataSet(test_part), batch_size=10240, verbose=verbose)
        return pred


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