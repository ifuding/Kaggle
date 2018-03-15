# Michael A. Alcorn (malcorn@redhat.com)
# A (slightly modified) implementation of the Recurrent Convolutional Neural Network (RCNN) found in [1].
# [1] Siwei, L., Xu, L., Kang, L., and Zhao, J. 2015. Recurrent convolutional
#         neural networks for text classification. In AAAI, pp. 2267-2273.
#         http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745

# import gensim
import numpy as np
import string
import os
from sklearn import metrics
import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed, SimpleRNN, \
        GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Activation, \
        SpatialDropout1D, Conv2D, Conv1D, Reshape, Flatten, AveragePooling2D, MaxPooling2D, Dropout, \
        MaxPooling1D, AveragePooling1D, Embedding
from tensorflow.python.keras.layers import concatenate
# from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import EarlyStopping, Callback
# from keras_train import RocAucEvaluation
# import vdcnn


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1, batch_interval = 1000000, verbose = 2, \
            scores = []):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
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
            print("Hi! on_batch_end() , batch=",batch,",logs:",logs)


class MySentences(object):
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for doc in self.corpus:
            # yield [str(word) for word in doc.split()]
            text = doc.strip().lower().translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
            yield text.split()


def get_word2vec_embedding(location = 'wv_model_norm.gensim', tokenizer = None, nb_words = 10000, \
                embed_size = 300, model_type = "fast_text"):
    """Returns trained word2vec

    Args:
        sentences: iterator for sentences

        location (str): Path to save/load word2vec
    """
    if not os.path.exists(location):
        print('Found {}'.format(location))
        return None
    print("-----Load Word2Vec Model-----")
    word_index = tokenizer.word_index
    if model_type == "word2vec":
        wv_model = gensim.models.KeyedVectors.load_word2vec_format(location, binary=True)
    elif model_type == "fast_text":
        wv_model = dict()
        with open(location, encoding="utf8") as emb_file:
            # with open("../../Data/normal_stem_wv_indata", "w+") as emb_file_indata:
            for line in emb_file:
                ls = line.strip().split(' ')
                word = ls[0]
                    # if word in word_index:
                    #     emb_file_indata.write(line)
                wv_model[word] = np.asarray(ls[1:], dtype='float32')
    print("word_index size: {0}".format(len(word_index)))
    embedding_matrix = np.zeros((nb_words, embed_size))
    word_in_corpus = 0
    for word, i in word_index.items():
        if i >= nb_words: continue
        if word in wv_model:
            embedding_matrix[i] = wv_model[word]
            word_in_corpus += 1
    print("{0} Words in corpus!".format(word_in_corpus))

    return embedding_matrix


class CNN_Model:
    """
    """
    def __init__(self, max_token, num_classes, context_vector_dim, hidden_dim, max_len, embedding_dim, \
                tokenizer, embedding_weight, batch_size, epochs, filter_size, fix_wv_model, batch_interval, \
                emb_dropout, full_connect_dropout, separate_label_layer, scores, resnet_hn, top_k):
        self.num_classes = num_classes
        self.context_vector_dim = context_vector_dim
        self.hidden_dim = hidden_dim
        self.max_token = max_token
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.embedding_weight = embedding_weight
        self.filter_size = filter_size
        self.fix_wv_model = fix_wv_model
        self.batch_size = batch_size
        self.epochs = epochs
        self.batch_interval = batch_interval
        self.emb_dropout = emb_dropout
        self.full_connect_dropout = full_connect_dropout
        self.separate_label_layer = separate_label_layer
        self.scores = scores
        self.resnet_hn = resnet_hn
        self.top_k = top_k
        self.model = self.Create_CNN()
        # self.model = vdcnn.build_model(num_filters = [64, 128, 256], sequence_max_length = self.max_len)


    def act_blend(self, linear_input):
        full_conv_relu = Activation('relu')(linear_input)
        return full_conv_relu
        full_conv_sigmoid = Activation('sigmoid')(linear_input)
        full_conv = concatenate([full_conv_relu, full_conv_sigmoid], axis = 1)
        return full_conv

    # k-max pooling (Finds values and indices of the k largest entries for the last dimension)
    def _top_k(self, x):
        x_shape = backend.int_shape(x)
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=self.top_k)
        return tf.reshape(k_max[0], (-1, x_shape[-1] * self.top_k))


    def pooling_blend(self, input):
        avg_pool = GlobalAveragePooling1D()(input)
        if self.top_k > 1:
            max_pool = Lambda(self._top_k)(input)
        else:
            max_pool = GlobalMaxPooling1D()(input)
        conc = concatenate([avg_pool, max_pool])
        return conc


    def partial_pooling_blend(self, input):
        avg_pool = AveragePooling1D(pool_size=3, strides=2)(input)
        max_pool = MaxPooling1D(pool_size=3, strides=2)(input)
        conc = concatenate([avg_pool, max_pool])
        return conc


    def pooling2d_blend(self, input, pool_size = (2, 2), strides = None, padding='valid'):
        avg_pool = AveragePooling2D(data_format = 'channels_last', pool_size = pool_size, \
                    strides = strides, padding = padding)(input)
        max_pool = MaxPooling2D(data_format = 'channels_last', pool_size = pool_size, \
                    strides = strides, padding = padding)(input)
        conc = concatenate([avg_pool, max_pool])
        return conc


    def full_connect_layer(self, input):
        full_connect = Dense(self.hidden_dim[0], activation = 'relu')(input)
        if self.full_connect_dropout > 0:
            full_connect = Dropout(self.full_connect_dropout)(full_connect)
        if self.resnet_hn:
            full_connect = concatenate([full_connect, input], axis = 1)
        return full_connect


    def ConvBlock(self, x, filter_size):
        kernel1_maps = Conv1D(filters = filter_size, kernel_size = 1, activation = 'linear')(x)
        kernel1_maps_act = self.act_blend(kernel1_maps)
        kernel1_conc = self.pooling_blend(kernel1_maps_act)

        kernel2_maps = Conv1D(filters = filter_size, kernel_size = 2, activation = 'linear')(x)
        kernel2_maps_act = self.act_blend(kernel2_maps)
        kernel2_conc = self.pooling_blend(kernel2_maps_act)

        kernel3_maps = Conv1D(filters = filter_size, kernel_size = 3, activation = 'linear')(x)
        kernel3_maps_act = self.act_blend(kernel3_maps)
        kernel3_conc = self.pooling_blend(kernel3_maps_act)

        kernel4_maps = Conv1D(filters = filter_size, kernel_size = 4, activation = 'linear')(x)
        kernel4_maps_act = self.act_blend(kernel4_maps)
        kernel4_conc = self.pooling_blend(kernel4_maps_act)

        kernel5_maps = Conv1D(filters = filter_size, kernel_size = 5, activation = 'linear')(x)
        kernel5_maps_act = self.act_blend(kernel5_maps)
        kernel5_conc = self.pooling_blend(kernel5_maps_act)

        kernel6_maps = Conv1D(filters = filter_size, kernel_size = 6, activation = 'linear')(x)
        kernel6_maps_act = self.act_blend(kernel6_maps)
        kernel6_conc = self.pooling_blend(kernel6_maps_act)

        kernel7_maps = Conv1D(filters = filter_size, kernel_size = 7, activation = 'linear')(x)
        kernel7_maps_act = self.act_blend(kernel7_maps)
        kernel7_conc = self.pooling_blend(kernel7_maps_act)

        conc = concatenate([kernel1_conc, kernel2_conc, kernel3_conc, kernel4_conc, \
                    kernel5_conc, kernel6_conc, kernel7_conc], axis = 1)
        return conc


    def Create_CNN(self):
        """
        """
        inp = Input(shape=(self.max_len, ))
        embedding = Embedding(self.max_token, self.embedding_dim, weights=[self.embedding_weight], trainable=not self.fix_wv_model)
        x = embedding(inp)
        if self.emb_dropout > 0:
            x = SpatialDropout1D(self.emb_dropout)(x)
        # x = Reshape(())
        # x = SpatialDropout1D(0.2)(x)
        have_cnn = False
        have_rnn = False
        for i in range(len(self.filter_size)):
            if self.filter_size[i] > 0:
                conc = self.ConvBlock(x, self.filter_size[i])
                have_cnn = True
                if i == 0:
                    cnn_conc = conc
                else:
                    cnn_conc = concatenate([conc, cnn_conc], axis = 1)

        for i in range(len(self.context_vector_dim)):
            if self.context_vector_dim[i] > 0:
                rnn_maps = Bidirectional(GRU(self.context_vector_dim[i], return_sequences=True))(x)
                conc = self.pooling_blend(rnn_maps)
                have_rnn = True
                if i == 0:
                    rnn_conc = conc
                else:
                    rnn_conc = concatenate([conc, rnn_conc], axis = 1)

        if have_cnn and have_rnn:
            conc = concatenate([cnn_conc, rnn_conc], axis = 1)
        elif have_rnn:
            conc = rnn_conc
        elif have_cnn:
            conc = cnn_conc

        # conc = self.pooling_blend(x)
        if self.separate_label_layer:
            for i in range(self.num_classes):
                full_connect = self.full_connect_layer(conc)
                proba = Dense(1, activation="sigmoid")(full_connect)
                if i == 0:
                    outp = proba
                else:
                    outp = concatenate([outp, proba], axis = 1)
        else:
            if self.hidden_dim[0] > 0:
                full_connect = self.full_connect_layer(conc)
            else:
                full_connect = conc
            # full_conv_0 = self.act_blend(full_conv_pre_act_0)
            # full_conv_pre_act_1 = Dense(self.hidden_dim[1])(full_conv_0)
            # full_conv_1 = self.act_blend(full_conv_pre_act_1)
            # flat = Flatten()(conc)
            outp = Dense(6, activation="sigmoid")(full_connect)

        model = Model(inputs = inp, outputs = outp)
        # print (model.summary())
        model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
        return model


    def Create_2DCNN(self):
        """
        """
        inp = Input(shape=(self.max_len, ))
        fixed_embedding = Embedding(self.max_token, self.embedding_dim, weights=[self.embedding_weight] , trainable=False)
        # retrain_embedding = Embedding(self.max_token, self.embedding_dim, weights=[self.embedding_weight] , trainable=True)
        fixed_x = fixed_embedding(inp)
        x = Bidirectional(GRU(self.context_vector_dim, return_sequences=True))(fixed_x)
        # retrain_x = retrain_embedding(inp)
        # x = Lambda(lambda x: backend.stack([x[0], x[1]], axis = 1))([fixed_x, retrain_x])
        # x = SpatialDropout1D(0.2)(x)
        x = Reshape((self.max_len, self.context_vector_dim * 2, 1))(x)

        # x = Conv2D(filters = self.filter_size, kernel_size = [3, 3], activation = 'relu', \
        #              data_format = 'channels_last', padding='same')(x)
        x1 = self.pooling2d_blend(x, pool_size = (10, 1), strides = None, padding = 'valid')
        x2 = self.pooling2d_blend(x, pool_size = (20, 1), strides = None, padding = 'valid')

        # x = Conv2D(filters = self.filter_size, kernel_size = [3, 3], activation = 'relu', \
        #              data_format = 'channels_last', padding='same')(x)
        # x = self.pooling2d_blend(x, pool_size = (2, 2))

        # x = Conv2D(filters = self.filter_size, kernel_size = [3, 3], activation = 'relu', \
        #              data_format = 'channels_last', padding='same')(x)
        # x = self.pooling2d_blend(x, pool_size = (5, 5))
        # kernel2_maps = Conv1D(filters = 50, kernel_size = 2, activation = 'linear')(x)
        # kernel2_maps_act = self.act_blend(kernel2_maps)
        # kernel2_conc = self.pooling_blend(kernel2_maps_act)

        # kernel3_maps = Conv1D(filters = 50, kernel_size = 3, activation = 'linear')(x)
        # kernel3_maps_act = self.act_blend(kernel3_maps)
        # kernel3_conc = self.pooling_blend(kernel3_maps_act)

        # kernel4_maps = Conv1D(filters = 50, kernel_size = 4, activation = 'linear')(x)
        # kernel4_maps_act = self.act_blend(kernel4_maps)
        # kernel4_conc = self.pooling_blend(kernel4_maps_act)

        conc = concatenate([x1, x2], axis = 1)

        # conc = self.pooling_blend(x)
        # full_conv_pre_act_0 = Dense(self.hidden_dim[0])(conc)
        # full_conv_0 = self.act_blend(full_conv_pre_act_0)
        # full_conv_pre_act_1 = Dense(self.hidden_dim[1])(full_conv_0)
        # full_conv_1 = self.act_blend(full_conv_pre_act_1)
        flat = Flatten()(conc)
        outp = Dense(6, activation="sigmoid")(flat)
        model = Model(inputs = inp, outputs = outp)
        print(model.summary())
        model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
        return model


    def train(self, train_part, train_part_label, valide_part, valide_part_label):
        """
        Keras Training
        """
        print("-----CNN training-----")

        # model = self.Create_2DCNN()

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
        print("-----CNN Test-----")
        pred = self.model.predict(test_part, batch_size=1024, verbose=verbose)
        return pred


if __name__ == '__main__':
    text = "This is some example text."
    text = text.strip().lower().translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    tokens = text.split()
    tokens = [word2vec.vocab[token].index if token in word2vec.vocab else MAX_TOKENS for token in tokens]

    doc_as_array = np.array([tokens])
    # We shift the document to the right to obtain the left-side contexts.
    left_context_as_array = np.array([[MAX_TOKENS] + tokens[:-1]])
    # We shift the document to the left to obtain the right-side contexts.
    right_context_as_array = np.array([tokens[1:] + [MAX_TOKENS]])

    target = np.array([NUM_CLASSES * [0]])
    target[0][3] = 1

    history = model.fit([doc_as_array, left_context_as_array, right_context_as_array], target, epochs = 1, verbose = 0)
    loss = history.history["loss"][0]
