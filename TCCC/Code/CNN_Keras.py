# Michael A. Alcorn (malcorn@redhat.com)
# A (slightly modified) implementation of the Recurrent Convolutional Neural Network (RCNN) found in [1].
# [1] Siwei, L., Xu, L., Kang, L., and Zhao, J. 2015. Recurrent convolutional
#         neural networks for text classification. In AAAI, pp. 2267-2273.
#         http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745

import gensim
import numpy as np
import string
import os

from keras import backend
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed, SimpleRNN, \
        GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Activation, \
        SpatialDropout1D, Conv2D, Conv1D, Reshape, Flatten, AveragePooling2D, MaxPooling2D, Dropout
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras_train import RocAucEvaluation
import vdcnn

## DNN Param
DNN_EPOCHS = 5
BATCH_SIZE = 32

class MySentences(object):
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for doc in self.corpus:
            # yield [str(word) for word in doc.split()]
            text = doc.strip().lower().translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
            yield text.split()


def get_word2vec_embedding(location = 'wv_model_norm.gensim', tokenizer = None, nb_words = 10000, embed_size = 300):
    """Returns trained word2vec

    Args:
        sentences: iterator for sentences

        location (str): Path to save/load word2vec
    """
    if not os.path.exists(location):
        print('Found {}'.format(location))
        return None
    print("-----Load Word2Vec Model-----")
    wv_model = gensim.models.KeyedVectors.load_word2vec_format(location, binary=True)
    word_index = tokenizer.word_index
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
                tokenizer, embedding_weight):
        self.num_classes = num_classes
        self.context_vector_dim = context_vector_dim
        self.hidden_dim = hidden_dim
        self.max_token = max_token
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.embedding_weight = embedding_weight
        self.filter_size = 100
        # self.model = self.Create_CNN()
        self.model = vdcnn.build_model(num_filters = [64, 128, 256], sequence_max_length = self.max_len)


    def act_blend(self, linear_input):
        full_conv_relu = Activation('relu')(linear_input)
        return full_conv_relu
        full_conv_sigmoid = Activation('sigmoid')(linear_input)
        full_conv = concatenate([full_conv_relu, full_conv_sigmoid], axis = 1)
        return full_conv


    def pooling_blend(self, input):
        avg_pool = GlobalAveragePooling1D()(input)
        max_pool = GlobalMaxPooling1D()(input)
        conc = concatenate([avg_pool, max_pool])
        return conc

    def pooling2d_blend(self, input, pool_size = (2, 2), strides = None, padding='valid'):
        avg_pool = AveragePooling2D(data_format = 'channels_last', pool_size = pool_size, \
                    strides = strides, padding = padding)(input)
        max_pool = MaxPooling2D(data_format = 'channels_last', pool_size = pool_size, \
                    strides = strides, padding = padding)(input)
        conc = concatenate([avg_pool, max_pool])
        return conc


    def Create_CNN(self):
        """
        """
        inp = Input(shape=(self.max_len, ))
        embedding = Embedding(self.max_token, self.embedding_dim) #, weights=[self.embedding_weight] , trainable=True)
        x = embedding(inp)
        x = SpatialDropout1D(0.2)(x)
        # rnn_maps = Bidirectional(GRU(self.context_vector_dim, return_sequences=True))(x)
        # rnn_conc = self.pooling_blend(rnn_maps)
        # x = Reshape(())
        # x = SpatialDropout1D(0.2)(x)

        kernel1_maps = Conv1D(filters = self.filter_size, kernel_size = 1, activation = 'linear')(x)
        kernel1_maps_act = self.act_blend(kernel1_maps)
        kernel1_conc = self.pooling_blend(kernel1_maps_act)

        kernel2_maps = Conv1D(filters = self.filter_size, kernel_size = 2, activation = 'linear')(x)
        kernel2_maps_act = self.act_blend(kernel2_maps)
        kernel2_conc = self.pooling_blend(kernel2_maps_act)

        kernel3_maps = Conv1D(filters = self.filter_size, kernel_size = 3, activation = 'linear')(x)
        kernel3_maps_act = self.act_blend(kernel3_maps)
        kernel3_conc = self.pooling_blend(kernel3_maps_act)

        kernel4_maps = Conv1D(filters = self.filter_size, kernel_size = 4, activation = 'linear')(x)
        kernel4_maps_act = self.act_blend(kernel4_maps)
        kernel4_conc = self.pooling_blend(kernel4_maps_act)

        kernel5_maps = Conv1D(filters = self.filter_size, kernel_size = 5, activation = 'linear')(x)
        kernel5_maps_act = self.act_blend(kernel5_maps)
        kernel5_conc = self.pooling_blend(kernel5_maps_act)

        kernel6_maps = Conv1D(filters = self.filter_size, kernel_size = 6, activation = 'linear')(x)
        kernel6_maps_act = self.act_blend(kernel6_maps)
        kernel6_conc = self.pooling_blend(kernel6_maps_act)

        kernel7_maps = Conv1D(filters = self.filter_size, kernel_size = 7, activation = 'linear')(x)
        kernel7_maps_act = self.act_blend(kernel7_maps)
        kernel7_conc = self.pooling_blend(kernel7_maps_act)

        conc = concatenate([kernel1_conc, kernel2_conc, kernel3_conc, kernel4_conc, kernel5_conc, kernel6_conc, kernel7_conc])

        # conc = self.pooling_blend(x)
        # full_conv = Dense(self.hidden_dim[0], activation = 'relu')(conc)
        # full_conv = Dropout(0.2)(full_conv)
        # full_conv_0 = self.act_blend(full_conv_pre_act_0)
        # full_conv_pre_act_1 = Dense(self.hidden_dim[1])(full_conv_0)
        # full_conv_1 = self.act_blend(full_conv_pre_act_1)

        outp = Dense(6, activation="sigmoid")(conc)

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
                RocAucEvaluation(validation_data=(valide_part, valide_part_label), interval=1)
                ]

        self.model.fit(train_part, train_part_label, batch_size=BATCH_SIZE, epochs=DNN_EPOCHS,
                    shuffle=True, verbose=2,
                    validation_data=(valide_part, valide_part_label)
                    , callbacks=callbacks)
        return self.model


    def predict(self, test_part, batch_size=BATCH_SIZE, verbose=2):
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
