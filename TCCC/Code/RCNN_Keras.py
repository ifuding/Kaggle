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
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed, SimpleRNN
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras_train import RocAucEvaluation


## DNN Param
DNN_EPOCHS = 1
BATCH_SIZE = 64

class MySentences(object):
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for doc in self.corpus:
            # yield [str(word) for word in doc.split()]
            text = doc.strip().lower().translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
            yield text.split()


def get_word2vec(corpus, location = 'wv_model_norm.gensim'):
    """Returns trained word2vec

    Args:
        sentences: iterator for sentences

        location (str): Path to save/load word2vec
    """
    if os.path.exists(location):
        print('Found {}'.format(location))
        model = gensim.models.Word2Vec.load(location)
        return model

    print('{} not found. training model'.format(location))
    model = gensim.models.Word2Vec(sentences = MySentences(corpus), size=32, window=5, min_count=5, workers=8)
    print('Model done training. Saving to disk')
    model.save(location)
    return model

class RCNN_Model:
    """
    """
    def __init__(self, wv_model_file, num_classes, context_vector_dim, hidden_dim, max_len):
        self.wv_model = gensim.models.Word2Vec.load(wv_model_file)
        self.num_classes = num_classes
        self.context_vector_dim = context_vector_dim
        self.hidden_dim = hidden_dim
        self.max_token = self.wv_model.wv.syn0.shape[0]
        self.embedding_dim = self.wv_model.wv.syn0.shape[1]
        self.max_len = max_len
        self.model = None
        print("wv_model shape: {0}".format(self.wv_model.wv.syn0.shape))

    def Create_RCNN(self):
        """
        """
        # word2vec = gensim.models.Word2Vec.load("wv_model_norm.gensim")
        ## We add an additional row of zeros to the embeddings matrix to represent unseen words and the NULL token.
        wv_array = self.wv_model.wv.syn0
        embeddings = np.zeros((self.max_token + 1, wv_array.shape[1]), dtype = "float32")
        embeddings[:self.max_token] = wv_array

        embedding_dim = self.embedding_dim #word2vec.syn0.shape[1]
        hidden_dim_1 = self.context_vector_dim #200
        hidden_dim_2 = self.hidden_dim #100
        # NUM_CLASSES = self.num_classes
        drop_out_rate = 0.8

        document = Input(shape = (None, ), dtype = "int32")
        left_context = Input(shape = (None, ), dtype = "int32")
        right_context = Input(shape = (None, ), dtype = "int32")

        embedder = Embedding(self.max_token + 1, embedding_dim) #, weights = [embeddings], trainable = False)
        # embedder = wv_model.wv.get_embedding_layer()
        doc_embedding = embedder(document)
        l_embedding = embedder(left_context)
        r_embedding = embedder(right_context)

        # I use LSTM RNNs instead of vanilla RNNs as described in the paper.
        forward = LSTM(hidden_dim_1, return_state = False)(l_embedding) # See equation (1).
        backward = LSTM(hidden_dim_1, return_state = False, go_backwards = True)(r_embedding) # See equation (2).
        # forward = SimpleRNN(hidden_dim_1, return_sequences = True, dropout = drop_out_rate, recurrent_dropout = drop_out_rate)(l_embedding) # See equation (1).
        # backward = SimpleRNN(hidden_dim_1, return_sequences = True, go_backwards = True, dropout = drop_out_rate, recurrent_dropout = drop_out_rate)(r_embedding) # See equation (2).
        # together = concatenate([forward, doc_embedding, backward], axis = 2) # See equation (3).
        together = concatenate([forward, backward], axis = 1) # See equation (3).

        semantic = Dense(hidden_dim_2, activation = "relu")(together)

        # semantic = TimeDistributed(Dense(hidden_dim_2, activation = "tanh"))(together) # See equation (4).

        # Keras provides its own max-pooling layers, but they cannot handle variable length input
        # (as far as I can tell). As a result, I define my own max-pooling layer here.
        # pool_rnn = semantic #Lambda(lambda x: backend.max(x, axis = 1), output_shape = (hidden_dim_2, ))(semantic) # See equation (5).

        output = Dense(6, input_dim = hidden_dim_2, activation = "sigmoid")(semantic) # See equations (6) and (7).

        model = Model(inputs = [document, left_context, right_context], outputs = output)
        model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
        return model


    def train(self, train_part, train_part_label, valide_part, valide_part_label):
        """
        Keras Training
        """
        print("-----RCNN training-----")

        model = self.Create_RCNN()

        X_train = self.gen_RCNN_Input(train_part)
        X_valid = self.gen_RCNN_Input(valide_part)

        callbacks = [
                # EarlyStopping(monitor='val_loss', patience=5, verbose=0),
                RocAucEvaluation(validation_data=(X_valid, valide_part_label), interval=1)
                ]

        model.fit(X_train, train_part_label, batch_size=BATCH_SIZE, epochs=DNN_EPOCHS,
                    shuffle=True, verbose=2,
                    validation_data=(X_valid, valide_part_label)
                    , callbacks=callbacks)
        self.model = model
        return model


    def predict(self, model, test_part, batch_size=BATCH_SIZE, verbose=2):
        """
        Keras Training
        """
        print("-----RCNN Test-----")
        Test = self.gen_RCNN_Input(test_part)
        pred = self.model.predict(Test, batch_size=BATCH_SIZE, verbose=verbose)
        return pred


    def gen_RCNN_Input(self, input_text):
        """
        """
        #tokenizer = Tokenizer(num_words = max_num_words)
        #tokenizer.fit_on_texts(input_text)
        # Pad the data
        MAX_TOKENS = self.wv_model.wv.syn0.shape[0]
        output_sequences = np.zeros((input_text.shape[0], self.max_len))
        print("RNN: Convert text to indice sequence!")
        ind = 0
        for tokens in MySentences(input_text):
            #text = text.strip().lower() # .translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
            #tokens = text.split(' ')[:max_len]
            tokens = tokens[:self.max_len]
            tokens = [self.wv_model.wv.vocab[token].index if token in self.wv_model.wv.vocab else MAX_TOKENS for token in tokens]

            doc_as_array = np.array([tokens])
            # print(doc_as_array.shape)
            output_sequences[ind, :doc_as_array.shape[1]] = doc_as_array
        #np.save('sequences', output_sequences)
        #exit(0)
        print(output_sequences.shape)
        # output_sequences = np.array(output_sequences)
        #output_sequences = [np.array(doc.split()[:max_len]) for doc in input_text] #text_to_word_sequence(input_text)
        # output_sequences = pad_sequences(output_sequences, maxlen= max_len)
        ## We shift the document to the right to obtain the left-side contexts.
        left_context_as_array = np.array([[self.max_token] + list(tokens[:-1]) for tokens in output_sequences])
        # left_context_as_array = np.array([np.array([' '] + list(tokens[:-1])) for tokens in output_sequences])
        # We shift the document to the left to obtain the right-side contexts.
        right_context_as_array = np.array([list(tokens[1:]) + [self.max_token] for tokens in output_sequences])
        # right_context_as_array = np.array([np.array(list(tokens[1:]) + [' ']) for tokens in output_sequences])
        return [output_sequences, left_context_as_array, right_context_as_array]


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
