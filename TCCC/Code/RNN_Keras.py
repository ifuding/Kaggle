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
        GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras_train import RocAucEvaluation


## DNN Param
DNN_EPOCHS = 2
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

class RNN_Model:
    """
    """
    def __init__(self, max_token, num_classes, context_vector_dim, hidden_dim, max_len, embedding_dim):
        self.num_classes = num_classes
        self.context_vector_dim = context_vector_dim
        self.hidden_dim = hidden_dim
        self.max_token = max_token
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.model = None

    def Create_RNN(self):
        """
        """
        inp = Input(shape=(self.max_len, ))
        x = Embedding(self.max_token, self.embedding_dim)(inp)
        # x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(GRU(self.context_vector_dim, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        full_conv = Dense(self.hidden_dim, activation="relu")(conc)
        outp = Dense(6, activation="sigmoid")(full_conv)

        model = Model(inputs = inp, outputs = outp)
        print (model.summary())
        model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
        return model


    def train(self, train_part, train_part_label, valide_part, valide_part_label):
        """
        Keras Training
        """
        print("-----RNN training-----")

        model = self.Create_RNN()

        callbacks = [
                EarlyStopping(monitor='val_loss', patience=3, verbose=0),
                RocAucEvaluation(validation_data=(valide_part, valide_part_label), interval=1)
                ]

        model.fit(train_part, train_part_label, batch_size=BATCH_SIZE, epochs=DNN_EPOCHS,
                    shuffle=True, verbose=2,
                    validation_data=(valide_part, valide_part_label)
                    , callbacks=callbacks)
        self.model = model
        return model


    def predict(self, model, test_part, batch_size=BATCH_SIZE, verbose=2):
        """
        Keras Training
        """
        print("-----RNN Test-----")
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
