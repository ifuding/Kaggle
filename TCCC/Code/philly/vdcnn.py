import tensorflow as tf

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Dense, Dropout, Lambda, Activation, BatchNormalization, \
        Flatten, Conv1D, Embedding, MaxPooling1D
from tensorflow.python.keras.optimizers import SGD
from CNN_Keras import RocAucEvaluation
from tensorflow.python.keras.callbacks import EarlyStopping, Callback

class ConvBlockLayer(object):
    """
    two layer ConvNet. Apply batch_norm and relu after each layer
    """

    def __init__(self, input_shape, num_filters):
        self.model = Sequential()
        # first conv layer
        self.model.add(Conv1D(filters=num_filters, kernel_size=3, strides=1, padding="same", input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        # second conv layer
        self.model.add(Conv1D(filters=num_filters, kernel_size=3, strides=1, padding="same"))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

    def __call__(self, inputs):
        return self.model(inputs)


def get_conv_shape(conv):
    return conv.get_shape().as_list()[1:]


class VDCNN_Model:
    """
    """
    def __init__(self, num_filters, sequence_max_length, hidden_dim, embedding_size, dense_dropout, \
                    batch_size, top_k, epochs):
        self.num_filters = num_filters
        self.sequence_max_length = sequence_max_length
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.dense_dropout = dense_dropout
        self.batch_size = batch_size
        self.top_k = top_k
        self.epochs = epochs
        self.model = self.build_model()


    def build_model(self):

        inputs = Input(shape=(self.sequence_max_length, ), dtype='int32', name='inputs')

        embedded_sent = Embedding(69, self.embedding_size, input_length=self.sequence_max_length)(inputs)

        # First conv layer
        conv = Conv1D(filters=64, kernel_size=3, strides=2, padding="same")(embedded_sent)

        # Each ConvBlock with one MaxPooling Layer
        for i in range(len(self.num_filters)):
            conv = ConvBlockLayer(get_conv_shape(conv), self.num_filters[i])(conv)
            conv = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

        # k-max pooling (Finds values and indices of the k largest entries for the last dimension)
        def _top_k(x):
            x = tf.transpose(x, [0, 2, 1])
            k_max = tf.nn.top_k(x, k=self.top_k)
            return tf.reshape(k_max[0], (-1, self.num_filters[-1] * self.top_k))
        k_max = Lambda(_top_k)(conv)

        # 3 fully-connected layer with dropout regularization
        if self.hidden_dim[0] > 0:
            fc1 = Dropout(self.dense_dropout)(Dense(self.hidden_dim[0], activation='relu', kernel_initializer='he_normal')(k_max))
        else:
            fc1 = k_max
        if self.hidden_dim[1] > 0:
            fc2 = Dropout(self.dense_dropout)(Dense(self.hidden_dim[1], activation='relu', kernel_initializer='he_normal')(fc1))
        else:
            fc2= fc1
        # fc3 = Dense(num_classes, activation='softmax')(fc2)
        # flatten = Flatten()(conv)
        fc3 = Dense(6, activation="sigmoid")(fc2)

        # define optimizer
        # sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=False)
        model = Model(inputs=inputs, outputs=fc3)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # if model_path is not None:
        #     model.load_weights(model_path)
        
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

        self.model.fit(train_part, train_part_label, batch_size=self.batch_size, epochs=self.epochs,
                    shuffle=True, verbose=2,
                    validation_data=(valide_part, valide_part_label)
                    , callbacks=callbacks)
        return self.model


    def predict(self, test_part, batch_size = 1024, verbose=2):
        """
        """
        print("-----CNN Test-----")
        pred = self.model.predict(test_part, batch_size=1024, verbose=verbose)
        return pred