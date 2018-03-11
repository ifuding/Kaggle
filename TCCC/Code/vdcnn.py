import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Dropout, Lambda, Activation, BatchNormalization, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import SGD


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


def build_model(num_classes = 2, num_filters = [64, 128, 256, 512], sequence_max_length=512, \
                num_quantized_chars=71, embedding_size=16, learning_rate=0.001, top_k=3, model_path=None):

    inputs = Input(shape=(sequence_max_length, ), dtype='int32', name='inputs')

    embedded_sent = Embedding(num_quantized_chars, embedding_size, input_length=sequence_max_length)(inputs)

    # First conv layer
    conv = Conv1D(filters=64, kernel_size=3, strides=2, padding="same")(embedded_sent)

    # Each ConvBlock with one MaxPooling Layer
    for i in range(len(num_filters)):
        conv = ConvBlockLayer(get_conv_shape(conv), num_filters[i])(conv)
        conv = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

    # k-max pooling (Finds values and indices of the k largest entries for the last dimension)
    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, num_filters[-1] * top_k))
    k_max = Lambda(_top_k, output_shape=(num_filters[-1] * top_k,))(conv)

    # 3 fully-connected layer with dropout regularization
    fc1 = Dropout(0.2)(Dense(512, activation='relu', kernel_initializer='he_normal')(k_max))
    fc2 = Dropout(0.2)(Dense(512, activation='relu', kernel_initializer='he_normal')(fc1))
    # fc3 = Dense(num_classes, activation='softmax')(fc2)
    # flatten = Flatten()(conv)
    fc3 = Dense(6, activation="sigmoid")(fc2)

    # define optimizer
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=False)
    model = Model(inputs=inputs, outputs=fc3)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if model_path is not None:
        model.load_weights(model_path)
    
    return model