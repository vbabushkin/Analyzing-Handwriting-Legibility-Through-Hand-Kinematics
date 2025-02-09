import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Flatten, Conv1D, Dense, Dropout, Activation, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import set_session
import tensorflow.compat.v1 as tf_compat


@tf.keras.utils.register_keras_serializable()
class Attention(tf.keras.layers.Layer):
    def __init__(self, units, name=None, **kwargs):
        super(Attention, self).__init__(name=name, **kwargs)
        self.units = units
        self.W = Dense(units)
        self.V = Dense(1)

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({"units": self.units})
        return config

    def call(self, inputs):
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * inputs
        return tf.reduce_sum(context_vector, axis=1)


def create_model(input_shape, kernel, dropout, lr, loss_fn, num_classes):
    """
    Creates and compiles the CNN model with an Attention layer.
    """
    tf.keras.backend.clear_session()
    config = tf_compat.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf_compat.Session(config=config)
    set_session(sess)

    inputs = Input(shape=input_shape)
    x = Conv1D(128, kernel_size=kernel, activation='relu', padding='valid')(inputs)
    x = BatchNormalization()(Activation("relu")(x))
    x = Dropout(dropout)(x)

    x = Conv1D(32, kernel_size=kernel, padding='valid')(x)
    x = BatchNormalization()(Activation("relu")(x))
    x = Dropout(dropout)(x)

    x = Attention(32)(x)
    x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss_fn, metrics=['categorical_accuracy'])

    return model, sess
