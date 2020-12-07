import keras
import tensorflow as tf
from tensorflow.keras import backend as K


class RandomizedResponse(keras.layers.Layer):
    def __init__(self, p, auto=True, min_value=0, max_value=-1):
        super(RandomizedResponse, self).__init__()
        assert 0 <= p <= 1
        assert auto or min_value <= max_value
        self.p = float(p)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.auto = True if auto else False

    def build(self, input_shape):
        pass

    def call(self, X, training=True):
        if not training:
            return X
        min_value = self.min_value
        max_value = self.max_value
        x_shape = tf.shape(X)
        x_dtype = K.dtype(X)
        if self.auto:
            min_value = tf.math.reduce_min(X)
            max_value = tf.math.reduce_max(X)
        choice = tf.random.categorical(tf.math.log([[1 - self.p, self.p]]), tf.size(X))
        choice = tf.cast(tf.reshape(choice, x_shape), x_dtype)
        rands = tf.random.uniform(x_shape, minval=min_value, maxval=max_value, dtype=x_dtype)
        return (X * tf.abs(choice - 1)) + (choice * rands)
