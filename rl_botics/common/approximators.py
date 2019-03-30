from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np

class MLP:
    """
    Multi-Layered Perceptron
    """
    def __init__(self, sess, input_dim, sizes, activations, layer_types, loss=None, optimizer=None, scope='MLP'):
        """
        :param sess: Tensorflow session
        :param input_dim: Input dimension of the tensor
        :param sizes: List of hidden layer sizes. e.g. sizes = [32, 32, output_dim]
        :param activations: Activations of each layer nodes. e.g. activations = ['tanh', 'tanh', 'tanh']
        :param scope: Name of the scope. e.g. 'Q'
        """
        self.sess = sess
        self.input_dim = input_dim
        assert len(sizes) == len(activations)
        self.sizes = sizes
        self.model = Sequential()
        self.model.add(Dense(sizes[0], activation=activations[0], input_dim=self.input_dim))
        sizes = sizes[1:]
        for l, nh in enumerate(sizes):
            if layer_types[l] == 'rnn':
                ouput = self.model.add(LSTM(nh, return_sequence=True))
            elif layer_types[l] == 'conv':
                output = self.model.add(Conv(nh, activation=activations[l], name=str(l)))
            else:
                output = self.model.add(Dense(nh, activation=activations[l], name=str(l)))
        self.output = output
        self.vars = tf.trainable_variables(scope=scope)

        # Compile model
        if optimizer: self.optimizer = optimizer
        else: self.optimizer = Adam
        if loss: self.loss = loss
        else: self.loss = 'mse'
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        # Initialize model
        self.init = tf.initializers.global_variables()
        self.sess.run(self.init)

    def get_trainable_vars(self):
        return self.sess.run(self.vars)

    def set_model_weights(self, weights):
        return 0

    def print_model_summary(self):
        print(self.model.summary())

    def fit(self, x, y, verbose=0):
        self.model.fit(x, y, verbose=0)

    def predict(self, x, batch_size=None):
        return self.model.predict(x, batch_size)