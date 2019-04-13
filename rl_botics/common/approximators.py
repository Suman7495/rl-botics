from keras.models import Model
from keras.layers import Input, Dense, LSTM, Conv2D
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np

class MLP:
    """
    Multi-Layered Perceptron
    """
    def __init__(self, sess, input_ph, sizes, activations, layer_types, batch_size=None, scope='MLP'):
        """
        :param sess: Tensorflow session
        :param input_ph: Input tensor placeholder
        :param sizes: List of hidden layer sizes. e.g. sizes = [32, 32, output_dim]
        :param activations: Activations of each layer nodes. e.g. activations = ['tanh', 'tanh', 'tanh']
        :param scope: Name of the scope. e.g. 'Q'
        """
        self.sess = sess
        K.set_session(self.sess)
        assert len(sizes) == len(activations), "Layer sizes and activations mismatch"
        self.input = input_ph
        with tf.variable_scope(scope):
            output = Input(tensor=self.input)
            for l, nh in enumerate(sizes):
                if layer_types[l] == 'lstm':
                    self.input = Input(tensor=K.expand_dims(output, 0))
                    output = self.input
                    output = LSTM(nh, return_sequences=True)(output)
                elif layer_types[l] == 'conv':
                    output =Conv2D(nh, activation=activations[l], name=str(l))(output)
                else:
                    output = Dense(nh, activation=activations[l], name=str(l))(output)
            self.output = output
            self.model = Model(inputs=[self.input], outputs=[self.output])
        self.vars = tf.trainable_variables(scope=scope)

    def train_op(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
        self.train_op = self.optimizer.minimize(self.loss, var_list=self.vars)

    def get_trainable_vars(self):
        return self.sess.run(self.vars)

    def set_model_weights(self, weights):
        return 0

    def print_model_summary(self):
        print(self.model.summary())

    def predict(self, x, batch_size=None):
        feed_dict = {self.input: x}
        y = self.sess.run(self.output, feed_dict=feed_dict)
        return y