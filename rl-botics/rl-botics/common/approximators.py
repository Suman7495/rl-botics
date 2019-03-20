from keras.models import Sequential
from keras.layers import Dense, Activation

class MLP:
    """
    Multi-Layered Perceptron
    """
    # TODO: complete generic neural network
    def __init__(self, sess, input, output, scope='MLP'):
        self.sess = sess
        self.input = input
        self.output = output
        # TODO: Get input and output dims
        self.in_dim = 5
        self.out_dim = 5

        self.scope = scope
        self.nn_output = []
        with tf.variable_scope(slf.scope):
            model = Sequential()
            model.add(Dense(32, activation='tanh', input_dim=self.obs_dim))
            model.add(Dense(32, activation='tanh'))
            self.nn_output.append(model.add(Dense(self.act_dim, activation='tanh')))

        self.vars = tf.trainable_variables(scope=scope)
