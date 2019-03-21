from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

class MLP:
    """
    Multi-Layered Perceptron
    """
    # TODO: introduce Conv and LSTM
    def __init__(self, sess, input_dim, sizes, activations, scope='MLP'):
        """
        :param sess: Tensorflow sesssion
        :param input_dim: Input dimension of the tensor
        :param sizes: List of hidden layer sizes. e.g. sizes = [32, 32, output_dim]
        :param activations: Activations of each layer nodes. e.g. activations = ['tanh', 'tanh', 'tanh']
        :param scope: Name of the scope. e.g. 'Q'
        """
        self.sess = sess
        self.input_dim = input_dim
        self.model = Sequential()
        self.model.add(Dense(sizes[0], activation=activations[0], input_dim=self.input_dim))
        for l, nh in enumerate(sizes):
            output = self.model.add(Dense(nh, activation=activations[l], name=str(l))

        self.output = output
        self.vars = tf.trainable_variables(scope=scope)