import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

class SoftmaxPolicy:
    def __init__(self, sess, input):
        self.sess = sess
        self.input = input

        # TODO: Complete the following
        self.act_logits
        self.act_dist = tfp.distributions.Categorical(logits=self.act_logits)
        self.ouput

        self.entropy

        self.act
        self.log_prob

    def pick_action(self):

    def get_log_prob(self):

    def get_entropy(self):

class RandPolicy:
    def __init__(self, sess, act_dim, std=1.0, name='pi'):
        self.sess = sess
        self.act_dim = act_dim
        self.std = std
        self.name = name

    def pick_action(self):
        return np.squeeze(np.random.normal(loc=0.0, scale=self.std, size=self.act_dim))