import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

class SoftmaxPolicy:
    def __init__(self, sess, obs, input):
        self.sess = sess
        self.obs = obs
        self.input = input

        # TODO: Complete the following
        self.act_logits = input
        self.act_dist = tfp.distributions.Categorical(logits=self.act_logits)
        self.sampled_action = self.act_dist.sample()

        # Utilities
        self.log_prob = self.act_dist.log_prob()
        self.mean = self.act_dist.mean()
        self.entropy = tf.reduce_mean(self.act_dist.entropy())

    def pick_action(self, obs):
        feed_dict = {self.obs: obs}
        action = self.sess.run(self.sampled_action, feed_dict=feed_dict)
        print(action)
        return action

    def get_log_prob(self):

    def get_entropy(self):

    def get_mean(self):


class RandPolicy:
    def __init__(self, sess, act_dim, std=1.0, name='pi'):
        self.sess = sess
        self.act_dim = act_dim
        self.std = std
        self.name = name

    def pick_action(self, obs):
        return np.squeeze(np.random.normal(loc=0.0, scale=self.std, size=self.act_dim))


class MultivariateGaussianPolicy:
    def __init__(self):
        self.name = 'MultiGaussPolicy'

    def pick_action(self, obs):
        return 0