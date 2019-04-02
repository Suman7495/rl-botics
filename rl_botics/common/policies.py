import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from rl_botics.common.approximators import *
import random


class MlpSoftmaxPolicy(MLP):
    def __init__(self, sess, obs, sizes, activations, layer_types, batch_size=None, scope='Softmax'):
        super().__init__(sess=sess,
                         input_ph=obs,
                         sizes=sizes,
                         activations=activations,
                         layer_types=layer_types,
                         batch_size=batch_size)
        self.sess = sess
        self.obs = obs
        self.input_ph = self.obs
        self.act = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        # self.act = tf.placeholder(dtype=tf.float32, shape=[None, sizes(-1)])
        self.batch_size = batch_size
        self.scope = scope

        # Get output from Neural Network and create Softmax Distribution
        self.act_logits = self.model.output
        self.act_dist = tfp.distributions.Categorical(logits=self.act_logits)
        self.sampled_action = self.act_dist.sample()

        # Utilities
        self.log_prob = tf.expand_dims(self.act_dist.log_prob(tf.squeeze(self.act, axis=-1)), axis=-1)
        self.entropy = tf.reduce_mean(self.act_dist.entropy())

    def pick_action(self, obs):
        feed_dict = {self.obs: np.atleast_2d(obs)}
        action = np.squeeze(self.sess.run(self.sampled_action, feed_dict=feed_dict))
        return action

    def get_log_prob(self, obs, act):
        feed_dict = {self.obs: obs, self.act: act}
        log_prob = self.sess.run(self.log_prob, feed_dict=feed_dict)
        return log_prob

    def get_entropy(self, obs):
        feed_dict = {self.obs: obs}
        entropy = self.sess.run(self.entropy, feed_dict=feed_dict)
        return entropy

    def get_old_act_logits(self, obs):
        feed_dict = {self.obs: obs}
        old_act_logits = self.sess.run(self.output, feed_dict=feed_dict)
        return old_act_logits


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


class MlpPolicy(MLP):
    """
        TODO: Broken currently. Modify.
    """
    def __init__(self, sess, input_dim, sizes, activations, layer_types, loss=None, optimizer=None, scope='MLP_Policy'):
        super().__init__(sess, input_dim, sizes, activations, layer_types, loss, optimizer)

    def pick_action(self, obs):
        obs = np.atleast_2d(obs)

        # Epsilon greedy exploration
        eps = 0.1
        if np.random.rand() < eps:
            return random.randrange(self.sizes[-1])
        act_probs = np.squeeze(self.predict(obs))
        action = np.argmax(act_probs)
        return action
