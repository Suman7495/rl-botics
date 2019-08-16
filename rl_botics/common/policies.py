import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from keras.models import model_from_json, model_from_yaml
from rl_botics.common.approximators import *
import random
import matplotlib.pyplot as plt
import time
from keras.models import Model
from keras import backend as K



class MlpSoftmaxPolicy(MLP):
    """
        General Softmax Policy with action logits from MLP output
    """
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
        with tf.variable_scope(scope):
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
        mean_entropy = self.sess.run(self.entropy, feed_dict=feed_dict)
        return mean_entropy

    def get_old_act_logits(self, obs):
        feed_dict = {self.obs: obs}
        old_act_logits = self.sess.run(self.act_logits, feed_dict=feed_dict)
        return old_act_logits


class ParametrizedSoftmaxPolicy(MlpSoftmaxPolicy):
    """
        Parametrized Softmax Policy. Used particularly for COPOS
    """
    def __init__(self, sess, obs, sizes, activations, layer_types, batch_size=None, scope='ParametrizedSoftmax'):
        super().__init__(sess, obs, sizes[:-1], activations[:-1], layer_types, batch_size, scope)

        with tf.variable_scope(scope):
            self.th = tf.get_variable(name="theta", shape=[sizes[-2], sizes[-1]],
                                         initializer=tf.random_uniform_initializer(-0.1, 0.1))


            # Action logits
            self.act_logits = tf.matmul(self.model.output, self.th, name="act_logits")
            # self.act_dist = tfp.distributions.Categorical(logits=self.act_logits)

            # Add action constraints
            self.constraints = tf.placeholder_with_default(input=tf.ones([1, sizes[-1]]), shape=[1, sizes[-1]])
            self.probs = tf.nn.softmax(self.act_logits, name="probs") * self.constraints
            self.act_dist = tfp.distributions.Categorical(probs=self.probs)

            # Sample action from the distribution
            self.sampled_action = self.act_dist.sample()

            # Utilities
            self.log_prob = tf.expand_dims(self.act_dist.log_prob(tf.squeeze(self.act, axis=-1)), axis=-1)
            self.actions = tf.placeholder(dtype=tf.float32, shape=[None, sizes[-1]])
            self.all_log_probs = tf.nn.log_softmax(self.act_logits)
            self.entropy = tf.reduce_mean(self.act_dist.entropy())

        self.theta = tf.trainable_variables(scope=scope)
        self.beta = self.vars

        # Variable Lengths
        self.theta_len = sizes[-2] * sizes[-1]

    def get_action_log_probs(self, obs, act_dim):
        feed_dict = {self.obs: obs}
        action_probs = self.sess.run(self.all_log_probs, feed_dict)
        return action_probs

    def pick_action(self, obs, show_distribution=False):
        # Add action constraint
        constraints = self._set_action_constraints(obs)
        feed_dict = {self.obs: np.atleast_2d(obs)}#, self.constraints: constraints}
        action = np.squeeze(self.sess.run(self.sampled_action, feed_dict=feed_dict))

        # Plot action distribution
        if show_distribution:
            plt.cla()
            probs = np.squeeze(self.sess.run(self.probs, feed_dict))
            barlist = plt.bar(np.linspace(1, probs.shape[0], probs.shape[0]), probs)
            barlist[action].set_color('r')
            plt.xlabel("Actions")
            plt.xticks(np.arange(1, probs.shape[0]+1))
            plt.ylabel("Probabilities")
            plt.ylim(0, 1)
            plt.title("Action Distribution")
            plt.pause(0.02)
            plt.show(block=False)
        # print(action)
        return action

    def pick_valid_action(self, obs, show_distribution=False):
        # Add action constraint
        feed_dict = {self.obs: np.atleast_2d(obs)}
        probs = np.squeeze(self.sess.run(self.probs, feed_dict=feed_dict))
        max_obj = 8
        cur_obs = obs[0, :4*max_obj].reshape(-1, 4)
        valid_move_actions = np.asarray(np.where(cur_obs[:, 0] != -10)[0]) + 1
        valid_remove_actions = valid_move_actions + max_obj
        valid_actions = np.concatenate((valid_move_actions, valid_remove_actions))
        valid_actions = np.insert(valid_actions, 0, 0)
        print(valid_actions)
        valid_probs = probs[valid_actions]
        valid_probs /= valid_probs.sum()
        action = np.squeeze(np.random.choice(valid_actions, size=1, p=valid_probs))
        return action


    def _set_action_constraints(self, obs):
        # Action Inhibitor
        max_obj = 10
        cur_obs = obs[:, :4*max_obj].reshape(-1, 4)
        invalid_move_actions = np.asarray(np.where(cur_obs[:, 1] == -10)) + 1
        tot_obj = max_obj - invalid_move_actions.shape[1]

        # assert invalid_move_actions.shape[0] == max_obj - tot_obj
        invalid_remove_actions = max_obj + invalid_move_actions
        # print(tot_obj, invalid_move_actions, invalid_remove_actions)

        invalid_actions = np.concatenate([invalid_move_actions, invalid_remove_actions])
        constraints = np.ones((1, 2*max_obj+1))
        if invalid_actions.any():
            constraints[:, invalid_actions] = 0
        return constraints


class RandPolicy:
    """ Random Policy """
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
    def __init__(self, sess, obs, sizes, activations, layer_types, batch_size=None, scope='MlpPolicy'):
        super().__init__(sess=sess,
                         input_ph=obs,
                         sizes=sizes,
                         activations=activations,
                         layer_types=layer_types,
                         batch_size=batch_size)
        self.sizes = sizes

    def pick_action(self, obs):
        obs = np.atleast_2d(obs)

        # Epsilon greedy exploration
        eps = 0.1
        if np.random.rand() < eps:
            return random.randrange(self.sizes[-1])
        act_probs = np.squeeze(self.predict(obs))
        action = np.argmax(act_probs)
        return int(action)
