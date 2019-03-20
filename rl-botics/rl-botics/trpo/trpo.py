import gym, gym.spaces
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from .utils import *
from common import *

class TRPO:
    def __init__(self, args):
        """
        Initialize COPOS agent class
        """
        env = gym.make(args.env)
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.render = True

        # Hyperparameters
        self.lr = args.lr
        self.gamma = args.gamma
        self.num_ep = args.num_ep
        self.cg_damping = args.cg_damping

        # Initialize empty reward list
        self.rew_list = []

        # Build policy model
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        """Build Tensorflow graph"""
        self.g = tf.Graph()
        with self.g.as_default():
            self._init_placeholders()
            self._build_policy()
            self._build_value_function()
            self._loss()
            self.init = tf.global_variables_initializer()

    def _init_placeholders(self):
        """
            Define Tensorflow placeholders
        """
        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim], name='obs')
        self.act = tf.placeholder(dtype=tf.float32, shape=[None, self.act_dim], name='act')
        self.adv = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='adv')
        self.old_log_probs = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='old_log_probs')
        self.old_std = tf.placeholder(dtype=tf.float32, shape=[None, self.act_dim], name='old_std')
        self.old_mean = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim], name='old_mean')

    def _build_policy(self):
        """
            Build the policy
        """
        # Create the neural network with the Softmax function as output layer
        # TODO: Build policy
        output = MLP()
        self.pi = SoftmaxPolicy(self.sess, output)

        # TODO: set action bounds

        self.act_dist = tfp.distributions.MultivariateNormalDiag(self.mean, self.std)

    def _build_value_function(self):
        """
            Value function
        """


    def _loss_(self):
        """
            Compute loss
        """
        # Log probabilities of new and old actions
        prob_ratio = tf.exp(self.pi.log_prob - self.old_log_probs)

        # Surrogate Loss
        self.surrogate_loss = -tf.reduce_mean(prob_ratio*self.adv)

        # TODO: Finish this section

    def _init_session(self):
            """Launch TensorFlow session and initialize variables"""
            self.sess = tf.Session(graph=self.g)
            self.sess.run(self.init)

    def train(self):
        """
            Train using TRPO algorithm
        """


    def print_results(self):
        """
            Plot the results
        """
        return