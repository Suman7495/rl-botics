import gym, gym.spaces
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from utils import *


class COPOS:
    def __init__(self, args, sess):
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

        # Initialize empty reward list
        self.rew_list = []

        # Build policy model
        self.__init__placeholders()
        self.build_policy()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def __init__placeholders(self):
        """
            Define Tensorflow placeholders
        """
        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim], name='obs')
        self.act = tf.placeholder(dtype=tf.float32, shape=[None, self.act_dim], name='act')
        self.adv = tf.placeholder(dtype=tf.float32, shape=[None], name='adv')
        self.old_std = tf.placeholder(dtype=tf.float32, shape=[None, self.act_dim], name='old_std')
        self.old_mean = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim], name='old_mean')

    def build_policy(self):
        """
            Neural Network Model of the COPOS agent
        """
        # Create the neural network with output as mean
        self.mean = [1., -1]
        self.std = [1, 2.]
        self.act_dist = tfp.distributions.MultivariateNormalDiag(self.mean, self.std)

    def pick_action(self, state):
        """
            Choose an action
        """
        action = self.act_dist.sample().eval()
        return np.argmax(action)

    def store_memory(self, transition):
        return

    def loss(self):
        """
            Compute loss
        """
        # Log probabilities of new and old actions
        old_act_dist = tfp.distributions.MultivariateNormalDiag(self.old_mean, self.old_std)
        self.old_log_prob = tf.reduce_sum(old_act_dist.log_prob(self.act))
        self.log_prob = tf.reduce_sum(self.act_dist.log_prob(self.act))
        prob_ratio = tf.exp(self.log_prob - self.old_log_prob)

        # Surrogate Loss
        self.surrogate_loss = -tf.reduce_mean(prob_ratio*self.adv)

        # KL Divergence
        self.kl = tfp.distributions.kl_divergence(self.act_dist, old_act_dist)

        # Entropy
        self.entropy = tf.reduce_mean(self.act_dist.entropy())

    def train(self):
        """
            Train using COPOS algorithm -
        """


    def rollout(self, timestep_limit=1000):
        """
            Simulate the agent for fixed timesteps
        """
        data = deque(maxlen=timestep_limit)
        obs = self.env.reset()
        tot_rew = 0
        done = False
        for t in range(timestep_limit):
            t += 1
            if self.render and t % 50 == 0:
                self.env.render()
            action = self.pick_action(obs)
            new_obs, rew, done, info = self.env.step(action)

            # Store transition
            transition = deque((obs, action, rew, new_obs, done))
            data.append(transition)

            if done:
                print("Terminated after %s timesteps" % t)
                return data
            obs = new_obs

    def print_results(self):
        """
            Plot the results
        """
        return