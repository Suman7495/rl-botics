import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
from collections import deque
from rl_botics.common.approximators import *
from rl_botics.common.data_collection import *
from rl_botics.common.policies import *
import hyperparameters as h
from replay_buffer import *


class DQN:
    def __init__(self, args, sess, env):
        """
            Initialize DQN agent class
        """
        self.sess = sess
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.render = args.render

        # Hyperparameters
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.eps
        self.min_epsilon = args.min_eps
        self.epsilon_decay = args.eps_decay
        self.maxiter = args.num_episodes
        self.iter = 0
        self.min_trans_per_iter = 512
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size

        # Policy network hyperparameters
        self.net_sizes = h.hidden_sizes + [self.act_dim]
        self.net_activations = h.activations
        self.net_layer_types = h.layer_types
        self.net_loss = h.loss
        self.net_optimizer = Adam(self.lr)

        # Replay Memory with capacity 2000
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Initialize an empty reward list
        self.rew_list = []

        # Build Tensorflow graph
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        """Build Tensorflow graph"""
        self._init_placeholders()
        self._build_policy()
        self._build_value_function()
        self._loss()
        self.init = tf.global_variables_initializer()

    def _init_placeholders(self):
        """
            Define Tensorflow placeholders
        """
        # Observations, actions, advantages
        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim], name='obs')
        self.act = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='act')
        self.adv = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='adv')
    def _build_policy(self):
        """
            Neural Network model of the DQN agent
        """
        self.policy = MlpPolicy(self.sess,
                     self.obs,
                     self.net_sizes,
                     self.net_activations,
                     self.net_layer_types
                    )
        self.policy.print_model_summary()

    def _build_value_function(self):
        return NotImplementedError

    def _loss(self):
        return NotADirectoryError

    def _init_session(self):
        """ Initialize tensorflow graph """
        self.sess.run(self.init)

    def learn(self):
        """
            Experience Replay
        """
        minibatch = self.memory.sample()
        for state, action, reward, next_state, done in minibatch:
            state = np.atleast_2d(state)
            next_state = np.atleast_2d(next_state)
            target = reward
            if not done:
                target += self.gamma * np.amax(self.policy.predict(next_state))
            target_new = self.policy.predict(state)
            target_new[0][action] = target
            # TODO: No fit function defined
            self.policy.model.fit(state, target_new, verbose=0)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def update_policy(self, feed_dict):
        return NotImplementedError

    def update_value(self, feed_dict):
        return NotImplementedError

    def process_paths(self, paths):
        feed_dict = {}
        return feed_dict

    def train(self):
        """
            Train using COPOS algorithm
        """
        paths = get_trajectories(self.env, self.policy, self.render, self.min_trans_per_iter)
        dct = self.process_paths(paths)
        self.update_policy(dct)
        prev_dct = dct

        for itr in range(int(self.maxiter)):
            self.iter += 1
            paths = get_trajectories(self.env, self.policy, self.render, self.min_trans_per_iter)
            dct = self.process_paths(paths)

            # Update Policy
            self.update_policy(dct)

            # Update value function
            self.update_value(prev_dct)

            # Update trajectories
            prev_dct = dct

            # TODO: Log data
        self.sess.close()

    def print_results(self):
        """
            Print the average reward and final Q-table
        """
        avg_rew = sum(self.rew_list)/self.iter
        print ("Score over time: " + str(avg_rew))
        plt.plot(self.rew_list)
        plt.show()


