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
        self.num_ep = args.num_episodes
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

        # Initialize graph
        self.policy = self.build_model()

    def build_model(self):
        """
            Neural Network model of the DQN agent
        """
        policy = MlpPolicy(self.sess,
                     self.obs,
                     self.net_sizes,
                     self.net_activations,
                     self.net_layer_types
                    )
        policy.print_model_summary()
        return policy

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
            self.policy.fit(state, target_new, verbose=0)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def train(self):
        """
            Train agent
        """
        for ep in range(int(self.num_ep)):
            paths = get_trajectories(self.env,
                                     agent=self.policy,
                                     max_transitions=self.batch_size,
                                     render=self.render)
            self.memory.add(paths)
            self.learn()
            print("Completed iteration: ", ep)

    def print_results(self):
        """
            Print the average reward and final Q-table
        """
        avg_rew = sum(self.rew_list)/self.num_ep
        print ("Score over time: " + str(avg_rew))
        plt.plot(self.rew_list)
        plt.show()


