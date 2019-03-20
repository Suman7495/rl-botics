import gym
import gym.spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque


class DQN:
    def __init__(self, args, sess):
        """
            Initialize DQN agent class
        """
        env = gym.make(args.env)
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.render = False
        print(self.act_dim)

        # Hyperparameters
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.min_epsilon = args.min_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.num_ep = args.num_episodes
        self.batch_size = args.batch_size

        # Replay Memory with capacity 2000
        self.memory = deque(maxlen=2000)

        # Initialize an empty reward list
        self.rew_list = []

        # Initialize graph
        self.model = self.build_model()

    def build_model(self):
        """
            Neural Network model of the DQN agent
        """
        model = Sequential()
        model.add(Dense(units=64, activation='relu', input_dim=self.obs_dim))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=self.act_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def store_memory(self, transition):
        """
            Store transition in replay memory
        """
        self.memory.append(transition)

    def replay(self):
        """
            Experience Replay
        """
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state))
            target_new = self.model.predict(state)
            target_new[0][action] = target
            self.model.fit(state, target_new, verbose=0)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def pick_action(self, state):
        """
            Choose epsilon-greedy action
        """
        if np.random.rand() < self.epsilon:
            return random.randrange(self.act_dim)
        action = self.model.predict(state)
        return np.argmax(action[0])

    def train(self):
        """
            Train using DQN algorithm
        """
        for ep in range(int(self.num_ep)):
            state = self.env.reset()
            state = np.reshape(state, [1, self.obs_dim])
            tot_rew = 0
            done = False
            t = 0
            # Iterate over timesteps
            while t < 1000:
                t += 1
                if self.render:
                    self.env.render()
                act = self.pick_action(state)

                # Get next state and reward from environment
                next_state, rew, done, info = self.env.step(act)
                next_state = np.reshape(next_state, [1, self.obs_dim])
                # Store transition in memory
                transition = deque((state, act, rew, next_state, done))
                self.store_memory(transition)
  
                tot_rew += rew
                state = next_state
                if done:
                    print("\nEpisode: {}/{}, score: {}"
                          .format(ep, self.num_ep, t))
                    break
                if len(self.memory) > self.batch_size:
                    self.replay()

            self.rew_list.append(tot_rew)

    def print_results(self):
        """
            Print the average reward and final Q-table
        """
        avg_rew = sum(self.rew_list)/self.num_ep
        print ("Score over time: " + str(avg_rew))
        plt.plot(rew_list)
        plt.show()
