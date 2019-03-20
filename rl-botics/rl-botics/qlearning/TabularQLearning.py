import gym
import gym.spaces
import numpy as np


class TabularQLearning:
    def __init__(self, args):
        """
            Initialize class
        """
        env = gym.make(args.env)
        self.env = env
        self.obs_dim = self.env.observation_space.n
        self.act_dim = self.env.action_space.n
        self.lr = args.lr
        self.gamma = args.gamma
        self.num_ep = args.num_episodes

        # Initialize Q-table with zeros
        self.Q = np.zeros([self.obs_dim, self.act_dim])

        # Initialize an empty reward list
        self.rew_list = []

    def pick_action(self, s, idx):
        """
            Choose action greedily (with noise)
        """
        action = np.argmax(self.Q[s, :] + np.random.randn(1, self.act_dim)*(1./(idx+1)))
        return action

    def train(self):
        """
            Train using Q-learning algorithm
        """
        for ep in range(int(self.num_ep)):
            s = self.env.reset()
            tot_rew = 0
            done = False
            t = 0
            # The Q-Table learning algorithm
            while t < 99:
                t += 1
                a = self.pick_action(s, ep)

                # Get new state and reward from environment
                next_s, rew, done, info = self.env.step(a)

                # Update Q-Table
                self.Q[s, a] = self.Q[s, a] + self.lr*(rew + self.gamma*np.max(self.Q[next_s, :]) - self.Q[s, a])
                tot_rew += rew
                s = next_s
                if done:
                    break
            self.rew_list.append(tot_rew)

    def print_results(self):
        """
            Print the average reward and final Q-table
        """
        avg_rew = sum(self.rew_list)/self.num_ep
        print ("Score over time: " + str(avg_rew))
        print ("Final Q-Table: ")
        print (self.Q)
