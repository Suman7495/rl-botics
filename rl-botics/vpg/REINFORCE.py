import gym
import gym.spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import collections


class REINFORCE:
    def __init__(self, args, sess):
        """
            Initialize REINFORCE agent class
        """
        env = gym.make(args.env)
        self.sess = sess
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.render = args.render

        # Hyperparameters
        self.lr = args.lr
        self.gamma = args.gamma
        self.num_ep = args.num_episodes

        # Initialize an empty reward list
        self.rew_list = []
        self.ep_rew_list = []
        
        # Initialize graph
        self.model = self.build_model()
        self.sess.run(tf.global_variables_initializer())
        
    def build_model(self):
        """
            Neural Network model of the REINFORCE agent
        """
        hidden_dim = 2
        self.inputs = tf.placeholder(shape=[1, self.obs_dim],
                                     dtype=tf.float32)
        self.actions = tf.placeholder(dtype=tf.int32, name="action")
        self.discounted_rewards = tf.placeholder(dtype=tf.float32)

        W1 = tf.Variable(tf.random_uniform([self.obs_dim, hidden_dim]),
                                            dtype=tf.float32)
        b1 = tf.Variable([hidden_dim], dtype = tf.float32)
        a1 = tf.nn.tanh(tf.matmul(self.inputs, W1)+b1)

        W2 = tf.Variable(tf.random_uniform([hidden_dim, self.act_dim]),
                                            dtype=tf.float32)
        b2 = tf.Variable([self.act_dim], dtype=tf.float32)
        self.action_probs = tf.nn.softmax(tf.matmul(a1, W2)+ b2)

        self.picked_action_prob = tf.gather(self.action_probs[0],
                                            self.actions)

        self.loss = -tf.log(self.picked_action_prob) * self.discounted_rewards
        optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.updateModel = optimizer.minimize(self.loss)
        
    def discounted_rewards_norm(self):
        """
            Compute the discounted rewards.
            Note: Normalized rewards gives poorer performance
        """       
        discounted_rewards = np.zeros_like(self.ep_rew_list)
        cumulative = 0.0
        for i in reversed(range(len(self.ep_rew_list))):
            cumulative = cumulative * self.gamma + self.ep_rew_list[i]
            discounted_rewards[i] = cumulative        
        # discounted_rewards -= np.mean(discounted_rewards)
        # discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards
        
    def pick_action(self, state):
        """
            Choose action weighted by the action probability
        """
        action_prob = self.action_probs.eval(feed_dict
                                        ={self.inputs:state[None, :]})
        action = np.random.choice(np.arange(self.act_dim),
                                  p=action_prob[0])
        return action

    def train(self):
        """
            Train using REINFORCE algorithm
        """
        self.sess.run(tf.global_variables_initializer())
        transit= collections.namedtuple("transition",
                    ["state", "action", "reward", "next_state", "done"])
        for ep in range(int(self.num_ep)):
            state = self.env.reset()
            tot_rew = 0
            done = False
            t = 0
            episode = []
            self.ep_rew_list = []
            while t < 1000:
                t += 1
                if self.render and (ep%50 == 0):
                    self.env.render()

                act = self.pick_action(state)

                # Get next state and reward from environment
                next_state, rew, done, info = self.env.step(act)
    
                # Store transition in memory
                episode.append(transit(state=state, action=act,
                        reward=rew, next_state=next_state, done=done))

                # Store rewards
                self.ep_rew_list.append(rew)
                tot_rew += rew
                state = next_state
                if done:
                    print("\nEpisode: {}/{}, score: {}"
                          .format(ep, self.num_ep, t))
                    break

            self.rew_list.append(tot_rew)
            discounted_rewards = self.discounted_rewards_norm()
            for t, transition in enumerate(episode):
                _,loss = self.sess.run([self.updateModel, self.loss],
                                        feed_dict={self.inputs:[transition.state],
                                                   self.discounted_rewards:discounted_rewards[t],
                                                   self.actions:transition.action})

    def print_results(self):
        """
            Plot reward received over training period
        """
        avg_rew = sum(self.rew_list)/self.num_ep
        print ("Score over time: " + str(avg_rew))
        plt.plot(self.rew_list)
        plt.title("Returns")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.show()
