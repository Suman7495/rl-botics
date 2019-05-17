import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from rl_botics.common.approximators import *
from rl_botics.common.data_collection import *
from rl_botics.common.policies import *
from rl_botics.common.utils import *
from rl_botics.common.plotter import *
import hyperparameters as h
from utils import *


class PPO:
    def __init__(self, args, sess, env):
        """
        Initialize PPO class
        """
        self.sess = sess
        self.env = env
        open('/tmp/rl_log.txt', 'w').close()
        try:
            self.obs_dim = self.env.observation_space.shape[0]
        except:
            self.obs_dim = self.env.observation_space.n
        self.act_dim = self.env.action_space.n

        if args.env == 'Rock-v0':
            self.obs_dim = 1
            self.act_dim = 5

        self.render = args.render
        self.env_continuous = False
        # self.logger = Logger(self.sess)

        # Hyperparameters
        self.gamma = args.gamma
        self.maxiter = args.maxiter
        self.cg_damping = args.cg_damping
        self.batch_size = args.batch_size
        self.kl_bound = args.kl_bound
        self.min_trans_per_iter = args.min_trans_per_iter

        # PPO specific hyperparameters
        self.kl_target = 0.003
        self.beta = 1
        self.beta_max = 20
        self.beta_min = 1 / 20
        self.ksi = 10
        self.n_policy_epochs = 20

        # Parameters for the policy network
        self.pi_sizes = h.pi_sizes + [self.act_dim]
        self.pi_activations = h.pi_activations + ['relu']
        self.pi_layer_types = h.pi_layer_types + ['dense']
        self.pi_batch_size = h.pi_batch_size
        self.pi_optimizer = tf.train.AdamOptimizer(learning_rate=h.pi_lr)

        # Parameters for the value network
        self.v_sizes = h.v_sizes
        self.v_activations = h.v_activations
        self.v_layer_types = h.v_layer_types
        self.v_batch_sizes = h.v_batch_sizes
        self.v_optimizer = tf.train.AdamOptimizer(learning_rate=h.v_lr)

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

        # Policy old log prob and action logits (ouput of neural net)
        self.old_log_probs = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='old_log_probs')
        self.old_act_logits = tf.placeholder(dtype=tf.float32, shape=[None, self.act_dim], name='old_act_logits')

        # Target for value function
        self.v_targ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='target_values')

        # PPO specific loss
        self.beta_ph = tf.placeholder(dtype='float32', shape=[], name='beta_2nd_loss')
        self.ksi_ph = tf.placeholder(dtype='float32', shape=[], name='eta_3rd_loss')

    def _build_policy(self):
        """
            Build Policy
        """
        self.policy = MlpSoftmaxPolicy(self.sess,
                                       self.obs,
                                       self.pi_sizes,
                                       self.pi_activations,
                                       self.pi_layer_types,
                                       self.pi_batch_size,
                                       )
        print("\nPolicy model: ")
        print(self.policy.print_model_summary())

    def _build_value_function(self):
        """
            Value function graph
        """
        self.value = MLP(self.sess,
                         self.obs,
                         self.v_sizes,
                         self.v_activations,
                         self.v_layer_types,
                         self.v_batch_sizes,
                         'value'
                         )
        self.v_loss = tf.losses.mean_squared_error(self.value.output, self.v_targ)
        self.v_train_step = self.v_optimizer.minimize(self.v_loss)

        print("\nValue model: ")
        print(self.value.print_model_summary())

    def _loss(self):
        """
            Compute PPO loss
        """
        # Log probabilities of new and old actions
        prob_ratio = tf.exp(self.policy.log_prob - self.old_log_probs)

        # Surrogate Loss
        self.surrogate_loss = -tf.reduce_mean(tf.multiply(prob_ratio, self.adv))

        # KL divergence
        self.old_policy = tfp.distributions.Categorical(self.old_act_logits)
        self.kl = self.old_policy.kl_divergence(self.policy.act_dist)
        # self.kl = self.policy.act_dist.kl_divergence(self.old_policy)

        # Loss terms
        loss_1 = self.surrogate_loss
        loss_2 = tf.reduce_mean(self.beta_ph * self.kl)
        loss_3 = tf.reduce_mean(self.ksi_ph * tf.square(tf.maximum(0.0, self.kl - 2 * self.kl_target)))

        # Compute adaptive kl loss
        self.loss = loss_1 + loss_2 + loss_3

        # Entropy
        self.entropy = self.policy.entropy

        # List of losses
        self.losses = [self.loss, self.kl, self.entropy]

        # Policy update step
        self.policy_train_op = self.pi_optimizer.minimize(self.loss)

    def _init_session(self):
        """
            Initialize tensorflow graph
        """
        self.sess.run(self.init)

    def update_policy(self, feed_dict):
        """
            Update policy parameters

            :param feed_dict: Dictionary to feed into tensorflow graph
        """
        for _ in range(self.n_policy_epochs):
            self.sess.run(self.policy_train_op, feed_dict=feed_dict)
            neg_policy_loss, kl, ent = self.sess.run(self.losses, feed_dict=feed_dict)
            mean_kl = np.mean(kl)
            if mean_kl > 4 * self.kl_target:
                break

        if mean_kl < self.kl_target / 1.5:
            self.beta /= 2
        elif mean_kl > self.kl_target * 1.5:
            self.beta *= 2
        self.beta = np.clip(self.beta, self.beta_min, self.beta_max)

    def update_value(self, prev_feed_dict):
        """
            Update value function

            :param prev_feed_dict: Processed data from previous iteration (to avoid overfitting)
        """
        # TODO: train in epochs and batches
        feed_dict = {self.obs: prev_feed_dict[self.obs],
                     self.v_targ: prev_feed_dict[self.adv]
                    }
        self.v_train_step.run(feed_dict)

    def process_paths(self, paths):
        """
            Process data

            :param paths: Obtain unprocessed data from training
            :return: feed_dict: Dict required for neural network training
        """
        paths = np.asarray(paths)

        # Average reward for iteration
        tot_rew = np.sum(paths[:, 2])
        ep_count = np.sum(paths[:, -1])
        avg_rew = tot_rew / ep_count
        filename = '/tmp/rl_log.txt'
        with open(filename, 'a') as f:
            f.write("\n%d" % (avg_rew))
            print("Average reward: ", avg_rew)

        # Process paths
        if self.obs_dim>1:
            obs = np.concatenate(paths[:, 0]).reshape(-1, self.obs_dim)
            new_obs = np.concatenate(paths[:, 3]).reshape(-1, self.obs_dim)
        else:
            obs = paths[:, 0].reshape(-1, self.obs_dim)
            new_obs = paths[:, 3].reshape(-1, self.obs_dim)

        act = paths[:, 1].reshape(-1,1)

        # Computed expected return, values and advantages
        expected_return = get_expected_return(paths, self.gamma, True)
        values = self.value.predict(obs)
        adv = expected_return-values

        # Log Data

        # Generate feed_dict with data
        feed_dict = {self.obs: obs,
                     self.act: act,
                     self.adv: adv,
                     self.old_log_probs: self.policy.get_log_prob(obs, act),
                     self.old_act_logits: self.policy.get_old_act_logits(obs),
                     self.policy.act: act,
                     self.beta_ph: self.beta,
                     self.ksi_ph: self.ksi
                     }
        return feed_dict

    def train(self):
        """
            Train using PPO algorithm
        """
        paths = get_trajectories(self.env, self.policy, self.render, self.min_trans_per_iter)
        dct = self.process_paths(paths)
        self.update_policy(dct)
        prev_dct = dct

        for itr in range(self.maxiter):
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
            Plot the results
        """
        # TODO: Finish this section
        plot("PPO")
        return
