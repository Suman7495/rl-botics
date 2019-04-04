import hyperparameters as h
from rl_botics.common.data_collection import *
from rl_botics.common.policies import *
from rl_botics.common.utils import *
import tensorflow as tf


class REINFORCE:
    def __init__(self, args, sess, env):
        """
            Initialize REINFORCE agent class
        """
        self.sess = sess
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.render = args.render

        # Hyperparameters
        self.lr = args.lr
        self.gamma = args.gamma
        self.maxiter = 1000
        self.batch_size = args.batch_size
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

        # Initialize an empty reward list
        self.rew_list = []
        self.ep_rew_list = []
        
        # Initialize graph
        self._build_graph()
        self._init_sess()

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
        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim], name='obs')
        self.act = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='act')
        self.adv = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='adv')

        # Target for value function
        self.v_targ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='target_values')

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
         Loss graph
        """
        # Policy Loss
        self.loss = -tf.reduce_mean(tf.multiply(tf.exp(policy.log_prob, self.adv)

        # Policy update step
        self.pi_train_step = self.pi_optimizer.minimize(self.loss)

    def _init_sess(self):
        """
            Initialize tensorflow graph
        """
        self.sess.run(self.init)

    def process_paths(self, paths):
        """
            Process data

            :param paths: Obtain unprocessed data from training
            :return feed_dict: Dict required for neural network training
        """
        paths = np.asarray(paths)

        # Process paths
        obs = np.concatenate(paths[:, 0]).reshape(-1, self.obs_dim)
        new_obs = np.concatenate(paths[:, 3]).reshape(-1, self.obs_dim)
        act = paths[:, 1].reshape(-1,1)

        # Computed expected return, values and advantages
        expected_return = get_expected_return(paths, self.gamma, normalize=True)
        values = self.value.predict(obs)
        adv = expected_return-values

        # Generate feed_dict with data
        feed_dict = {self.obs: obs,
                     self.act: act,
                     self.adv: adv
                     }
        return feed_dict

    def update_policy(self, feed_dict):
        """
        :param feed_dict:
        """
        for _ in range(self.n_policy_epochs):
            self.sess.run(self.pi_train_step, feed_dict=feed_dict)

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

    def train(self):
        """
            Train using VPG algorithm
        """
        paths = get_trajectories(self.env, self.policy, self.render)
        dct = self.process_paths(paths)
        self.update_policy(dct)
        prev_dct = dct

        for itr in range(self.maxiter):
            paths = get_trajectories(self.env, self.policy, self.render)
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
            Plot reward received over training period
        """

