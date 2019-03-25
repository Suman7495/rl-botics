import gym, gym.spaces
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
import matplotlib.pyplot as plt

from common import *
from .hyperparameters import *
from .utils import *

class TRPO:
    def __init__(self, args, sess):
        """
        Initialize COPOS agent class
        """
        self.sess = sess
        self.env = gym.make(args.env)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.render = True
        self.env_continuous = False

        # Hyperparameters
        self.lr = args.lr
        self.gamma = args.gamma
        self.num_ep = args.num_ep
        self.cg_damping = args.cg_damping
        self.kl_bound = args.kl_bound

        # Initialize empty reward list
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
        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim], name='obs')
        self.act = tf.placeholder(dtype=tf.float32, shape=[None, self.act_dim], name='act')
        self.adv = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='adv')

        # Placeholders below might not be required
        self.old_log_probs = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='old_log_probs')
        self.old_std = tf.placeholder(dtype=tf.float32, shape=[None, self.act_dim], name='old_std')
        self.old_mean = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim], name='old_mean')

    def _build_policy(self):
        """
            Build the policy
        """
        # Create the neural network with the Softmax function as output layer
        # TODO: Figure out how to introduce the session
        self.pi_net = MLP(self.obs_dim, pi_sizes, pi_activations, scope='policy')
        self.pi = SoftmaxPolicy(self.sess, self.pi_net)

    def _build_value_function(self):
        """
            Value function
        """
        return

    def _loss(self):
        """
            Compute loss
        """
        # Log probabilities of new and old actions
        prob_ratio = tf.exp(self.pi.log_prob - self.old_log_probs)

        # Surrogate Loss
        if self.env_continuous:
            self.params = self.pi_net.vars
        else:
            self.params = [self.pi_net.vars, self.pi.vars]
        self.surrogate_loss = -tf.reduce_mean(prob_ratio*self.adv)
        self.pg = flatgrad(self.surrogate_loss, self.params)

        # Compute Gradient Vector Product and Hessian Vector Product
        self.shapes = [param.shape.as_list() for param in self.params]
        self.size_params = np.sum([np.prod(shape) for shape in self.shapes])
        self.flat_tangents = tf.placeholder(tf.float32, (self.size_params,), name='flat_tangents')

        # Compute gradients of KL wrt policy parameters
        grads = tf.gradients(pi.kl, self.params)
        tangents = []
        start = 0
        for shape in self.shapes:
            size = np.prod(shape)
            tangents.append(tf.reshape(self.flat_tangents[start:start + size], shape))
            start += size

        # Gradient Vector Product
        gvp = tf.add_n([tf.reduce_sum(g * tangent) for (g, tangent) in zip(grads, tangents)])
        # Fisher Vector Product (Hessian Vector Product)
        self.hvp = flatgrad(gvp, self.params)

        # Update operations - reshape flat parameters
        # TODO: Make it into a function in util.py
        self.flat_params = tf.concat([tf.reshape(param, [-1]) for param in self.params], axis=0)
        self.flat_params_ph = tf.placeholder(dtype=tf.float32, (self.size_params,))
        self.param_update = []
        start = 0
        assert len(self.params) == len(self.shapes), "Wrong shapes."
        for i, shape in enumerate(self.shapes):
            size = np.prod(shape)
            param = tf.reshape(self.flat_params_ph[start:start + size], shape)
            self.param_update.append(self.params[i].assign(param))
            start += size

        assert start == self.size_params, "Wrong shapes."

        # TODO: Finish this section
        self.kl = kl(self.pi, self.old_pi)
        self.entropy =
        self.loss = self.surrogate_loss

    def _init_session(self):
            """Launch TensorFlow session and initialize variables"""
            self.sess.run(self.init)

    def get_flat_params(self):
        return self.sess.run(self.flat_params)

    def set_flat_params(self, params):
        feed_dict = {self.flat_params_ph: params}
        self.sess.run(self.param_update, feed_dict=feed_dict)

    def update(self, dct):
        """
            Update policy parameters
        """
        prev_params = self.get_flat_params()

        def get_pg():
            return self.sess.run(self.pg, dct)

        def get_hvp(p):
            dct[self.flat_tangents] = p
            return self.sess.run(self.hvp, dct) + self.cg_damping * p

        def get_loss(params):
            self.set_flat_params(params)
            return self.sess.run([self.loss, self.kl], dct)

        pg = get_pg()  # vanilla gradient
        if np.allclose(pg, 0):
            print("Got zero gradient. Not updating.")
            return
        stepdir = cg(f_Ax=get_hvp, b=-pg)  # natural gradient direction
        shs = 0.5 * stepdir.dot(get_hvp(stepdir))
        lm = np.sqrt(shs / self.kl_bound)  # optimal stepsize (see Eq 3-5 in https://arxiv.org/pdf/1703.02660.pdf)
        fullstep = stepdir / lm
        expected_improve = -pg.dot(stepdir) / lm
        success, new_params = linesearch(get_loss, prev_params, fullstep, expected_improve, self.kl_bound)
        self.set_flat_params(new_params)


    def train(self):
        """
            Train using TRPO algorithm
            Test Git 2.0
        """
        for itr in range(self.maxiter):
            trajectories = get_trajectories(self.env, agent=pi)

            # Update V
            # TODO: Complete section on Value

            # Estimate advantage
            # TODO: Complete section on Advantage Estimation

            # Old policy
            self.old_log_prob = pi.log_prob
            self.old_mean =
            self.old_std =

            # Update policy
            feed_dict = {}
            loss_before = self.sess.run()
            self.update(feed_dict)
            loss_after = self.sess.run()

            # Evaluate policy
            avg_rew =
            entropy =
            kl =

            # Log data

        self.sess.close()

    def print_results(self):
        """
            Plot the results
        """
        # TODO: Finish this section
        return
