import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import scipy.optimize
from rl_botics.common.approximators import *
from rl_botics.common.data_collection import *
from rl_botics.common.policies import *
from rl_botics.common.utils import *
from rl_botics.common.plotter import *
import hyperparameters as h
from utils import *
from pprint import pprint

class COPOS:
    def __init__(self, args, sess, env):
        """
        Initialize COPOS class
        """
        self.sess = sess
        self.env = env
        try:
            self.obs_dim = self.env.observation_space.shape[0]
        except:
            self.obs_dim = self.env.observation_space.n
        self.act_dim = self.env.action_space.n


        print(self.act_dim)
        self.render = args.render
        self.env_continuous = False
        self.filename = 'COPOS_log.txt'
        open('/tmp/rl_log.txt', 'w').close()
        open('/tmp/rl_success.txt', 'w').close()
        open('/tmp/rl_ent.txt', 'w').close()


        # Hyperparameters
        self.gamma = args.gamma
        self.maxiter = args.maxiter
        self.cg_damping = args.cg_damping
        self.batch_size = args.batch_size
        self.min_trans_per_iter = args.min_trans_per_iter
        self.iter = 1
        self.ent_gain = np.linspace(1, 10, self.maxiter+1)

        # Constraints parameters
        self.kl_bound = args.kl_bound
        self.ent_bound = args.ent_bound
        self.eta = 1
        self.omega = 0.5
        self.init_eta = self.eta
        self.init_omega = self.omega

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
        self.act_log_probs = tf.placeholder(dtype=tf.float32, shape=[None, None], name='old_log_probs')

        # Target for value function.
        self.v_targ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='target_values')

        # COPOS specific placeholders
        # eta: log-linear parameters.
        # beta: neural network nonlinear parameters
        self.eta_ph = tf.placeholder(dtype=tf.float32, shape=[], name="eta_ph")
        self.omega_ph = tf.placeholder(dtype=tf.float32, shape=[], name="omega_ph")
        self.batch_size_ph = tf.placeholder(dtype=tf.float32, shape=[], name='batch_size_ph')
        self.mean_entropy_ph = tf.placeholder(dtype=tf.float32, shape=[], name='mean_entropy')

    def _build_policy(self):
        """
            Build Policy
        """
        self.policy = ParametrizedSoftmaxPolicy(self.sess,
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
        self.v_params = self.value.vars

        print("\nValue model: ")
        print(self.value.print_model_summary())

    def _loss(self):
        """
            Compute COPOS loss
        """
        # Log probabilities of new and old actions
        prob_ratio = tf.exp(self.policy.log_prob - self.old_log_probs)

        # Policy parameter
        # self.params = self.policy.vars
        self.params = self.policy.theta + self.policy.beta

        # Surrogate Loss
        self.surrogate_loss = -tf.reduce_mean(tf.multiply(prob_ratio, self.adv))
        self.pg = flatgrad(self.surrogate_loss, self.params)

        # KL divergence
        self.old_policy = tfp.distributions.Categorical(self.old_act_logits)
        self.kl = self.old_policy.kl_divergence(self.policy.act_dist)
        self.m_kl = tf.reduce_mean(self.kl)
        # Entropy
        self.entropy = self.policy.entropy
        self.old_entropy = self.old_policy.entropy()
        self.ent_diff = self.entropy - self.old_entropy

        # All losses
        self.losses = [self.surrogate_loss, self.kl, self.entropy]

        # Compute Gradient Vector Product and Hessian Vector Product
        self.shapes = [list(param.shape) for param in self.params]
        self.size_params = np.sum([np.prod(shape) for shape in self.shapes])
        self.flat_tangents = tf.placeholder(tf.float32, (self.size_params,), name='flat_tangents')

        # Define Compatible Value Function and Lagrangian
        self._comp_val_fn()
        self._dual()

        # Compute gradients of KL wrt policy parameters
        # grads = tf.gradients(self.kl, self.params)
        grads = tf.gradients(self.m_kl, self.params)
        tangents = unflatten_params(self.flat_tangents, self.shapes)

        # Gradient Vector Product
        gvp = tf.add_n([tf.reduce_sum(g * tangent) for (g, tangent) in zip(grads, tangents)])
        # Fisher Vector Product (Hessian Vector Product)
        self.hvp = flatgrad(gvp, self.params)

        # Update operations - reshape flat parameters
        self.flat_params = tf.concat([tf.reshape(param, [-1]) for param in self.params], axis=0)
        self.flat_params_ph = tf.placeholder(tf.float32, (self.size_params,))
        self.param_update = []
        start = 0
        assert len(self.params) == len(self.shapes), "Wrong shapes."
        for i, shape in enumerate(self.shapes):
            size = np.prod(shape)
            param = tf.reshape(self.flat_params_ph[start:start + size], shape)
            self.param_update.append(self.params[i].assign(param))
            start += size

        assert start == self.size_params, "Wrong shapes."

    def _comp_val_fn(self):
        """
            Compatible Value Function Approximation Graph
        """
        # Compatible Weights
        self.flat_comp_w = tf.placeholder(dtype=tf.float32, shape=[self.size_params], name='flat_comp_w')
        comp_w = unflatten_params(self.flat_comp_w, self.shapes)

        # Compatible Value Function Approximation
        self.v = tf.placeholder(tf.float32, shape=self.policy.act_logits.get_shape())

        # Get Jacobian Vector Product (df/dx)u with v as a dummy variable
        jacob_vec_prod = jvp(f=self.policy.act_logits, x=self.params, u=comp_w, v=self.v)
        expected_jvp = tf.reduce_mean(jacob_vec_prod)
        self.comp_val_fn = tf.squeeze(jacob_vec_prod) - expected_jvp

    def _dual(self):
        """
            Computation of the COPOS dual function
        """
        # self.ent_bound *= self.ent_gain[self.iter]
        sum_eta_omega = self.eta_ph + self.omega_ph
        inv_batch_size = 1 / self.batch_size_ph
        # inv_batch_size = 1
        self.dual = self.eta_ph * self.kl_bound + self.omega_ph * (self.ent_bound - self.mean_entropy_ph) + \
                    sum_eta_omega  * inv_batch_size * \
                    tf.reduce_sum(tf.reduce_logsumexp((self.eta_ph * self.act_log_probs + self.comp_val_fn) /
                    sum_eta_omega, axis=1))
        self.dual_grad = tf.gradients(ys=self.dual, xs=[self.eta_ph, self.omega_ph])

    def _init_session(self):
        """ Initialize tensorflow graph """
        self.sess.run(self.init)

    def get_flat_params(self):
        """
            Retrieve policy parameters
            :return: Flattened parameters
        """
        return self.sess.run(self.flat_params)

    def set_flat_params(self, params):
        """
            Update policy parameters.
            :param params: New policy parameters required to update policy
        """
        feed_dict = {self.flat_params_ph: params}
        self.sess.run(self.param_update, feed_dict=feed_dict)

    def update_policy(self, feed_dict):
        """
            Update policy parameters
            :param feed_dict: Dictionary to feed into tensorflow graph
        """
        def get_pg():
            return self.sess.run(self.pg, feed_dict)

        def get_hvp(p):
            feed_dict[self.flat_tangents] = p
            return self.sess.run(self.hvp, feed_dict) + self.cg_damping * p

        pg = get_pg()  # vanilla gradient
        if np.allclose(pg, 0):
            print("Got zero gradient. Not updating.")
            return

        # Obtain Compatible Weights w by Conjugate Gradient (alternative: minimise MSE which is more inefficient)
        w = cg(f_Ax=get_hvp, b=-pg)
        # self.trpo_update(w, feed_dict)
        self.copos_update(w, feed_dict)

    def trpo_update(self, stepdir, feed_dict):
        def get_pg():
            return self.sess.run(self.pg, feed_dict)

        def get_hvp(p):
            feed_dict[self.flat_tangents] = p
            return self.sess.run(self.hvp, feed_dict) + self.cg_damping * p

        def get_loss(params):
            self.set_flat_params(params)
            return self.sess.run(self.losses, feed_dict)
        pg = get_pg()
        prev_params = self.get_flat_params()
        loss_before = get_loss(prev_params)
        surr_before = np.mean(loss_before[0])

        step_size = 1.0
        shs = 0.5 * stepdir.dot(get_hvp(stepdir))
        lm = np.sqrt(shs / self.kl_bound)
        fullstep = stepdir / lm
        expected_improve_rate = -pg.dot(stepdir) / lm

        # Perform Linesearch to rescale update stepsize
        for itr in range(20):
            new_params = prev_params + fullstep * step_size
            surr_loss, kl, ent = get_loss(new_params)
            mean_kl = np.mean(kl)
            surr_loss = np.mean(surr_loss)
            improve = surr_loss - surr_before
            expected_improve = expected_improve_rate * step_size
            ratio = improve / expected_improve
            if mean_kl > self.kl_bound * 1.5:
                print("KL bound exceeded.")
            elif improve > 0:
                print("Surrogate Loss didn't improve")
            else:
                # Print Results
                print("\n---------Iter %d---------- \n"
                      "Avg Reward: %f             \n"
                      "SurrogateL  %f             \n"
                      "KL:         %f             \n"
                      "Entropy:    %f             \n"
                      "--------------------------" % (self.iter, self.avg_rew, surr_loss, mean_kl, np.mean(ent)))
                break
            step_size *= .5
        else:
            print("Failed to update. Keeping old parameters")
            self.set_flat_params(prev_params)

    def copos_update(self, w, feed_dict):
        """
        :param w: Weights
        :param feed_dict: Dictionary for TensorFlow
        """
        def get_mean_loss(params):
            self.set_flat_params(params)
            surr, kl, ent = self.sess.run(self.losses, feed_dict)
            return np.mean(surr), np.mean(kl), np.mean(ent)

        def get_dual(x):
            eta, omega = x
            error_return_val = 1e6, np.array([0., 0.])
            if (eta + omega < 0) or (eta == 0) or np.isnan(eta):
                print("Error in dual optimization! Got eta: ", eta)
                return error_return_val
            feed_dict[self.eta_ph] = eta
            feed_dict[self.omega_ph] = omega
            dual, dual_grad, comp_val_fn = self.sess.run([self.dual, self.dual_grad, self.comp_val_fn], feed_dict)
            return np.asarray(dual), np.asarray(dual_grad)

        def get_new_params():
            """ Return new parameters """
            new_theta = (eta * theta_old + w_theta) / (eta + omega)
            new_beta = beta_old + w_beta / eta
            new_theta_beta = np.concatenate((new_theta, new_beta))
            return new_theta_beta

        def check_constraints(n_params):
            """
            :param n_params: New parameters
            :return: Returns True if constraints are satisfied, otherwise False
            """
            sur_before, kl_before, ent_before = get_mean_loss(prev_params)
            sur, kl, ent = get_mean_loss(n_params)
            improve = sur - sur_before

            if 0 <= kl < self.kl_bound:
                if improve < 0:
                    return True
            return False

        # Get previous parameters
        prev_params = self.get_flat_params()
        theta_old = prev_params[0:self.policy.theta_len]
        beta_old = prev_params[self.policy.theta_len:]

        # Split compatible weights w in w_theta and w_beta
        w_theta = w[0:self.policy.theta_len]
        w_beta = w[self.policy.theta_len:]

        # Add to feed_dict
        feed_dict[self.flat_comp_w] = w
        feed_dict[self.v] = np.zeros((self.obs_dim, self.act_dim))

        # Solve constraint optimization of the dual to obtain Lagrange Multipliers eta, omega
        # Optimization 1
        x0 = np.asarray([1, 0.5])
        bounds = ((1e-12, None), (1e-12, None))
        res, eta, omega = optimize_dual(get_dual, x0, bounds)

        if res.success and not np.isnan(eta):
            params1 = None
            new_params = get_new_params()
            if check_constraints(new_params):
                self.eta = eta
                self.omega = omega
                params1 = new_params
                self.set_flat_params(params1)
        else:
            print("Failed: Iteration %d. Cause: Optimization 1." %(self.iter))
            print(res.message)
            self.set_flat_params(prev_params)
            return

        # Optimization 2 (Binary search)
        # Optimize eta only
        surr_before, _, _ = get_mean_loss(prev_params)
        min_gain = 0.1
        max_gain = 10
        gain = 0.5 * (max_gain + min_gain)
        # gain = max_gain
        params2 = None
        for _ in range(15):
            # print(gain)
            cur_eta = gain * eta
            cur_theta = (cur_eta * theta_old + w_theta) / (cur_eta + omega)
            cur_beta = beta_old + w_beta / cur_eta
            new_params = np.concatenate([cur_theta, cur_beta])
            surr, kl, ent = get_mean_loss(new_params)
            # print(kl)
            improve = surr - surr_before
            if 0 <= kl < self.kl_bound:
                # print("KL success")
                if improve < 0:
                    # print("Binary success")
                    eta = cur_eta
                    self.eta = eta
                    params2 = new_params
                max_gain = gain
            else:
                min_gain = gain

            # Update eta then gain
            gain = 0.5 * (min_gain + max_gain)

        if params2 is not None:
            # print("Binary ")
            self.set_flat_params(params2)
        elif params1 is not None:
            print("Failed: Iteration %d. Cause: Binary Search. Updating approximate" % (self.iter))
            self.set_flat_params(params1)
            return
        else:
            print("Failed: Iteration %d. Cause: Binary Search. Performing TRPO update." % (self.iter))
            # self.set_flat_params(prev_params)
            self.trpo_update(w, feed_dict)
            return

        # Optimize 3
        # Optimize omega only
        x0 = np.asarray([self.eta, self.omega])
        eta_lower = np.max([self.eta - 1e-3, 1e-12])
        bounds = ((eta_lower, self.eta + 1e-3), (1e-12, None))
        res, eta, omega = optimize_dual(get_dual, x0, bounds, 1e-16)

        if res.success and not np.isnan(eta):
            params3 = get_new_params()
            if check_constraints(params3):
                self.eta = eta
                self.omega = omega
                print("Updating params 3")
                update_params = params3
            elif params2 is not None:
                print("Updating params 2")
                update_params = params2
            else:
                print("Updating params 1")
                update_params = params1

            surr, kl, ent = get_mean_loss(update_params)
            self.set_flat_params(update_params)

            # Print Results
            print("\n---------Iter %d---------- \n"
                  "Avg Reward: %f             \n"
                  "SurrogateL  %f             \n"
                  "KL:         %f             \n"
                  "Entropy:    %f             \n"
                  "Eta:        %f             \n"
                  "Omega:      %f             \n" 
                  "--------------------------" % (self.iter, self.avg_rew, surr, kl, ent, self.eta, self.omega))
            filename = '/tmp/rl_ent.txt'
            with open(filename, 'a') as f:
                f.write("\n%d" % (ent))

        else:
            print("Failed: Iteration %d. Cause: Optimization 2." % (self.iter))
            print(res.message)
            self.set_flat_params(prev_params)

    def update_value(self, prev_feed_dict):
        """
            Update value function
            :param prev_feed_dict: Processed data from previous iteration (to avoid overfitting)
        """
        # TODO: train in epochs and batches
        feed_dict = {self.obs: prev_feed_dict[self.obs],
                     self.v_targ: prev_feed_dict[self.adv]}
        self.v_train_step.run(feed_dict)

    def process_paths(self, paths):
        """
            Process data
            :param paths: Obtain unprocessed data from training
            :return: feed_dict: Dict required for neural network training
        """
        paths = np.asarray(paths)

        # Average episode reward for iteration
        tot_rew = np.sum(paths[:, 2])
        ep_count = np.sum(paths[:, -1])
        if ep_count:
            self.avg_rew = tot_rew / ep_count
        else:
            self.avg_rew = -100
        filename = '/tmp/rl_log.txt'
        with open(filename, 'a') as f:
            f.write("\n%d" % (self.avg_rew))
            # print("Average reward: ", self.avg_rew)

        # Process paths
        if self.obs_dim>1:
            obs = np.concatenate(paths[:, 0]).reshape(-1, self.obs_dim)
            new_obs = np.concatenate(paths[:, 3]).reshape(-1, self.obs_dim)
        else:
            obs = paths[:, 0].reshape(-1, self.obs_dim)
            new_obs = paths[:, 3].reshape(-1, self.obs_dim)
        act = paths[:, 1].reshape(-1, 1)

        # Computed expected return, values and advantages
        expected_return = get_expected_return(paths, self.gamma)
        values = self.value.predict(obs)
        adv = expected_return-values

        # Get action log probs
        action_log_probs = self.policy.get_action_log_probs(obs, self.act_dim)

        # Batch entropy
        mean_ent = self.policy.get_entropy(obs)

        # Generate feed_dict with data
        feed_dict = {self.obs: obs,
                     self.act: act,
                     self.adv: adv,
                     self.old_log_probs: self.policy.get_log_prob(obs, act),
                     self.old_act_logits: self.policy.get_old_act_logits(obs),
                     self.policy.act: act,
                     self.batch_size_ph: paths.shape[0],
                     self.mean_entropy_ph: mean_ent,
                     self.act_log_probs: action_log_probs}
        return feed_dict

    def train(self):
        """
            Train using COPOS algorithm
        """
        paths = get_trajectories(self.env, self.policy, self.render, self.min_trans_per_iter)
        dct = self.process_paths(paths)
        self.update_policy(dct)
        prev_dct = dct

        for itr in range(self.maxiter):
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
        print("Final eta and omega", self.eta, self.omega)
        self.sess.close()

    def print_results(self):
        """
            Plot the results
        """
        # TODO: Finish this section
        plot("COPOS", '/tmp/rl_log.txt', 'Iterations', 'Average Reward')
        plot("Success", '/tmp/rl_success.txt', 'Iterations', 'Success Percentage')
        plot('Entropy', '/tmp/rl_ent.txt', 'Iterations', 'Mean Entropy')
        return

