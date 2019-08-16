import argparse
import tensorflow as tf
from rl_botics.envs.table_continuous import *
from rl_botics.envs.table_continuous_with_time import *
from rl_botics.common.utils import save_model
import rl_botics.copos.hyperparameters as h
from rl_botics.copos.copos import *
import os
import random

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=h.env_name)
    parser.add_argument('--gamma', type=float, default=h.gamma)
    parser.add_argument('--pi_lr', type=float, default=h.pi_lr)
    parser.add_argument('--maxiter', type=int, default=h.maxiter)
    parser.add_argument('--render', type=bool, default=h.render)
    parser.add_argument('--batch_size', type=int, default=h.pi_batch_size)
    parser.add_argument('--cg_damping', type=float, default=h.cg_damping)
    parser.add_argument('--kl_bound', type=float, default=h.kl_bound)
    parser.add_argument('--ent_bound', type=float, default=h.ent_bound)
    parser.add_argument('--min_trans_per_iter', type=float, default=h.min_trans_per_iter)
    return parser.parse_args()


def main():
    """
        Main script
        Default environment: CartPole-v0
    """
    args = argparser()
    max_clean = 3
    max_dirty = 4
    max_human = 1
    for seed in range(0, 10):
        tf.reset_default_graph()
        print("\nRunning seed: ", seed)
        random.seed()
        extension = str(max_clean) + "_" + str(max_dirty) + "_" + str(max_human) + "_seed_" + str(seed)
        f_ent = 'results/final/trpo_ent_' + extension + '.txt'
        f_succ = 'results/final/trpo_success_' + extension + '.txt'
        f_rew = 'results/final/trpo_rew_' + extension + '.txt'
        # env = gym.make(args.env)
        env = ContinuousTable(max_clean=max_clean,
                              max_dirty=max_dirty,
                              max_human=max_human,
                              partial=True,
                              noise=True,
                              hist_len=8,
                              obj_width=0.1
                              )
        # env = ContinuousTableWithTime()

        with tf.Session() as sess:
            agent = COPOS(args, sess, env, f_ent, f_succ, f_rew)
            print("Training agent...\n")
            agent.train()
            # agent.print_results()

            # Save trained policy
            dirname = "models"
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            save_model(os.getcwd() + "/" + dirname + "/model.ckpt")


if __name__ == '__main__':
    main()
