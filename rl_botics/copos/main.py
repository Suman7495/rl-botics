import argparse
from copos import *
import tensorflow as tf
import hyperparameters as h
import gym, gym.spaces


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
    env = gym.make(args.env)
    with tf.Session() as sess:
        agent = COPOS(args, sess, env)
        print("Training agent...\n")
        agent.train()
        agent.print_results()

if __name__ == '__main__':
    main()