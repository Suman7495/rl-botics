import argparse
from trpo import *
import tensorflow as tf


def main():
    """
        Main script
        Default environment: CartPole-v0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--gamma', type=float, default=.995)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_ep', type=int, default=1000)
    parser.add_argument('--cg_damping', type=float, default=1e-1)
    args = parser.parse_args()

    with tf.Session() as sess:
        agent = TRPO(args, sess)
        print("Training agent...\n")
        agent.train()
        agent.print_results()

if __name__ == '__main__':
    main()