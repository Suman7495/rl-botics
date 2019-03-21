import argparse
from trpo import *
import tensorflow as tf
from .hyperparameters as tf


def main():
    """
        Main script
        Default environment: CartPole-v0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--gamma', type=float, default=gamma)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--num_ep', type=int, default=num_ep)
    parser.add_argument('--cg_damping', type=float, default=cg_damping)
    args = parser.parse_args()

    agent = TRPO(args, sess)
    print("Training agent...\n")
    agent.train()
    agent.print_results()

if __name__ == '__main__':
    main()