import argparse
import tensorflow as tf
from dqn import *
import hyperparameters as h


def argparser():
    """
    Input argument parser.
    Loads default hyperparameters from hyperparameters.py
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    # General Parameters
    parser.add_argument('--env', type=str, default=h.env_name)
    parser.add_argument('--gamma', type=float, default=h.gamma)
    parser.add_argument('--lr', type=float, default=h.lr)
    parser.add_argument('--num_episodes', type=int, default=h.num_ep)

    # DQN Specific Parameters
    parser.add_argument('--eps', type=float, default=h.eps)
    parser.add_argument('--min_eps', type=float, default=h.min_eps)
    parser.add_argument('--eps_decay', type=float, default=h.eps_decay)
    parser.add_argument('--batch_size', type=int, default=h.batch_size)
    parser.add_argument('--render', type=bool, default=h.render)
    parser.add_argument('--buffer_size', type=int, default=h.buffer_size)
    return parser.parse_args()


def main():
    """
        Main script.
    """
    args = argparser()

    with tf.Session() as sess:
        agent = DQN(args, sess)
        print("Training agent...\n")
        agent.train()
        print("Training completed successfully.\nPrinting Results.\n")
        agent.print_results()


if __name__ == "__main__":
    main()
