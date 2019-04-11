import gym, gym.spaces
import argparse
from reinforce import *
import tensorflow as tf


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_episodes', type=int, default=1e3)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()

def main():
    """
        Main script.
        Default environment: CartPole-v0
        Import default from .py
    """

    args = argparser()
    env = gym.make(args.env)
    with tf.Session() as sess:
        agent = REINFORCE(args, sess, env)
        print("Training agent...\n")
        agent.train()
        print("Training completed successfully.\nPrinting Results.\n")
        agent.print_results()

if __name__ == "__main__":
	main()
