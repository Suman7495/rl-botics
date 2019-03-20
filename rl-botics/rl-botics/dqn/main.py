import argparse
import gym
from DQN import *
import tensorflow as tf


def main():
    """
        Main script.
        Default environment: CartPole-v0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_episodes', type=int, default=1e2)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--min_epsilon', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    with tf.Session() as sess:
        agent = DQN(args, sess)
        print("Training agent...\n")
        agent.train()
        print("Training completed successfully.\nPrinting Results.\n")
        agent.print_results()

if __name__ == "__main__":
	main()
