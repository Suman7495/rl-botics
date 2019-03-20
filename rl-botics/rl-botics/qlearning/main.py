import argparse
import gym
from TabularQLearning import *


def main():
    """
        Main script.
        Default environment: FrozenLake-v0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='FrozenLake-v0')
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=0.8)
    parser.add_argument('--num_episodes', type=int, default=1e3)
    args = parser.parse_args()

    agent = TabularQLearning(args)
    print("Training agent...\n")
    agent.train()
    print("Training completed successfully.\nPrinting Results.\n")
    agent.print_results()

if __name__ == "__main__":
	main()
