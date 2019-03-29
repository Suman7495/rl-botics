import argparse
from qlearning import *
import hyperparameters as h


def argparser():
    """
    Input argument parser.
    Loads default hyperparameters from hyperparameters.py
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=h.env_name)
    parser.add_argument('--gamma', type=float, default=h.gamma)
    parser.add_argument('--lr', type=float, default=h.lr)
    parser.add_argument('--num_episodes', type=int, default=h.num_ep)
    return parser.parse_args()


def main():
    """
        Main script.
        Default environment: FrozenLake-v0
    """
    args = argparser()
    agent = TabularQLearning(args)
    print("Training agent...\n")
    agent.train()
    print("Training completed successfully.\nPrinting Results.\n")
    agent.print_results()


if __name__ == "__main__":
    main()
