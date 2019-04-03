import numpy as np


def get_expected_return(paths, gamma):
    """
    :param paths: raw paths from data_collections.py converted to np.array
    :param gamma: discount factor
    :return: Expected return G (column vector)
    """
    done = paths[:, -1]
    rew = paths[:, 2].reshape(-1, 1)
    g = np.zeros_like(rew)
    g_next = 0
    for k in reversed(range(len(rew))):
        if done[k]:
            g[k] = 0
        g[k] = rew[k] + gamma * g_next * (1. - done[k])
        g_next = g[k]
    return g