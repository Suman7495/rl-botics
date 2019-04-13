import numpy as np


def get_expected_return(paths, gamma, normalize=True):
    """
    :param paths: raw paths from data_collections.py converted to np.array
    :param gamma: discount factor
    :return: Normalized expected return g (column vector)
    """
    done = paths[:, -1]
    rew = paths[:, 2].reshape(-1, 1)
    g = np.zeros_like(rew)
    cumulative = 0.0
    for k in reversed(range(len(rew))):
        if done[k]:
            g[k] = 0
        cumulative = rew[k] + gamma * cumulative * (1.0 - done[k])
        g[k] = cumulative

    # Normalize
    g = np.float32(g)
    if normalize:
        g -= np.mean(g)
        g /= np.std(g)
    return g