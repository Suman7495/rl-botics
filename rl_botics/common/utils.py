import numpy as np
import tensorflow as tf

def get_expected_return(paths, gamma, normal=True):
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
    if normal:
        normalize(g)
    return g


def normalize(x):
    """
    :param x: vector
    :return: normalized x
    """
    return (x - np.mean(x)) / np.std(x)


def save_model(fname="model.ckpt"):
    """
    Save trained model
    :param fname: File path with name
    :param sess: Tf Session
    :return:
    """
    saver = tf.train.Saver()
    saver.save(tf.get_default_session(), fname)


def load_model(fname, sess):
    """
    Load previously saved model
    :param fname:
    :return:
    """
    saver = tf.train.import_meta_graph(fname + '.meta')
    # saver.restore(tf.get_default_session(), fname)
    saver.restore(sess, fname)