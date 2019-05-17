import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(title,
         filename,
         xlabel=None,
         ylabel=None):
    with open(filename, 'r') as f:
        x = f.readlines()

    x = [(e.strip()) for e in x]
    x = x[1:]
    x = np.asarray([float(e) for e in x])
    x = pd.Series(x).rolling(10, min_periods=10).mean()
    plt.plot(x)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    print("Completed")

#
# plot("COPOS", '/tmp/rl_log.txt', 'Iterations', 'Average Reward')
# plot("Success", '/tmp/rl_success.txt', 'Iterations', 'Success Percentage')
# plot('Entropy', '/tmp/rl_ent.txt', 'Iterations', 'Mean Entropy')

# plot("COPOS - FVRS")