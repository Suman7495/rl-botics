from collections import deque
import random


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, paths):
        if self.memory:
            self.memory += paths
        else:
            self.memory = paths

    def sample(self):
        minibatch = random.sample(self.memory, self.batch_size)
        return minibatch

    def __len__(self):
        """
        :return: Current memory size
        """
        return len(self.memory)