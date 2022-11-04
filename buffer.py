# Simple replay buffer class

import random
import numpy as np

# Basic replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size, *args):
        self.buffer_size = buffer_size
        self.write = -1
        self.buffer = []

        self.element_count = 0

    # Storing new experience
    def store(self, experience):
        self.write = (self.write + 1) % self.buffer_size
        if self.size() < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.write] = experience

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []

    # Sample random batch from the buffer
    def sample(self, batch_size, *args):
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        batch = [np.array(e) for e in list(zip(*batch))]

        return batch

