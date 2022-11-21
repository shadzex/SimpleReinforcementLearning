# Simple replay buffer class

import random
import numpy as np

# Basic replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size, *args):
        self.buffer_size = buffer_size
        self.write = -1
        self.buffer = []

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

# Prioritized experience replay(Without SumTree implementation as in the paper)
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, beta):
        super(PrioritizedReplayBuffer, self).__init__(buffer_size)

        self.buffer_size = buffer_size
        self.indices = []
        self.priorities = []

        self.beta = beta

    def store(self, experience):
        priority, transition = experience

        self.write = (self.write + 1) % self.buffer_size
        if self.size() < self.buffer_size:
            self.indices.append(self.write)
            self.buffer.append(None)
            self.priorities.append(None)
        self.buffer[self.write] = transition
        self.priorities[self.write] = priority

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []

    def sample(self, batch_size, *args):
        batch = []
        sampling_probabilities = []

        self.beta = np.min([1., self.beta + 0.001])

        probabilities = np.divide(self.priorities, sum(self.priorities))

        if self.size() < batch_size:
            indices = np.random.choice(self.indices, self.size(), replace=False, p=probabilities)
        else:
            indices = np.random.choice(self.indices, batch_size, replace=False, p=probabilities)

        for idx in indices:
            sampling_probabilities.append(probabilities[idx])
            batch.append(self.buffer[idx])

        importance_weights = np.power(self.size() * np.array(sampling_probabilities), -self.beta)
        importance_weights /= importance_weights.max()

        batch = [np.array(e) for e in list(zip(*batch))]

        batch.append(indices)
        batch.append(importance_weights)

        return batch

    def update(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = p

