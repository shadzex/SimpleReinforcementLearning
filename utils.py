import math
import numpy as np

epsilon = 1e-6

def calculate_conv_output(input_value, kernel_size, padding, stride):
    return math.floor(((input_value - kernel_size + (2 * padding)) / stride) + 1)

def calculate_conv2d_output(inputs, kernel_size, padding, stride):
    return calculate_conv_output(inputs[0], kernel_size[0], padding[0], stride[0]), calculate_conv_output(inputs[1], kernel_size[1], padding[1], stride[1])

def calculate_max_pool_output(input_value, kernel_size, padding, stride):
    return math.floor(((input_value - (kernel_size - 1) + (2 * padding) - 1) / stride) + 1)

def calculate_max_pool_2d_output(inputs, kernel_size, padding, stride):
    return calculate_max_pool_output(inputs[0], kernel_size[0], padding, stride[0]), calculate_max_pool_output(inputs[1], kernel_size[1], padding, stride[1])

def calculate_conv_transpose_output(input_value, kernel_size, padding, stride):
    return math.ceil((input_value - 1) * stride - 2 * padding + (kernel_size - 1) + 1)

def calculate_conv_transpose_2d_output(inputs, kernel_size, padding, stride):
    return calculate_conv_output(inputs[0], kernel_size[0], padding[0], stride[0]), calculate_conv_output(inputs[1], kernel_size[1], padding[1], stride[1])


def soft_update(target, source, tau=0.01):

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    target.load_state_dict(source.state_dict())

# Normalizer classes
# Will be used to normalize observations, goals and other inputs
# Disabled for visual observation
class BaseNormalizer:
    def __init__(self, dimension):
        self.dimension = dimension

    def update(self, value):
        raise NotImplementedError

    def normalize(self, value):
        raise NotImplementedError

    def states(self):
        raise NotImplementedError

    def load(self, states):
        raise NotImplementedError

# Normalizer that does nothing at all
class EmptyNormalizer(BaseNormalizer):
    def __init__(self, dimension):
        super(EmptyNormalizer, self).__init__(dimension)

    def update(self, value):
        return

    def normalize(self, value):
        return value

    def states(self):
        return []

    def load(self, states):
        return

# Standard normalization
class StandardNormalizer(BaseNormalizer):
    def __init__(self, dimension):
        super(StandardNormalizer, self).__init__(dimension)

        self.sum = np.zeros([1] + dimension, dtype=float)
        self.count = 0

        self.mean = np.zeros([1] + dimension, dtype=float)

        self.squared_sum = np.zeros([1] + dimension, dtype=float)
        self.std = np.ones([1] + dimension, dtype=float)

    def update(self, value):
        # value can have multiple batches
        count = np.shape(value)[0]
        sum = np.sum(value, axis=0, keepdims=True)

        self.count += count
        self.sum += sum

        self.mean = self.sum / self.count

        squared_sum = np.sum(np.square(value), axis=0, keepdims=True)
        self.squared_sum += squared_sum

        squared_mean = self.squared_sum / self.count
        var = squared_mean - np.square(self.mean)

        self.std = np.sqrt(var + epsilon)

    def normalize(self, value):
        # value's dimension: (B, dimension...)
        return (value - self.mean) / self.std

    def states(self):
        return self.sum, self.count, self.mean, self.squared_sum, self.std

    def load(self, states):
        sum, count, mean, squared_sum, std = states

        self.sum = sum.copy()
        self.count = count.copy()
        self.mean = mean.copy()
        self.squared_sum = squared_sum.copy()
        self.std = std.copy()

# Zero-centered, not using standard deviation
class ZeroCenteredNormalizer(BaseNormalizer):
    def __init__(self, dimension):
        super(ZeroCenteredNormalizer, self).__init__(dimension)

        self.sum = np.zeros([1] + dimension, dtype=float)
        self.count = 0

        self.mean = np.zeros([1] + dimension, dtype=float)

    def update(self, value):
        # value can have multiple batches
        count = np.shape(value)[0]
        sum = np.sum(value, axis=0, keepdims=True)

        self.count += count
        self.sum += sum

        self.mean = self.sum / self.count

    def normalize(self, value):
        # value's dimension: (B, dimension...)
        return value - self.mean

    def states(self):
        return self.sum, self.count, self.mean

    def load(self, states):
        sum, count, mean= states

        self.sum = sum.copy()
        self.count = count.copy()
        self.mean = mean.copy()

class Normalizer:
    def __init__(self, dimension, method):

        # If we use visual observation, we will not use normalization
        if len(dimension) == 3:
            self.normalizer = EmptyNormalizer(dimension)
        else:
            # If we use vector state, we will use normalization
            if method == 'zero_centered':
                self.normalizer = ZeroCenteredNormalizer(dimension)
            elif method == 'standard' or method == 'normalization':
                self.normalizer = StandardNormalizer(dimension)
            elif method == None:
                self.normalizer = EmptyNormalizer(dimension)
            else:
                raise NotImplementedError

    def update(self, value):
        self.normalizer.update(value)

    def normalize(self, value):
        return self.normalizer.normalize(value)

    def states(self):
        return self.normalizer.states()

    def load(self, states):
        self.normalizer.load(states)

