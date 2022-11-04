# Single Layer class for constructing neural network

import torch.nn as nn

import numpy as np

from utils import calculate_conv2d_output, calculate_max_pool_2d_output, calculate_conv_transpose_2d_output

from typing import Union


# torch layer expansion for automatic network generation by hyperparameter specification
class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)

    @property
    def output_dim(self):
        return self.out_features

class Conv2d(nn.Conv2d):
    def __init__(self, input_dim, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(Conv2d, self).__init__(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride,
                                     padding,
                                     bias=bias)

        self.input_dim = input_dim

    @property
    def output_dim(self):
        return [self.out_channels] + list(calculate_conv2d_output(self.input_dim[1:], self.kernel_size, self.padding, self.stride))

class Flatten(nn.Flatten):
    def __init__(self, input_dim):
        super(Flatten, self).__init__()

        self.input_dim = input_dim

    @property
    def output_dim(self):
        return int(np.prod(self.input_dim))

class ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, input_dim, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ConvTranspose2d, self).__init__(in_channels,
                                              out_channels,
                                              kernel_size,
                                              stride,
                                              padding,
                                              bias=bias)

        self.input_dim = input_dim

    @property
    def output_dim(self):
        return [self.out_channels] + list(calculate_conv_transpose_2d_output(self.input_dim[1:], self.kernel_size, self.padding, self.stride))

class MaxPool2d(nn.MaxPool2d):
    def __init__(self, input_dim, kernel_size, stride):
        super(MaxPool2d, self).__init__(kernel_size,
                                        stride)

        self.input_dim = input_dim

    @property
    def output_dim(self):
        return [self.input_dim[0]] + list(calculate_max_pool_2d_output(self.input_dim[1:], self.kernel_size, 0, self.stride))

class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, input_dim):
        super(BatchNorm1d, self).__init__(input_dim)

        self.input_dim = input_dim

    @property
    def output_dim(self):
        return self.input_dim

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, input_dim):
        super(BatchNorm2d, self).__init__(input_dim[0])

        self.input_dim = input_dim

    @property
    def output_dim(self):
        return self.input_dim

class LayerNorm(nn.LayerNorm):
    def __init__(self, input_dim):
        super(LayerNorm, self).__init__(input_dim)

        self.input_dim = input_dim

    @property
    def output_dim(self):
        return self.input_dim

MODULES = {'linear': Linear,
           'conv2d': Conv2d,
           'convtranspose2d': ConvTranspose2d,
           'maxpool2d': MaxPool2d,
           'flatten': Flatten,
           'batchnorm1d': BatchNorm1d,
           'batchnorm2d': BatchNorm2d,
           'layernorm': LayerNorm}

ACTIVATIONS = {'sigmoid': nn.Sigmoid(),
               'tanh': nn.Tanh(),
               'softmax': nn.Softmax(dim=-1),
               'relu': nn.ReLU(),
               'elu': nn.ELU()}

class Layer(nn.Module):
    def __init__(self,
                 input_dim: Union[int, list],
                 number: int,
                 layer_type: str,
                 parameters: list):
        super(Layer, self).__init__()

        self.layer = nn.Sequential()

        layer_type = layer_type.lower()

        # Make input dimension integer if it has only a single length
        if isinstance(input_dim, list) and len(input_dim) == 1:
            input_dim = input_dim[0]

        if layer_type != 'flatten':
            activation = parameters[-1]
            parameters = parameters[:-1]
        else:
            activation = None

        layer = MODULES[layer_type](input_dim, *parameters)
        self.output_dim = layer.output_dim

        self.layer.add_module('{}{}'.format(layer_type, number), layer)
        if activation != None:
            self.layer.add_module('{}{}'.format(activation, number), ACTIVATIONS[activation])

    def forward(self, inputs):
        return self.layer(inputs)

