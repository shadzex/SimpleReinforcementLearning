# General network generator

import torch.nn as nn

from layers import Layer

# Network class
class Network(nn.Module):
    def __init__(self, input_dim=0, hyperparameters={}):
        super(Network, self).__init__()

        self.network = nn.Sequential()

        self.output_dim = input_dim

        if isinstance(self.output_dim, list) and len(self.output_dim) == 1:
            self.output_dim = self.output_dim[0]

        for i, (layer_type, parameters) in enumerate(hyperparameters):
            self.output_dim = self.add_layer(input_dim, i, layer_type, parameters)

            input_dim = self.output_dim

    def add_layer(self, input_dim, number, layer_type, parameters):
        layer = Layer(input_dim, number, layer_type, parameters)
        self.network.add_module('layer{}'.format(number), layer)

        return layer.output_dim

    def forward(self, x):
        return self.network(x)

    @property
    def output_size(self):
        return self.output_dim

# General network class
class GeneralNetwork(nn.Module):
    def __init__(self,
                 input_dim,

                 hyperparameters):
        super(GeneralNetwork, self).__init__()

        self.network = Network(input_dim, hyperparameters['layers'])

        self.output_dim = self.network.output_dim

    def forward(self, inputs):
        return self.network(inputs)