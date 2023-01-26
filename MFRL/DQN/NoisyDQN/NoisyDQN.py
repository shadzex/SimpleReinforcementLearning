# Deep Q Network with Noisy Network

import torch
import torch.nn as nn
import torch.nn.functional as f

from MFRL.DQN.DoubleDQN.DoubleDQN import DoubleDQN

import numpy as np

from network import GeneralNetwork


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))

        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.init_parameters()

    def init_parameters(self):
        mu_range = 1 / np.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.bias_sigma.size(0)))

    def forward(self, inputs):
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_mu * self.bias_epsilon

        return f.linear(inputs, weight, bias)

    def act(self, inputs):
        weight = self.weight_mu
        bias = self.bias_mu

        return f.linear(inputs, weight, bias)

    def reset_noise(self):
        epsilon_p = torch.randn(self.in_features)
        epsilon_p = epsilon_p.sign() * epsilon_p.abs().sqrt()

        epsilon_q = torch.randn(self.out_features)
        epsilon_q = epsilon_q.sign() * epsilon_q.abs().sqrt()

        self.weight_epsilon.copy_(epsilon_q.ger(epsilon_p))
        self.bias_epsilon.copy_(epsilon_q)

class Q(GeneralNetwork):
    def __init__(self,
                 state_dim,
                 action_num,

                 hyperparameters):
        super(Q, self).__init__(state_dim,

                                hyperparameters)

        self.q = NoisyLinear(self.output_dim, action_num)

    def forward(self, state):
        x = super(Q, self).forward(state)

        q = self.q(x)

        return q

    def act(self, state):
        x = super(Q, self).forward(state)

        q = self.q.act(x)

        return q

    def reset_noise(self):
        self.q.reset_noise()
    
class NoisyDQN(DoubleDQN):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):
        super(NoisyDQN, self).__init__(device,
                                       seed,

                                       env_info,

                                       hyperparameters)

    def explore(self, state):
        self.q.reset_noise()
        self.q_target.reset_noise()

        state = self.preprocess_inputs(state, self.state_normalizer)

        action = self.q(state).argmax().item()

        return action

    def act(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        action = self.q.act(state).argmax().item()

        return action

    def process(self,
                state,
                action,
                next_state,
                reward: float,
                done: bool,
                info: dict):

        self.buffer.store((state, action, reward, 1. - done, next_state))

if __name__ == '__main__':
    from runner import run
    run(NoisyDQN)