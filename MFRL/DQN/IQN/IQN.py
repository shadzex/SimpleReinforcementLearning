# Implicit Quantile Network

import torch
import torch.nn as nn
import torch.nn.functional as f

from utils import hard_update

from MFRL.DQN.QR_DQN.QR_DQN import QR_DQN

from network import GeneralNetwork
from layers import Linear
from optimizer import Optimizer

import random

import numpy as np

class Q(GeneralNetwork):
    def __init__(self,
                 state_dim,
                 action_num,

                 hyperparameters):
        super(Q, self).__init__(state_dim,

                                hyperparameters)

        self.phi = Linear(1, self.output_dim, bias=False)
        self.phi_bias = nn.Parameter(torch.zeros(self.output_dim), requires_grad=True)

        self.q = Linear(self.output_dim, action_num)

    def forward(self, state, quantile, i):
        x = super(Q, self).forward(state)

        x = x.unsqueeze(1)
        phi = f.relu(self.phi(torch.cos(np.pi * i * quantile).unsqueeze(2)).mean(1) + self.phi_bias.unsqueeze(0)).unsqueeze(0)
        x = x * phi

        q = self.q(x).transpose(1, 2)

        return q

class IQN(QR_DQN):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):

        super(IQN, self).__init__(device,
                                  seed,

                                  env_info,

                                  hyperparameters)

        self.i = torch.arange(self.quantile_num, device=self.device).float()

    def build_networks_and_optimizers(self):
        # Networks
        self.q = Q(self.state_dim, self.action_num, self.hyperparameters.q).to(self.device)
        self.q_target = Q(self.state_dim, self.action_num, self.hyperparameters.q).to(self.device)

        hard_update(self.q_target, self.q)

        # Optimizers
        self.optimizer = Optimizer(self.q, self.hyperparameters.optimizer)

    def explore(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        if random.random() < self.epsilon:
            action = random.randrange(self.action_num)
        else:
            quantile = torch.rand(self.quantile_num, 1, device=self.device)
            action = self.q(state, quantile, self.i).mean(2).argmax().item()

        return action

    def act(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        quantile = torch.rand(self.quantile_num, 1, device=self.device)
        action = torch.argmax(self.q(state, quantile, self.i).mean(2)).item()

        return action

    def get_target(self, rewards, dones, next_states):
        target_quantile = torch.rand(self.quantile_num, 1, device=self.device)
        target_support = self.q_target(next_states, target_quantile, self.i).detach()
        target_actions = torch.argmax(target_support.mean(2), dim=1).view(-1, 1, 1).expand(-1, 1, self.quantile_num)
        target_support = self.reward_scale * rewards + self.discount_factor * dones * target_support.gather(1, target_actions).squeeze(1)

        target_support = target_support.unsqueeze(1)

        return target_support

    def get_loss(self, states, actions, target):
        quantile = torch.rand(self.quantile_num, 1, device=self.device)
        support = self.q(states, quantile, self.i).gather(1, actions).squeeze(1).unsqueeze(2)

        difference = target.detach() - support

        huber_loss = torch.where(difference.abs() <= self.kappa, 0.5 * difference.pow(2), self.kappa * (difference.abs() - 0.5 * self.kappa))
        loss = (huber_loss * (quantile.unsqueeze(0) - (difference < 0).float()).abs()).mean()

        return loss

if __name__ == '__main__':
    from runner import run
    run(IQN)