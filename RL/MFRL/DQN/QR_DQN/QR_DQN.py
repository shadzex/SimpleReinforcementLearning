# Quantile Regression Deep Q Network

import torch

from utils import hard_update, soft_update

from RL.MFRL.DQN.DoubleDQN.DoubleDQN import DoubleDQN

from network import GeneralNetwork
from layers import Linear
from optimizer import Optimizer

import random

import numpy as np

class Q(GeneralNetwork):
    def __init__(self,
                 state_dim,
                 action_num,
                 quantile_num,

                 hyperparameters):
        super(Q, self).__init__(state_dim,

                                hyperparameters)

        self.q = Linear(self.output_dim, action_num * quantile_num)

        self.action_num = action_num
        self.quantile_num = quantile_num

    def forward(self, state):
        x = super(Q, self).forward(state)

        q = self.q(x)
        q = q.view(-1, self.action_num, self.quantile_num)

        return q

class QR_DQN(DoubleDQN):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):
        self.quantile_num = hyperparameters.quantile_num
        self.kappa = hyperparameters.kappa

        super(QR_DQN, self).__init__(device,
                                     seed,

                                     env_info,

                                     hyperparameters)

        self.quantile_midpoint = torch.tensor((2 * np.arange(self.quantile_num) + 1) / (2 * self.quantile_num), device=self.device).detach().float()

    def build_networks_and_optimizers(self):
        # Networks
        self.q = Q(self.state_dim, self.action_num, self.quantile_num, self.hyperparameters.q).to(self.device)
        self.q_target = Q(self.state_dim, self.action_num, self.quantile_num, self.hyperparameters.q).to(self.device)

        hard_update(self.q_target, self.q)

        # Optimizers
        self.optimizer = Optimizer(self.q, self.hyperparameters.optimizer)

    def explore(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        if random.random() < self.epsilon:
            action = random.randrange(self.action_num)
        else:
            action = self.q(state).mean(2).argmax().item()

        return action

    def act(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        action = torch.argmax(self.q(state).mean(2)).item()

        return action

        # Loss function

    def get_target(self, rewards, dones, next_states):
        target_support = self.q_target(next_states).detach()
        target_actions = torch.argmax(target_support.mean(2), dim=1).view(-1, 1, 1).expand(-1, 1, self.quantile_num)
        target_support = self.reward_scale * rewards + self.discount_factor * dones * target_support.gather(1, target_actions).squeeze(1)

        target_support = target_support.unsqueeze(1)

        return target_support

    def get_loss(self, states, actions, target):
        support = self.q(states).gather(1, actions).squeeze(1).unsqueeze(2)

        difference = target.detach() - support

        huber_loss = torch.where(difference.abs() <= self.kappa, 0.5 * difference.pow(2), self.kappa * (difference.abs() - 0.5 * self.kappa))
        loss = (huber_loss * (self.quantile_midpoint.view(1, -1, 1) - (difference < 0).float()).abs()).mean()

        return loss

    def sample_data(self):
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)

        states = self.preprocess(states)
        next_states = self.preprocess(next_states)

        self.state_normalizer.update(states)
        states = self.state_normalizer.normalize(states)
        next_states = self.state_normalizer.normalize(next_states)

        states = torch.tensor(states, device=self.device).detach().float()
        actions = torch.tensor(actions, device=self.device).view(-1, 1, 1).expand(-1, -1, self.quantile_num).detach().long()
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(1).expand(-1, self.quantile_num).detach().float()
        dones = torch.tensor(dones, device=self.device).unsqueeze(1).expand(-1, self.quantile_num).detach().float()
        next_states = torch.tensor(next_states, device=self.device).detach().float()

        if self.normalize_reward:
            rewards = (rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + 1e-6)

        return states, actions, rewards, dones, next_states

if __name__ == '__main__':
    from runner import run
    run(QR_DQN)