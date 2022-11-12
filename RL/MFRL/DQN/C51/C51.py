# Categorical 51

import torch
import torch.nn.functional as f

from utils import hard_update, soft_update

from RL.MFRL.DQN.DoubleDQN.DoubleDQN import DoubleDQN

from network import GeneralNetwork
from layers import Linear
from optimizer import Optimizer

import random

class Q(GeneralNetwork):
    def __init__(self,
                 state_dim,
                 action_num,
                 support_num,

                 hyperparameters):
        super(Q, self).__init__(state_dim,

                                hyperparameters)

        self.z = Linear(self.output_dim, action_num * support_num)

        self.action_num = action_num
        self.support_num = support_num

    def forward(self, state):
        x = super(Q, self).forward(state)

        z = self.z(x)
        z = f.softmax(z.view(-1, self.action_num, self.support_num), dim=-1)

        return z

class C51(DoubleDQN):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):
        self.support_num = hyperparameters.support_num
        self.v_min = hyperparameters.v_min
        self.v_max = hyperparameters.v_max
        self.delta_z = (self.v_max - self.v_min) / (self.support_num - 1)

        super(C51, self).__init__(device,
                                  seed,

                                  env_info,

                                  hyperparameters)

        self.support = torch.linspace(self.v_min, self.v_max, self.support_num, device=self.device).detach()

    def build_networks_and_optimizers(self):
        # Networks
        self.q = Q(self.state_dim, self.action_num, self.support_num, self.hyperparameters.q).to(self.device)
        self.q_target = Q(self.state_dim, self.action_num, self.support_num, self.hyperparameters.q).to(self.device)

        hard_update(self.q_target, self.q)

        # Optimizers
        self.optimizer = Optimizer(self.q, self.hyperparameters.optimizer)

    def explore(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        if random.random() < self.epsilon:
            action = random.randrange(self.action_num)
        else:
            dist = self.q(state)
            action = torch.argmax((dist * self.support.view(1, 1, -1)).sum(2)).item()

        return action

    def act(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        action = torch.argmax((self.q(state) * self.support).sum(2)).item()

        return action

    def get_target(self, rewards, dones, next_states):
        target_dist = self.q_target(next_states)
        target_actions = torch.argmax((target_dist * self.support.view(1, 1, -1)).sum(2), dim=1).view(-1, 1, 1).expand(-1, -1, self.support_num)
        target_dist = target_dist.gather(1, target_actions).squeeze(1)

        z_j = self.support.unsqueeze(0).expand(self.batch_size, -1)

        bellman_z_j = torch.clamp(self.reward_scale * rewards + self.discount_factor * dones * z_j, self.v_min, self.v_max)

        return [target_dist, bellman_z_j]

    def get_loss(self, states, actions, target):

        target_dist, bellman_z_j = target

        dist = self.q(states).gather(1, actions).squeeze(1)

        b = (bellman_z_j - self.v_min) / self.delta_z

        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (self.batch_size - 1) * self.support_num, self.batch_size).long().unsqueeze(1).expand(-1, self.support_num).to(self.device)

        m = torch.zeros(target_dist.shape, device=self.device).detach()

        m.view(-1).index_add_(0, (l + offset).view(-1), (target_dist * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), (target_dist * (b - l.float())).view(-1))

        loss = -(m * torch.log(dist + 1e-8)).mean()

        return loss

if __name__ == '__main__':
    from runner import run
    run(C51)