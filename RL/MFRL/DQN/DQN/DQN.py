# Deep Q Network

import torch

from os.path import isfile

from buffer import ReplayBuffer
from base import BaseRLAlgorithm
from network import GeneralNetwork
from layers import Linear
from optimizer import Optimizer
from utils import hard_update, soft_update
from loss import MSE

import random

# Neural network definition for Q function
class Q(GeneralNetwork):
    def __init__(self,
                 state_dim,
                 action_num,

                 hyperparameters):
        super(Q, self).__init__(state_dim,

                                hyperparameters)

        self.q = Linear(self.output_dim, action_num)

    def forward(self, state):
        x = super(Q, self).forward(state)

        q = self.q(x)

        return q

class DQN(BaseRLAlgorithm):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):
        super(DQN, self).__init__(device,
                                  seed,

                                  env_info,

                                  hyperparameters)

        # Hyperparameters
        self.epsilon = hyperparameters.epsilon
        self.epsilon_min = hyperparameters.epsilon_min
        self.epsilon_decay = hyperparameters.epsilon_decay
        self.epsilon_decay_rate = hyperparameters.epsilon_decay_rate

        self.buffer_size = hyperparameters.buffer_size
        self.batch_size = hyperparameters.batch_size

        self.normalize_state = hyperparameters.normalize_state
        self.normalization_method = hyperparameters.normalization_method
        self.normalize_reward = hyperparameters.normalize_reward
        self.reward_scale = hyperparameters.reward_scale
        self.target_update_rate = hyperparameters.target_update_rate
        self.tau = hyperparameters.tau

        self.build()

    def build_memories(self):
        self.buffer = ReplayBuffer(self.buffer_size)

    def build_networks_and_optimizers(self):
        # Networks
        self.q = Q(self.state_dim, self.action_num, self.hyperparameters.q).to(self.device)
        self.q_target = Q(self.state_dim, self.action_num, self.hyperparameters.q).to(self.device)

        hard_update(self.q_target, self.q)

        # Optimizers
        self.optimizer = Optimizer(self.q, self.hyperparameters.optimizer)

    def build_losses(self):
        self.loss = MSE()

    def reset(self):
        return

    def explore(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        if random.random() < self.epsilon:
            action = random.randrange(self.action_num)
        else:
            action = self.q(state).argmax().item()

        return action

    def act(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        action = self.q(state).argmax().item()

        return action

    def process(self,
                state,
                action,
                next_state,
                reward: float,
                done: bool,
                info: dict):

        if self.iteration % self.epsilon_decay_rate == 0:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        self.buffer.store((state, action, reward, 1. - done, next_state))

    # Loss function
    def get_target(self, rewards, dones, next_states):
        # Bellman target
        # r + γ * max_a0(Q(s',a';θi−1))
        with torch.no_grad():
            y = self.reward_scale * rewards + self.discount_factor * dones * self.q_target(next_states).max(1)[0]

        return y

    def get_loss(self, states, actions, target):
        return self.loss(self.q(states).gather(1, actions), target.unsqueeze(1))

    def get_other_loss(self):
        return 0.

    def sample_data(self):
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)

        states = self.preprocess(states)
        next_states = self.preprocess(next_states)

        self.state_normalizer.update(states)
        states = self.state_normalizer.normalize(states)
        next_states = self.state_normalizer.normalize(next_states)

        states = torch.tensor(states, device=self.device).detach().float()
        actions = torch.tensor(actions, device=self.device).unsqueeze(1).detach().long()
        rewards = torch.tensor(rewards, device=self.device).detach().float()
        dones = torch.tensor(dones, device=self.device).detach().float()
        next_states = torch.tensor(next_states, device=self.device).detach().float()

        if self.normalize_reward:
            rewards = (rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + 1e-6)

        return states, actions, rewards, dones, next_states

    def update(self):
        states, actions, rewards, dones, next_states = self.sample_data()

        y = self.get_target(rewards, dones, next_states)

        loss = self.get_loss(states, actions, y)

        self.optimizer.step(loss)

        if self.iteration % self.target_update_rate == 0:
            soft_update(self.q_target, self.q, self.tau)

        self.iteration += 1

        train_info = {'loss': loss.item()}

        hyperparameter_info = {'lr': self.optimizer.get_lr()
                               }

        return 1, train_info, hyperparameter_info

    def trainable(self):
        return self.buffer.size() >= self.batch_size

    def change_train_mode(self, mode):
        self.q.train(mode)
        self.q_target.train(mode)

    def save_models(self,
                    path: str):
        torch.save({'q': self.q.state_dict(),
                    'q_target': self.q_target.state_dict()
                    }, path)

    def load_models(self,
                    path: str,
                    test: bool = False):
        # Pre-build

        if isfile(path):
            checkpoint = torch.load(path)

            self.q.load_state_dict(checkpoint['q'])
            if not test:
                self.q_target.load_state_dict(checkpoint['q_target'])

if __name__ == '__main__':
    from runner import run
    run(DQN)
