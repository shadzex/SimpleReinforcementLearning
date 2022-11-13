# Deep Deterministic Policy Gradient

import torch
import torch.nn.functional as f

import numpy as np

from os.path import isfile

from RL.MFRL.DQN.DQN.DQN import DQN
from layers import Linear
from network import GeneralNetwork, Network
from optimizer import Optimizer
from utils import hard_update, soft_update, OrnsteinUhlenbeckProcess

class Actor(GeneralNetwork):
    def __init__(self,
                 state_dim,
                 action_num,
                 action_scale,

                 hyperparameters):
        super(Actor, self).__init__(state_dim, hyperparameters)

        self.action = Linear(self.output_dim, action_num)

        if 'init' in hyperparameters.keys():
            range = hyperparameters['init']['tail'][1][0]

            torch.nn.init.uniform_(self.action.weight, -range, range)
            torch.nn.init.uniform_(self.action.bias, -range, range)

        self.action_scale = action_scale

    def forward(self, state):
        x = super(Actor, self).forward(state)

        action = self.action_scale * torch.tanh(self.action(x))

        return action

class Q(Network):
    def __init__(self,
                 state_dim,
                 action_num,

                 hyperparameters):
        action_inclusion = hyperparameters['action_inclusion']
        layers = hyperparameters['layers']

        super(Q, self).__init__(state_dim, layers[:action_inclusion])

        self.body = Network(self.output_dim + action_num, layers[action_inclusion:])

        # tail
        self.q = Linear(self.body.output_dim, 1)

        if 'init' in hyperparameters.keys():
            range = hyperparameters['init']['tail'][1][0]

            torch.nn.init.uniform_(self.action.weight, -range, range)
            torch.nn.init.uniform_(self.action.bias, -range, range)

    def forward(self, state, action):
        x = super(Q, self).forward(state)

        x = torch.cat([x, action], dim=-1)

        x = self.body(x)

        q = self.q(x)

        return q

class DDPG(DQN):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):
        self.tau = hyperparameters.tau
        super().__init__(device,
                         seed,

                         env_info,

                         hyperparameters)

    def build_networks_and_optimizers(self):
        # Networks
        self.actor = Actor(self.state_dim, self.action_num, self.action_scale, self.hyperparameters.actor).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_num, self.action_scale, self.hyperparameters.actor).to(self.device)

        self.critic = Q(self.state_dim, self.action_num, self.hyperparameters.critic).to(self.device)
        self.critic_target = Q(self.state_dim, self.action_num, self.hyperparameters.critic).to(self.device)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Optimizers
        self.actor_optimizer = Optimizer(self.actor, self.hyperparameters.actor_optimizer)
        self.critic_optimizer = Optimizer(self.critic, self.hyperparameters.critic_optimizer)

    def build_others(self):
        self.random_process = OrnsteinUhlenbeckProcess(self.action_num)

    def reset(self):
        self.random_process.reset()

    def explore(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        action = self.actor(state)

        action = action.tolist()[0]

        action = action + self.random_process.noise()

        action = np.clip(action, -self.action_scale, self.action_scale)

        return action

    def act(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        action = self.actor(state)

        action = action.tolist()[0]

        return action

    def process(self,
                state,
                action,
                next_state,
                reward: float,
                done: bool,
                info: dict):

        self.buffer.store((state, action, reward, 1. - done, next_state))

    def get_target(self, rewards, dones, next_states):
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            y = self.reward_scale * rewards + self.discount_factor * dones * self.critic_target(next_states, next_actions)

        return y

    def get_actor_loss(self, states):
        actor_loss = -self.critic(states, self.actor(states)).mean()

        return actor_loss

    def get_critic_loss(self, states, actions, target):
        critic_loss = f.mse_loss(self.critic(states, actions), target)

        return critic_loss

    def sample_data(self):
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)

        states = self.preprocess(states)
        next_states = self.preprocess(next_states)

        self.state_normalizer.update(states)
        states = self.state_normalizer.normalize(states)
        next_states = self.state_normalizer.normalize(next_states)

        states = torch.tensor(states, device=self.device).detach().float()
        actions = torch.tensor(actions, device=self.device).detach().float()
        rewards = torch.tensor(rewards, device=self.device).detach().unsqueeze(1).float()
        dones = torch.tensor(dones, device=self.device).detach().unsqueeze(1).float()
        next_states = torch.tensor(next_states, device=self.device).detach().float()

        if self.normalize_reward:
            rewards = (rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + 1e-6)

        return states, actions, rewards, dones, next_states

    def update(self):
        states, actions, rewards, dones, next_states = self.sample_data()

        y = self.get_target(rewards, dones, next_states)

        critic_loss = self.get_critic_loss(states, actions, y)

        self.critic_optimizer.step(critic_loss)

        actor_loss = self.get_actor_loss(states)

        self.actor_optimizer.step(actor_loss)

        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

        train_info = {'actor_loss': actor_loss.item(),
                      'critic_loss': critic_loss.item()}

        hyperparameter_info = {'actor_lr': self.actor_optimizer.get_lr(),
                               'critic_lr': self.critic_optimizer.get_lr(),
                               }

        return 1, train_info, hyperparameter_info

    def change_train_mode(self, mode):
        self.actor.train(mode)
        self.actor_target.train(mode)

        self.critic.train(mode)
        self.critic_target.train(mode)

    def save_models(self, path: str):
        torch.save({'actor': self.actor.state_dict(),
                    'actor_target': self.actor_target.state_dict(),
                    'critic': self.critic.state_dict(),
                    'critic_target': self.critic_target.state_dict(),
                    'normalizer': self.state_normalizer,
                    }, path)

    def load_models(self,
                    path: str,
                    test: bool = False):
        if isfile(path):
            checkpoint = torch.load(path)

            self.actor.load_state_dict(checkpoint['actor'])
            self.state_normalizer = checkpoint['normalizer']
            if not test:
                self.actor_target.load_state_dict(checkpoint['actor_target'])
                self.critic.load_state_dict(checkpoint['critic'])
                self.critic_target.load_state_dict(checkpoint['critic_target'])

if __name__ == '__main__':
    from runner import run_rl
    run_rl(DDPG)
