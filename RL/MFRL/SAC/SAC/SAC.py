# Soft Actor Critic

import torch
import torch.nn.functional as f
from torch.distributions import Normal

import numpy as np

from os.path import isfile

from layers import Linear
from network import GeneralNetwork
from RL.MFRL.DDPG.DDPG.DDPG import DDPG, Q
from optimizer import Optimizer
from utils import hard_update, soft_update

log_std_max = 2
log_std_min = -20
epsilon = 1e-6

class Actor(GeneralNetwork):
    def __init__(self,
                 state_dim,
                 action_num,

                 hyperparameters):
        super(Actor, self).__init__(state_dim,

                                    hyperparameters)
        # tail
        self.mu = Linear(self.output_dim, action_num)
        self.sigma = Linear(self.output_dim, action_num)

    def forward(self, state):
        x = super(Actor, self).forward(state)

        mu = self.mu(x)
        log_std = self.sigma(x)

        sigma = torch.clamp(log_std, min=log_std_min, max=log_std_max).exp()

        return mu, sigma

class SAC(DDPG):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):
        self.adjust_temperature = hyperparameters.adjust_temperature

        self.alpha = hyperparameters.alpha
        self.target_update_rate = hyperparameters.target_update_rate
        self.update_epochs = hyperparameters.update_epochs

        super().__init__(device,
                         seed,

                         env_info,

                         hyperparameters)

    def build_networks_and_optimizers(self):
        # Networks
        self.actor = Actor(self.state_dim, self.action_num, self.hyperparameters.actor).to(self.device)

        self.q1 = Q(self.state_dim, self.action_num, self.hyperparameters.critic).to(self.device)
        self.q1_target = Q(self.state_dim, self.action_num, self.hyperparameters.critic).to(self.device)

        self.q2 = Q(self.state_dim, self.action_num, self.hyperparameters.critic).to(self.device)
        self.q2_target = Q(self.state_dim, self.action_num, self.hyperparameters.critic).to(self.device)

        hard_update(self.q1_target, self.q1)
        hard_update(self.q2_target, self.q2)

        self.log_alpha = torch.tensor(np.log(self.alpha)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -self.action_num

        # Optimizers
        self.actor_optimizer = Optimizer(self.actor, self.hyperparameters.actor_optimizer)
        self.q_optimizer = Optimizer([self.q1, self.q2], self.hyperparameters.critic_optimizer)
        self.temperature_optimizer = Optimizer(self.log_alpha, self.hyperparameters.temperature_optimizer)

    def explore(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        mu, sigma = self.actor(state)

        policy = Normal(loc=mu, scale=sigma)

        action = self.action_scale * torch.tanh(policy.rsample())

        action = action.tolist()[0]

        return action

    def act(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        action, _ = self.actor(state)

        action = self.action_scale * torch.tanh(action)

        action = action.tolist()[0]

        return action

    def get_log_prob(self, policy):
        sample = policy.rsample()
        bound_sample = torch.tanh(sample)
        action = self.action_scale * bound_sample
        log_prob = policy.log_prob(sample)
        log_prob -= torch.log(self.action_scale * (1 - bound_sample.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

    def get_target(self, rewards, dones, next_states, alpha):
        with torch.no_grad():
            next_mu, next_sigma = self.actor(next_states)
            next_pi = Normal(loc=next_mu, scale=next_sigma)

            next_actions, next_log_pi = self.get_log_prob(next_pi)

            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)

            y = self.reward_scale * rewards + self.discount_factor * dones * (torch.min(q1_next, q2_next) - alpha * next_log_pi)

        return y

    def get_actor_loss(self, states, alpha):
        new_mu, new_sigma = self.actor(states)
        new_pi = Normal(loc=new_mu, scale=new_sigma)

        new_actions, new_log_pi = self.get_log_prob(new_pi)

        new_q1 = self.q1(states, new_actions)
        new_q2 = self.q2(states, new_actions)

        actor_loss = (alpha.detach() * new_log_pi - torch.min(new_q1, new_q2)).mean()

        return actor_loss, new_log_pi

    def get_critic_loss(self, states, actions, target):
        q1_loss = f.mse_loss(self.q1(states, actions), target)

        q2_loss = f.mse_loss(self.q2(states, actions), target)

        return q1_loss, q2_loss

    def get_temperature_loss(self, alpha, new_log_pi):
        alpha_loss = (alpha * (-new_log_pi - self.target_entropy).detach()).mean()

        return alpha_loss

    def update(self):
        actor_losses = []
        q1_losses = []
        q2_losses = []

        for i in range(self.update_epochs):
            states, actions, rewards, dones, next_states = self.sample_data()

            alpha = self.log_alpha.exp()

            y = self.get_target(rewards, dones, next_states, alpha)

            q1_loss, q2_loss = self.get_critic_loss(states, actions, y)

            q_loss = q1_loss + q2_loss

            self.q_optimizer.step(q_loss)

            actor_loss, new_log_pi = self.get_actor_loss(states, alpha)

            self.actor_optimizer.step(actor_loss)

            if self.adjust_temperature:
                alpha_loss = self.get_temperature_loss(alpha, new_log_pi)

                self.temperature_optimizer.step(alpha_loss)

            if self.iteration % self.target_update_rate == 0:
                soft_update(self.q1_target, self.q1, self.tau)
                soft_update(self.q2_target, self.q2, self.tau)

            self.iteration += 1

            actor_losses.append(actor_loss.item())
            q1_losses.append(q1_loss.item())
            q2_losses.append(q2_loss.item())

        train_info = {'actor_loss': np.mean(actor_losses),
                      'q1_loss': np.mean(q1_losses),
                      'q2_loss': np.mean(q2_losses)}

        hyperparameter_info = {'actor_lr': self.actor_optimizer.get_lr(),
                               'q1_lr': self.q_optimizer.get_lr(),
                               'alpha_lr': self.temperature_optimizer.get_lr(),
                               'alpha': self.log_alpha.exp().item()
                               }

        return self.update_epochs, train_info, hyperparameter_info

    def change_train_mode(self, mode):
        self.actor.train(mode)

        self.q1.train(mode)
        self.q1_target.train(mode)

        self.q2.train(mode)
        self.q2_target.train(mode)

    def save_models(self,
                    path: str):
        torch.save({'actor': self.actor.state_dict(),
                    'q1': self.q1.state_dict(),
                    'q2': self.q2.state_dict(),
                    'log_alpha': self.log_alpha,
                    'q1_target': self.q1_target.state_dict(),
                    'q2_target': self.q2_target.state_dict(),
                    'normalizer': self.state_normalizer,
                    'buffer': self.buffer
                    }, path)

    def load_models(self,
                    path: str,
                    test: bool = False):

        # Pre-build

        if isfile(path):
            checkpoint = torch.load(path)

            self.actor.load_state_dict(checkpoint['actor'])
            self.state_normalizer = checkpoint['normalizer']
            if not test:
                self.q1.load_state_dict(checkpoint['q1'])
                self.q2.load_state_dict(checkpoint['q2'])

                self.log_alpha = checkpoint['log_alpha']

                self.q1_target.load_state_dict(checkpoint['q1_target'])
                self.q2_target.load_state_dict(checkpoint['q2_target'])

                self.buffer = checkpoint['buffer']


if __name__ == '__main__':
    from runner import run
    run(SAC)