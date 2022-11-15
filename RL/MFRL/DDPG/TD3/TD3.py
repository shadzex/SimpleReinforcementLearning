# Twin Delayed Deep Deterministic Policy Gradient

import torch
import torch.nn.functional as f

import numpy as np

from os.path import isfile

from RL.MFRL.DDPG.DDPG.DDPG import DDPG, Actor, Q
from optimizer import Optimizer
from utils import hard_update, soft_update

class TD3(DDPG):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):
        self.policy_noise = hyperparameters.policy_noise
        self.sigma = hyperparameters.sigma

        self.clip_range = hyperparameters.clip_range
        self.actor_update_delay = hyperparameters.actor_update_delay

        super().__init__(device,
                         seed,

                         env_info,

                         hyperparameters)

    def build_networks_and_optimizers(self):
        # Networks
        self.actor = Actor(self.state_dim, self.action_num, self.action_scale, self.hyperparameters.actor).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_num, self.action_scale, self.hyperparameters.actor).to(self.device)

        self.critic1 = Q(self.state_dim, self.action_num, self.hyperparameters.critic).to(self.device)
        self.critic1_target = Q(self.state_dim, self.action_num, self.hyperparameters.critic).to(self.device)

        self.critic2 = Q(self.state_dim, self.action_num, self.hyperparameters.critic).to(self.device)
        self.critic2_target = Q(self.state_dim, self.action_num, self.hyperparameters.critic).to(self.device)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic1_target, self.critic1)
        hard_update(self.critic2_target, self.critic2)

        # Optimizers
        self.actor_optimizer = Optimizer(self.actor, self.hyperparameters.actor_optimizer)
        self.critic_optimizer = Optimizer([self.critic1, self.critic2], self.hyperparameters.critic_optimizer)

    def reset(self):
        return

    def explore(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        action = self.actor(state)

        action = action.tolist()[0]

        action = action + np.random.normal(0, self.sigma, self.action_num)

        action = np.clip(action, -self.action_scale, self.action_scale)

        return action

    def act(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        action = self.actor(state)

        action = action.tolist()[0]

        return action

    def get_target(self, actions, rewards, dones, next_states):
        with torch.no_grad():
            noise = torch.clamp(torch.randn_like(actions, device=self.device) * self.policy_noise, -self.clip_range, self.clip_range)

            next_actions = (self.actor_target(next_states) + noise).clamp(-self.action_scale, self.action_scale)

            q1_target = self.critic1_target(next_states, next_actions)
            q2_target = self.critic2_target(next_states, next_actions)

            y = self.reward_scale * rewards + self.discount_factor * dones * torch.min(q1_target, q2_target)

        return y

    def get_actor_loss(self, states):
        actor_loss = -self.critic1(states, self.actor(states)).mean()

        return actor_loss

    def get_critic_loss(self, states, actions, target):
        critic1_loss = f.mse_loss(self.critic1(states, actions), target)

        critic2_loss = f.mse_loss(self.critic2(states, actions), target)

        return critic1_loss, critic2_loss

    def update(self):
        states, actions, rewards, dones, next_states = self.sample_data()

        y = self.get_target(actions, rewards, dones, next_states)

        critic1_loss, critic2_loss = self.get_critic_loss(states, actions, y)

        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.step(critic_loss)

        if self.iteration % self.actor_update_delay == 0:

            actor_loss = self.get_actor_loss(states)

            self.actor_optimizer.step(actor_loss)

            soft_update(self.critic1_target, self.critic1, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

            train_info = {'actor_loss': actor_loss.item(),
                          'critic1_loss': critic1_loss.item(),
                          'critic2_loss': critic2_loss.item()}
        else:
            train_info = {'actor_loss': None,
                          'critic1_loss': critic1_loss.item(),
                          'critic2_loss': critic2_loss.item()}

        self.iteration += 1

        hyperparameter_info = {'actor_lr': self.actor_optimizer.get_lr(),
                               'critic1_lr': self.critic_optimizer.get_lr()
                               }

        return 1, train_info, hyperparameter_info

    def change_train_mode(self, mode):
        self.actor.train(mode)
        self.actor_target.train(mode)

        self.critic1.train(mode)
        self.critic1_target.train(mode)

        self.critic2.train(mode)
        self.critic2_target.train(mode)

    def save_models(self, path: str):
        torch.save({'actor': self.actor.state_dict(),
                    'actor_target': self.actor_target.state_dict(),
                    'critic1': self.critic1.state_dict(),
                    'critic1_target': self.critic1_target.state_dict(),
                    'critic2': self.critic2.state_dict(),
                    'critic2_target': self.critic2_target.state_dict(),
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
                self.critic1.load_state_dict(checkpoint['critic1'])
                self.critic1_target.load_state_dict(checkpoint['critic1_target'])
                self.critic2.load_state_dict(checkpoint['critic2'])
                self.critic2_target.load_state_dict(checkpoint['critic2_target'])

if __name__ == '__main__':
    from runner import run
    run(TD3)
