# Advantage Actor Critic

import torch
from torch.distributions import Categorical, MultivariateNormal

import numpy as np

from os.path import isfile

from base import BaseRLAlgorithm
from layers import Layer, Linear
from network import GeneralNetwork, Network
from optimizer import Optimizer

class DiscretePolicy(Network):
    def __init__(self, input_dim, action_num):
        super(DiscretePolicy, self).__init__()
        self.probs = Layer(input_dim, -1, 'linear', [action_num, 'softmax'])

    def forward(self, x):
        probs = self.probs(x)

        return probs

class ContinousPolicy(Network):
    def __init__(self, input_dim, action_num):
        super(ContinousPolicy, self).__init__()

        self.mu = Linear(input_dim, action_num)

    def forward(self, x):
        mu = self.mu(x)

        return mu


class Actor(GeneralNetwork):
    def __init__(self,
                 state_dim,
                 action_space_type,
                 action_num,

                 hyperparameters):
        super(Actor, self).__init__(state_dim,
                                    hyperparameters)

        # tail
        if action_space_type == 'discrete':
            self.policy = DiscretePolicy(self.output_dim, action_num)
        else:
            self.policy = ContinousPolicy(self.output_dim, action_num)

    def forward(self, state):
        x = super(Actor, self).forward(state)

        policy = self.policy(x)

        return policy

class V(GeneralNetwork):
    def __init__(self,
                 state_dim,

                 hyperparameters):
        super(V, self).__init__(state_dim,

                                hyperparameters)

        self.value = Linear(self.output_dim, 1)

    def forward(self, state):
        x = super(V, self).forward(state)

        value = self.value(x)

        return value

class A2C(BaseRLAlgorithm):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):
        super().__init__(device,
                         seed,

                         env_info,

                         hyperparameters)

        # Hyperparameters
        self.actor_learning_rate = hyperparameters.actor_learning_rate
        self.critic_learning_rate = hyperparameters.critic_learning_rate

        if self.action_space_type == 'continuous':
            self.std_init = hyperparameters.std_init
            self.std = self.std_init
            self.std_min = hyperparameters.std_min
            self.std_decay = hyperparameters.std_decay
            self.std_decay_rate = hyperparameters.std_decay_rate

            self.var = torch.full((self.action_num,), self.std_init * self.std_init).to(self.device)

        self.horizon = hyperparameters.horizon

        self.normalize_state = hyperparameters.normalize_state
        self.normalization_method = hyperparameters.normalization_method
        self.normalize_reward = hyperparameters.normalize_reward
        self.reward_scale = hyperparameters.reward_scale
        self.value_coefficient = hyperparameters.value_coefficient
        self.entropy_coefficient = hyperparameters.entropy_coefficient

        self.build()

    def build_memories(self):

        self.states = []
        self.actions = []
        self.rewards = []

    def build_networks_and_optimizers(self):
        # Networks
        self.actor = Actor(self.state_dim, self.action_space_type, self.action_num, self.hyperparameters.actor).to(self.device)
        self.critic = V(self.state_dim, self.hyperparameters.critic).to(self.device)

        # Optimizers
        self.optimizer = Optimizer([self.actor, self.critic], self.hyperparameters.optimizer)

    def reset(self):
        return

    def explore(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        if self.action_space_type == 'discrete':
            probs = self.actor(state)
            policy = Categorical(probs=probs)
            action = policy.sample()
            action = action.item()

        else:
            mu = self.actor(state)
            cov_mat = torch.diag(self.var).unsqueeze(dim=0)

            policy = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)
            action = policy.rsample()
            action = action.tolist()[0]

        return action

    def act(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        if self.action_space_type == 'discrete':
            probs = self.actor(state)

            action = torch.argmax(probs).item()
        else:
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

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.last_state = next_state

        self.done = done

        self.iteration += 1

    def update(self):

        last_state = self.preprocess_inputs(self.last_state, self.state_normalizer)
        last_value = self.critic(last_state)
        last_value = last_value.item()

        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)

        r = (1.0 - self.done) * last_value
        returns = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            r = rewards[t] + self.discount_factor * r
            returns[t] = r

        # Normalizing the returns
        if self.normalize_reward:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        returns = self.reward_scale * returns

        # Train
        states = self.preprocess(states)

        self.state_normalizer.update(states)
        states = self.state_normalizer.normalize(states)

        states = torch.tensor(states, device=self.device).detach().float()
        actions = torch.tensor(actions, device=self.device).detach()
        returns = torch.tensor(returns, device=self.device).detach().float()

        values = self.critic(states).squeeze()

        if self.action_space_type == 'discrete':
            probs = self.actor(states)
            policy = Categorical(probs=probs)

        else:
            mu = self.actor(states)
            var = self.var.expand_as(mu)
            cov_mat = torch.diag_embed(var).to(self.device)
            policy = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)

        log_probs = policy.log_prob(actions)

        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()

        critic_loss = self.value_coefficient * advantages.pow(2).mean()

        entropy = -self.entropy_coefficient * policy.entropy().mean()

        loss = actor_loss + critic_loss + entropy

        self.optimizer.step(loss)

        if self.action_space_type == 'continuous' and self.iteration % self.std_decay_rate == 0:
            self.std = max(self.std - self.std_decay, self.std_min)
            self.var = torch.full((self.action_num,), self.std * self.std).to(self.device)

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

        train_info = {'actor_loss': actor_loss.item(),
                      'critic_loss': critic_loss.item()}

        hyperparameter_info = {'lr': self.optimizer.get_lr(),
                               }

        return 1, train_info, hyperparameter_info

    def trainable(self, *args):
        return self.iteration % self.horizon == 0 or self.done

    def change_train_mode(self, mode):
        self.actor.train(mode)
        self.critic.train(mode)

    def save_models(self,
                    path: str):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict()
                    }, path)

    def load_models(self,
                    path: str,
                    test: bool = False):
        # Pre-build

        if isfile(path):
            checkpoint = torch.load(path)

            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])

if __name__ == '__main__':
    from runner import run_rl
    run_rl(A2C)