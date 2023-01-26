# Proximal Policy Optimization

import torch
import torch.nn.functional as f
from torch.distributions import Categorical, MultivariateNormal

import numpy as np

from MFRL.A2C.A2C.A2C import A2C


def epsilon_update_func(func_type, epsilon, hyperparameters):
    if func_type == 'Identity':
        return epsilon
    else:
        epsilon_min = hyperparameters.epsilon_min
        return max(epsilon - 0.001, epsilon_min)

class PPO_clip(A2C):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):
        self.gae_lambda = hyperparameters.gae_lambda
        self.epsilon = hyperparameters.epsilon
        self.epsilon_update_func_type = hyperparameters.epsilon_update_func_type
        self.epsilon_update_rate = hyperparameters.epsilon_update_rate
        self.epsilon_update_hyperparameters = hyperparameters.epsilon_update_hyperparameters
        self.update_epochs = hyperparameters.update_epochs

        self.prev_done = False

        super().__init__(device,
                         seed,

                         env_info,

                         hyperparameters)
        self.mini_batch_size = self.horizon // hyperparameters.mini_batch_num

    def build_memories(self):

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def explore(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        if self.action_space_type == 'discrete':
            probs = self.actor(state)
            policy = Categorical(probs=probs)
            action = policy.sample()
            self.log_prob = policy.log_prob(action).item()
            action = action.item()

        else:
            mu = self.actor(state)
            cov_mat = torch.diag(self.var).unsqueeze(dim=0)

            policy = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)
            action = policy.rsample()
            self.log_prob = policy.log_prob(action).tolist()[0]
            action = action.tolist()[0]

        self.value = self.critic(state).item()

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
        self.log_probs.append(self.log_prob)
        self.rewards.append(reward)
        self.values.append(self.value)
        self.dones.append(1.0 - self.prev_done)
        self.last_state = next_state
        self.last_done = 1.0 - done

        self.iteration += 1

        self.prev_done = done

    def update(self):

        last_state = self.preprocess_inputs(self.last_state, self.state_normalizer)
        last_value = self.critic(last_state)
        last_value = last_value.item()

        self.values.append(last_value)
        self.dones.append(self.last_done)

        states = np.array(self.states)
        actions = np.array(self.actions)
        log_probs = np.array(self.log_probs)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        # Calculate Generalized Advantage Estimation

        gae = 0.
        advantages = np.zeros_like(rewards)
        for t in reversed(range(self.horizon)):
            delta = rewards[t] + self.discount_factor * values[t + 1] * dones[t + 1] - values[t]
            advantages[t] = gae = delta + self.discount_factor * self.gae_lambda * dones[t + 1] * gae

        targets = advantages + values[:-1]

        # Normalizing the rewards
        if self.normalize_reward:
            targets = (targets - targets.mean()) / (targets.std() + 1e-6)
        targets = self.reward_scale * targets

        # Train
        actor_losses = []
        critic_losses = []

        states = self.preprocess(states)

        self.state_normalizer.update(states)
        states = self.state_normalizer.normalize(states)

        states = torch.tensor(states, device=self.device).detach().float()
        actions = torch.tensor(actions, device=self.device).detach()
        old_log_probs = torch.tensor(log_probs, device=self.device).detach().float()
        targets = torch.tensor(targets, device=self.device).detach().float()

        for epoch in range(self.update_epochs):
            for start in range(0, self.horizon, self.mini_batch_size):
                end = start + self.mini_batch_size

                states_batch = states[start:end]
                actions_batch = actions[start:end]
                old_log_probs_batch = old_log_probs[start:end]
                targets_batch = targets[start:end]

                value = self.critic(states_batch).squeeze()

                if self.action_space_type == 'discrete':
                    probs = self.actor(states_batch)
                    policy = Categorical(probs=probs)

                else:
                    mu = self.actor(states_batch)
                    var = self.var.expand_as(mu)
                    cov_mat = torch.diag_embed(var).to(self.device)
                    policy = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)

                log_probs = policy.log_prob(actions_batch)

                advantages = targets_batch - value.detach()

                ratio = torch.exp(log_probs - old_log_probs_batch)

                surrogate = ratio * advantages
                clipped_surrogate = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages

                actor_loss = -torch.min(surrogate, clipped_surrogate).mean()

                critic_loss = self.value_coefficient * f.mse_loss(value, targets_batch)

                entropy = -self.entropy_coefficient * policy.entropy().mean()

                loss = actor_loss + critic_loss + entropy

                self.optimizer.step(loss)

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        if self.iteration % self.epsilon_update_rate == 0:
            self.epsilon = epsilon_update_func(self.epsilon_update_func_type, self.epsilon, self)

        if self.action_space_type == 'continuous' and self.iteration % self.std_decay_rate == 0:
            self.std = max(self.std - self.std_decay, self.std_min)
            self.var = torch.full((self.action_num,), self.std * self.std).to(self.device)

        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

        train_info = {'actor_loss': np.mean(actor_losses),
                      'critic_loss': np.mean(critic_losses)}

        hyperparameter_info = {'lr': self.optimizer.get_lr()
                               }

        return 1, train_info, hyperparameter_info

    def trainable(self, *args):
        return self.iteration % self.horizon == 0

if __name__ == '__main__':
    from runner import run
    run(PPO_clip)