# RAINBOW DQN

import torch
import torch.nn.functional as f

from buffer import PrioritizedReplayBuffer

from utils import hard_update, soft_update

from RL.MFRL.DQN.NoisyDQN.NoisyDQN import NoisyDQN, NoisyLinear

from network import GeneralNetwork
from optimizer import Optimizer

import numpy as np

class Q(GeneralNetwork):
    def __init__(self,
                 state_dim,
                 action_num,
                 support_num,

                 hyperparameters):
        super(Q, self).__init__(state_dim,

                                hyperparameters)

        self.v = NoisyLinear(self.output_dim, 1)
        self.a = NoisyLinear(self.output_dim, action_num * support_num)

        self.action_num = action_num
        self.support_num = support_num

    def forward(self, state):
        x = super(Q, self).forward(state)

        value = self.v(x)

        advantage = self.a(x)

        q = value + (advantage - advantage.mean(1).unsqueeze(1))

        z = f.softmax(q.view(-1, self.action_num, self.support_num), dim=-1)

        return z

    def act(self, state):
        x = super(Q, self).forward(state)

        value = self.v.act(x)

        advantage = self.a.act(x)

        q = value + (advantage - advantage.mean(1).unsqueeze(1))

        z = f.softmax(q.view(-1, self.action_num, self.support_num), dim=-1)

        return z

    def reset_noise(self):
        self.v.reset_noise()
        self.a.reset_noise()

class RAINBOW(NoisyDQN):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):


        self.per_alpha = hyperparameters.per_alpha
        self.per_beta = hyperparameters.per_beta
        self.per_epsilon = hyperparameters.per_epsilon

        self.n_step = hyperparameters.n_step
        self.mini_batch_size = hyperparameters.mini_batch_size
        self.support_num = hyperparameters.support_num
        self.v_min = hyperparameters.v_min
        self.v_max = hyperparameters.v_max
        self.delta_z = (self.v_max - self.v_min) / (self.support_num - 1)

        super(RAINBOW, self).__init__(device,
                                      seed,

                                      env_info,

                                      hyperparameters)

        self.support = torch.linspace(self.v_min, self.v_max, self.support_num, device=self.device).detach()

    def build_memories(self):
        self.buffer = PrioritizedReplayBuffer(self.buffer_size, self.per_beta)
        self.step_buffer = []
        self.local_buffer = []

    def build_networks_and_optimizers(self):
        # Networks
        self.q = Q(self.state_dim, self.action_num, self.support_num, self.hyperparameters.q).to(self.device)
        self.q_target = Q(self.state_dim, self.action_num, self.support_num, self.hyperparameters.q).to(self.device)

        hard_update(self.q_target, self.q)

        # Optimizers
        self.optimizer = Optimizer(self.q, self.hyperparameters.optimizer)


    def explore(self, state):
        self.q.reset_noise()
        self.q_target.reset_noise()

        state = self.preprocess_inputs(state, self.state_normalizer)

        dist = self.q(state)
        action = torch.argmax((dist * self.support.view(1, 1, -1)).sum(2)).item()

        return action

    def act(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        action = torch.argmax((self.q.act(state) * self.support).sum(2)).item()

        return action

    def make_n_step(self):
        transition = self.step_buffer.pop(0)

        state, action, reward, done, next_state, info = transition

        multi_step_reward = reward
        discount_factor = self.discount_factor
        done_pn = done
        next_state_pn = next_state
        for i in range(len(self.step_buffer)):
            state_pi, action_pi, reward_pi, done_pi, next_state_pi, info_pi = self.step_buffer[i]
            multi_step_reward += discount_factor * reward_pi
            discount_factor *= self.discount_factor

            done_pn = done_pi
            next_state_pn = next_state_pi

        self.local_buffer.append((state, action, multi_step_reward, 1. - done_pn, next_state_pn, discount_factor))

    def process(self,
                state,
                action,
                next_state,
                reward: float,
                done: bool,
                info: dict):

        self.step_buffer.append((state, action, reward, done, next_state, info))

        if len(self.step_buffer) >= self.n_step:
            self.make_n_step()

            if done:
                for i in range(len(self.step_buffer)):
                    self.make_n_step()

        if len(self.local_buffer) >= self.mini_batch_size:
            batch = self.local_buffer[:self.mini_batch_size]

            del self.local_buffer[:self.mini_batch_size]

            batch_transposed = list(zip(*batch))

            states, actions, rewards, dones, next_states, discount_factors = batch_transposed

            states = torch.tensor(np.array(states), device=self.device).float()
            actions = torch.tensor(actions, device=self.device).view(-1, 1, 1).expand(-1, -1, self.support_num).long()
            rewards = torch.tensor(rewards, device=self.device).unsqueeze(1).expand(-1, self.support_num).float()
            dones = torch.tensor(dones, device=self.device).unsqueeze(1).expand(-1, self.support_num).float()
            next_states = torch.tensor(np.array(next_states), device=self.device).float()
            discount_factors = torch.tensor(discount_factors, device=self.device).unsqueeze(1).expand(-1, self.support_num).float()

            dist = self.q(states).gather(1, actions).squeeze(1)

            target_dist = self.q_target(next_states)
            target_actions = torch.argmax((self.q(next_states) * self.support.view(1, 1, -1)).sum(2), dim=1).view(-1, 1, 1).expand(-1, -1, self.support_num)
            target_dist = target_dist.gather(1, target_actions).squeeze(1)

            z_j = self.support.unsqueeze(0).expand(self.mini_batch_size, -1)

            bellman_z_j = torch.clamp(rewards + discount_factors * dones * z_j, self.v_min, self.v_max)

            b = (bellman_z_j - self.v_min) / self.delta_z

            l = b.floor().long()
            u = b.ceil().long()

            offset = torch.linspace(0, (self.mini_batch_size - 1) * self.support_num, self.mini_batch_size, device=self.device).long().unsqueeze(1).expand(-1, self.support_num)

            m = torch.zeros(target_dist.shape, device=self.device).detach()

            m.view(-1).index_add_(0, (l + offset).view(-1), (target_dist * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (target_dist * (b - l.float())).view(-1))

            kl_div = -(m * torch.log(dist + 1e-8))

            kl_div = kl_div.mean(1)

            priorities = torch.pow(torch.abs(kl_div) + self.per_epsilon, self.per_alpha).tolist()

            for i in range(self.mini_batch_size):
                self.buffer.store([priorities[i], batch[i]])

    def get_target(self, rewards, dones, discount_factors, next_states):
        target_dist = self.q_target(next_states)
        target_actions = torch.argmax((target_dist * self.support.view(1, 1, -1)).sum(2), dim=1).view(-1, 1, 1).expand(-1, -1, self.support_num)
        target_dist = target_dist.gather(1, target_actions).squeeze(1)

        z_j = self.support.unsqueeze(0).expand(self.batch_size, -1)

        bellman_z_j = torch.clamp(self.reward_scale * rewards + discount_factors * dones * z_j, self.v_min, self.v_max)

        return [target_dist, bellman_z_j]

    def get_loss(self, states, actions, target, importance_weights):
        target_dist, bellman_z_j = target

        dist = self.q(states).gather(1, actions).squeeze(1)

        b = (bellman_z_j - self.v_min) / self.delta_z

        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (self.batch_size - 1) * self.support_num, self.batch_size).long().unsqueeze(
            1).expand(-1, self.support_num).to(self.device)

        m = torch.zeros(target_dist.shape, device=self.device).detach()

        m.view(-1).index_add_(0, (l + offset).view(-1), (target_dist * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), (target_dist * (b - l.float())).view(-1))

        kl_div = -(m * torch.log(dist + 1e-8))

        kl_div = kl_div.mean(1).unsqueeze(1)

        priorities = torch.pow(torch.abs(kl_div.detach()) + self.per_epsilon, self.per_alpha).squeeze(1).tolist()

        loss = kl_div * importance_weights

        loss = loss.mean()

        return loss, priorities

    def sample_data(self):
        states, actions, rewards, dones, next_states, discount_factors, indices, importance_weights = self.buffer.sample(self.batch_size)

        states = self.preprocess(states)
        next_states = self.preprocess(next_states)

        self.state_normalizer.update(states)
        states = self.state_normalizer.normalize(states)
        next_states = self.state_normalizer.normalize(next_states)

        states = torch.tensor(states, device=self.device).detach().float()
        actions = torch.tensor(actions, device=self.device).view(-1, 1, 1).expand(-1, -1, self.support_num).detach().long()
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(1).expand(-1, self.support_num).detach().float()
        dones = torch.tensor(dones, device=self.device).unsqueeze(1).expand(-1, self.support_num).detach().float()
        next_states = torch.tensor(next_states, device=self.device).detach().float()
        discount_factors = torch.tensor(discount_factors, device=self.device).unsqueeze(1).expand(-1, self.support_num).detach().float()
        importance_weights = torch.tensor(importance_weights, device=self.device).unsqueeze(1).detach().float()

        if self.normalize_reward:
            rewards = (rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + 1e-6)

        return states, actions, rewards, dones, next_states, discount_factors, indices, importance_weights

    def update(self):
        states, actions, rewards, dones, next_states, discount_factors, indices, importance_weights = self.sample_data()

        y = self.get_target(rewards, dones, discount_factors, next_states)

        loss, priorities = self.get_loss(states, actions, y, importance_weights)

        self.optimizer.step(loss)

        self.buffer.update(indices, priorities)

        if self.iteration % self.target_update_rate == 0:
            soft_update(self.q_target, self.q, self.tau)

        self.iteration += 1

        train_info = {'loss': loss.item()}

        hyperparameter_info = {'lr': self.optimizer.get_lr()
                               }

        return 1, train_info, hyperparameter_info

if __name__ == '__main__':
    from runner import run
    run(RAINBOW)