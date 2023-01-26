# Deep Q Network with Prioritized Experience Replay

import torch

from buffer import PrioritizedReplayBuffer
from utils import soft_update

from MFRL.DQN.DoubleDQN.DoubleDQN import DoubleDQN

class PrioritizedDQN(DoubleDQN):
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

        super(PrioritizedDQN, self).__init__(device,
                                             seed,

                                             env_info,

                                             hyperparameters)

    def build_memories(self):
        self.buffer = PrioritizedReplayBuffer(self.buffer_size, self.per_beta)

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

        target_action = self.q(torch.tensor(next_state, device=self.device).unsqueeze(0).float()).argmax(1, keepdim=True)
        td = reward + self.discount_factor * (1 - done) * self.q_target(torch.tensor(next_state, device=self.device).unsqueeze(0).float()).gather(1, target_action).item()
        td = td - self.q(torch.tensor(state, device=self.device).unsqueeze(0).float()).gather(1, torch.tensor([[action]], device=self.device))

        # Propotional priority calculation
        # Rank-based is not implemented yet
        priority = torch.pow(torch.abs(td) + self.per_epsilon, self.per_alpha).item()

        self.buffer.store([priority, (state, action, reward, 1. - done, next_state)])

    def get_loss(self, states, actions, target, importance_weights):
        td_error = target.unsqueeze(1) - self.q(states).gather(1, actions)

        priorities = torch.pow(torch.abs(td_error.detach()) + self.per_epsilon, self.per_alpha).squeeze(1).tolist()

        loss = torch.pow(td_error, 2) * importance_weights
        loss = loss.mean()

        return loss, priorities

    def sample_data(self):
        states, actions, rewards, dones, next_states, indices, importance_weights = self.buffer.sample(self.batch_size)

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
        importance_weights = torch.tensor(importance_weights, device=self.device).unsqueeze(1).detach().float()

        if self.normalize_reward:
            rewards = (rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + 1e-6)

        return states, actions, rewards, dones, next_states, indices, importance_weights

    def update(self):
        states, actions, rewards, dones, next_states, indices, importance_weights = self.sample_data()

        y = self.get_target(rewards, dones, next_states)

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
    run(PrioritizedDQN)