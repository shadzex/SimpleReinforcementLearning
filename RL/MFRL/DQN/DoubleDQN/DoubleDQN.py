# Double Deep Q Network

import torch
import torch.nn.functional as f

from utils import soft_update

from RL.MFRL.DQN.DQN.DQN import DQN

class DoubleDQN(DQN):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):
        super(DoubleDQN, self).__init__(device,
                                        seed,

                                        env_info,

                                        hyperparameters)

    def update(self):
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

        with torch.no_grad():
            target_actions = self.q(next_states).argmax(1, keepdim=True)
            y = self.reward_scale * rewards + self.discount_factor * dones * self.q_target(next_states).gather(1, target_actions).squeeze(1)

        loss = f.mse_loss(self.q(states).gather(1, actions), y.unsqueeze(1))

        self.optimizer.step(loss)

        if self.iteration % self.target_update_rate == 0:
            soft_update(self.q_target, self.q, self.tau)

        self.iteration += 1

        train_info = {'loss': loss.item()}

        hyperparameter_info = {'lr': self.optimizer.get_lr()
                               }

        return 1, train_info, hyperparameter_info

if __name__ == '__main__':
    from runner import run
    run(DoubleDQN)