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

    def get_target(self, rewards, dones, next_states):
        with torch.no_grad():
            target_actions = self.q(next_states).argmax(1, keepdim=True)
            y = self.reward_scale * rewards + self.discount_factor * dones * self.q_target(next_states).gather(1,target_actions).squeeze(1)

        return y

if __name__ == '__main__':
    from runner import run
    run(DoubleDQN)