# Dueling Deep Q Network

from utils import hard_update

from MFRL.DQN.DoubleDQN.DoubleDQN import DoubleDQN

from network import GeneralNetwork
from layers import Linear
from optimizer import Optimizer

class Q(GeneralNetwork):
    def __init__(self,
                 state_dim,
                 action_num,

                 hyperparameters):
        super(Q, self).__init__(state_dim,

                                hyperparameters)

        self.v = Linear(self.output_dim, action_num)
        self.a = Linear(self.output_dim, action_num)

    def forward(self, state):
        x = super(Q, self).forward(state)

        value = self.v(x)
        advantage = self.a(x)

        q = value + (advantage - advantage.mean(1).unsqueeze(1))

        return q

class DuelingDQN(DoubleDQN):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):
        super(DuelingDQN, self).__init__(device,
                                         seed,

                                         env_info,

                                         hyperparameters)

    def build_networks_and_optimizers(self):
        # Networks
        self.q = Q(self.state_dim, self.action_num, self.hyperparameters.q).to(self.device)
        self.q_target = Q(self.state_dim, self.action_num, self.hyperparameters.q).to(self.device)

        hard_update(self.q_target, self.q)

        # Optimizers
        self.optimizer = Optimizer(self.q, self.hyperparameters.optimizer)

if __name__ == '__main__':
    from runner import run
    run(DuelingDQN)