# Dueling Deep Q Network

from MFRL.DQN.DoubleDQN.DoubleDQN import DoubleDQN

from network import GeneralNetwork
from layers import Linear

# Dueling architecture
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

    def init_hyperparameters(self, hyperparameters):
        super(DuelingDQN, self).init_hyperparameters(hyperparameters)

        self.Q = Q

if __name__ == '__main__':
    from runner import run
    run(DuelingDQN)