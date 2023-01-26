# Simple Reinforcement Learning Algorithm implementation

This is simple implementation for reinforcement learning algorithms for study and understanding of the baselines.

## Requirements


### required
- python 3 (3.8 or 3.9 recommended)
- pytorch (1.13.0 <= recommended)
- gym (0.26 <=)
- opencv-python
- matplotlib
- tqdm

# Implemented Algorithms


| Type       | Algorithms      | Papers                                                                                | Note                   |
|------------|-----------------|---------------------------------------------------------------------------------------|------------------------|
| Model-free | DQN             | [link](https://arxiv.org/abs/1312.5602)                                               | Discrete only          |
| Model-free | Double DQN      | [link](https://arxiv.org/abs/1509.06461)                                              | Discrete only          |
| Model-free | Prioritized DQN | [link](https://arxiv.org/abs/1511.05952)                                              | Discrete only          |
| Model-free | Dueling DQN     | [link](https://arxiv.org/abs/1511.06581)                                              | Discrete only          |
| Model-free | Noisy DQN       | [link](https://arxiv.org/abs/1706.10295)                                              | Discrete only          |
| Model-free | C51             | [link](https://arxiv.org/abs/1707.06887)                                              | Discrete only          |                                                                                            
| Model-free | RAINBOW         | [link](https://arxiv.org/abs/1710.02298)                                              | Discrete only          |
| Model-free | QR-DQN          | [link](https://arxiv.org/abs/1710.10044)                                              | Discrete only          |
| Model-free | IQN             | [link](https://arxiv.org/abs/1806.06923)                                              | Discrete only          |
| Model-free | A2C             | [link](https://arxiv.org/abs/1602.01783)                                              | Discrete/Continuous    |
| Model-free | A3C             | [link](https://arxiv.org/abs/1602.01783)                                              | Discrete/Continuous    |
| Model-free | PPO(Clip)       | [link](https://arxiv.org/abs/1707.06347)                                              | Discrete/Continuous    |
| Model-free | PPO(Penalty)    | [link](https://arxiv.org/abs/1707.06347)                                              | Discrete/Continuous    |
| Model-free | DDPG            | [link](https://arxiv.org/abs/1509.02971)                                              | Continuous only        |
 | Model-free | TD3             | [link](https://arxiv.org/abs/1802.09477)                                              | Continuous only        |
| Model-free | SAC             | [link](https://arxiv.org/abs/1801.01290)<br/>[link](https://arxiv.org/abs/1812.05905) | Continuous only        |

# Usage

You can simply run each algorithm by running corresponding python file.

For example, if you want to run DQN file with CartPole-v1 environment, you can just do it with following codes on the cmd window.
    
    C:\Path to DQN.py>python DQN.py -e=CartPole-v1 -m=100000 -x=0 -f=0 -v=-1 --train --view --test
Options for running command are as follows.

| Options  | Descriptions             | Note                                                                    |
|----------|--------------------------|-------------------------------------------------------------------------|
| -e       | environment name         | Specified environment name                                              |
| -s       | seed                     | Random seed                                                             |
| -m       | max iteration            | Maximum training iteration                                              |
| -x       | exploration              | Iteration for collecting data with random action                        |
| -f       | test/save frequency      | Test and save current model at every specified episodes during training |
| -w       | the number for workers   | The number of workers used with distributed algorithms                  |
| -v       | model version            | Saved model version. if specified with -1, it means creating new one    |
| --train  | train model              | If specified, train the model                                           |
| --test   | test model               | If specified, test the model                                            |
| --view   | view training graph      | If specified, view the training graph                                   |
| --render | render during training   | Not recommended to use                                                  |


You have to write your own hyperparameter files as .json to use in unknown or custom environments.

Hyperparameter files are available for each algorithm, by placing them in a directory named "hyperparameters", which is in the same directory with algorithm.py file

Each hyperparameter file must have a name corresponding to the environment it represents.

Ex) If you want to add hyperparameter for CartPole-v1 using DQN, you have to create CartPole-v1.json in the hyperparameters directory next to the DQN.py file.

If you want to change hyperparameters, you can simply modify existing hyperparameter json file by yourself. 


