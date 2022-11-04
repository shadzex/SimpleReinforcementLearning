# Simple Reinforcement Learning Algorithm implementation

This is simple implementation for reinforcement learning algorithms for study and understanding of the baselines.

## Requirements


### required
- python 3 (3.8 or 3.9 recommended)
- pytorch
- gym (0.26 <=)
- opencv-python
- matplotlib
- tqdm

# Implemented Algorithms


| Type       | Algorithms     | Papers                                     | Note                     |
|------------|----------------|--------------------------------------------|--------------------------|
| Model-free | Deep Q Network | [link](https://arxiv.org/abs/1312.5602)    | Discrete only            |
                                                                                                          

# Usage

You can simply run each algorithm by running corresponding python file.

For example, if you want to run DQN file with CartPole-v1 environment, you can just do it with following codes on the cmd window.
    
    C:\Path to DQN.py>python DQN.py -e=CartPole-v1 -m=100000 -x=0 -f=0 -v=-1 --train --view --test --best

Options for running command are as follows.

| Options  | Descriptions             | note                                                                    |
|----------|--------------------------|-------------------------------------------------------------------------|
| -e       | environment name         | Specified environment name                                              |
| -s       | seed                     | Random seed                                                             |
| -m       | max iteration            | Maximum training iteration                                              |
| -x       | exploration              | Iteration for collecting data with random action                        |
| -f       | test/save frequency      | Test and save current model at every specified episodes during training |
| -v       | model version            | Saved model version. if specified with -1, it means creating new one    |
| --train  | train model              | If specified, train the model                                           |
| --test   | test model               | If specified, test the model                                            |
| --view   | view training graph      | If specified, view the training graph                                   |
| --best   | test with the best model | If specified, test with the model with the best performance             |
| --render | render during training   | Not recommended to use                                                  |


You have to write your own hyperparameter files as .json to use in unknown or custom environments.

Hyperparameter files are available for each algorithm, by placing them in a directory named "hyperparameters", which is in the same directory with algorithm.py file

Each hyperparameter file must have a name corresponding to the environment it represents.

Ex) If you want to add hyperparameter for CartPole-v1 using DQN, you have to create CartPole-v1.json in the hyperparameters directory next to the DQN.py file.

If you want to change hyperparameters, you can simply modify existing hyperparameter json file by yourself. 


