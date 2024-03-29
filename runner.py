# Common train or test functions

from os.path import isdir
from os import makedirs, listdir

from viewer import Viewer

import argparse

from base import DistributedRunner, Hyperparameters
from env.env import Environment

from datetime import datetime

import torch

class EnvironmentInfo:
    def __init__(self):
        self.env = ''
        self.state_dim = []
        self.goal_dim = []
        self.action_space_type = 'discrete'
        self.action_num = 0
        self.action_scale = 1.0

    def copy(self):
        info = EnvironmentInfo()
        info.env = self.env
        info.state_dim = self.state_dim.copy()
        info.goal_dim = self.goal_dim.copy()
        info.action_space_type = self.action_space_type
        info.action_num = self.action_num
        info.action_scale = self.action_scale

        return info

def get_env_info(env_name):
    env = Environment(env_name, render=False)

    env_info = EnvironmentInfo()

    env_info.env = env_name
    env_info.state_dim = env.state_dim
    env_info.goal_dim = env.goal_dim
    env_info.action_num = env.action_num
    env_info.action_space_type = env.action_space_type
    env_info.action_scale = env.action_scale

    env.close()

    return env_info

def path_check(path):
    if not isdir(path):
        try:
            makedirs(path)
        except FileNotFoundError:
            print('Failed to create directory')
            exit()

def default_setting(base_path, name, version):
    now = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss')

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    hyperparameters = Hyperparameters()
    hyperparameters.load(base_path + '/hyperparameters/{}.json'.format(name))

    base_model_path = base_path + '/models/{}'.format(name)
    base_log_path = base_path + '/logs/{}'.format(name)

    path_check(base_model_path)
    path_check(base_log_path)

    file_names = [file_name for file_name in listdir(base_model_path) if file_name[-1].isdecimal()]
    numberings = [int(file_name[-1]) for file_name in file_names]
    if version == -1:
        # Create new numbering for new model
        numbering = len(file_names)
        file_name = '/{}_version_{}'.format(now, numbering)
    else:
        # Find existing numbering or create new one
        try:
            file_name = '/' + file_names[numberings.index(version)]
        except ValueError:
            file_name = '/{}_version_{}'.format(now, version)

    model_path = base_model_path + file_name
    log_path = base_log_path + file_name

    return device, hyperparameters, model_path, log_path

def run(algorithm_class):
    parser = argparse.ArgumentParser(description='Input Argument')

    # Arguments
    parser.add_argument('-e', '--env', type=str, default='CartPole-v1') # Environment
    parser.add_argument('-s', '--seed', type=int, default=1) # Seed
    parser.add_argument('-v', '--version', type=int, default=-1) # Model version
    parser.add_argument('-f', '--evaluate_freq', type=int, default=-1) # Save/evaluation frequency
    parser.add_argument('-w', '--worker_num', type=int, default=4) # Worker number for distributed algorithms, disabled for non-distributed algorithms
    parser.add_argument('-m', '--max_iteration', type=int, default=1000000) # Max iteration for training
    parser.add_argument('-x', '--explore_iteration', type=int, default=0) # First some iterations can be used for random exploration with random action
    parser.add_argument('--train', action='store_true') # Decide to train
    parser.add_argument('--view', action='store_true') # Decide to view the graph of the log
    parser.add_argument('--test', action='store_true') # Decide to test
    parser.add_argument('--render', action='store_true') # Decide to render during training(Automatically true for the test)

    args = parser.parse_args()

    seed = args.seed
    version = args.version
    evaluate_freq = args.evaluate_freq
    worker_num = args.worker_num
    max_iteration = args.max_iteration
    explore_iteration = args.explore_iteration

    train = args.train
    view = args.view
    test = args.test

    env = args.env

    env_info = get_env_info(env)

    render = args.render

    device, hyperparameters, model_path, log_path = default_setting('.', env, version)

    algorithm = algorithm_class(device,
                                seed,

                                env_info,

                                hyperparameters)

    if train:
        # If the algorithm is distributed,
        if not isinstance(algorithm, DistributedRunner):
            worker_num = 1
        algorithm.train(log_path, model_path, evaluate_freq, worker_num, max_iteration, explore_iteration, render=render)

    if view:
        viewer = Viewer(log_path, 10)
        viewer.view()

    if test:
        algorithm.load_models(model_path)
        algorithm.test(env)
