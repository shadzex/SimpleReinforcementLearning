# Common train or test functions
import importlib
from os.path import isdir
from os import makedirs, listdir

from viewer import Viewer

import argparse

from base import BaseRLAlgorithm, DistributedRunner, Hyperparameters
from RL.env.env import Environment

from logger import Logger

import torch

import re

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
    env = Environment(env_name)

    env_info = EnvironmentInfo()

    env_info.env = env_name
    env_info.state_dim = env.state_dim
    env_info.goal_dim = env.goal_dim
    env_info.action_num = env.action_num
    env_info.action_space_type = env.action_space_type
    env_info.action_scale = env.action_scale

    env.close()

    return env_info

def default_setting(base_path, name, version):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    hyperparameters = Hyperparameters()
    hyperparameters.load(base_path + '/hyperparameters/{}.json'.format(name))

    base_model_path = base_path + '/models/{}'.format(name)
    base_log_path = base_path + '/logs/{}'.format(name)

    if not isdir(base_model_path):
        try:
            makedirs(base_model_path)
        except:
            print('폴더 생성 실패')
            exit()

    if not isdir(base_log_path):
        try:
            makedirs(base_log_path)
        except:
            print('폴더 생성 실패')
            exit()

    if version == -1:
        dir_list = list({int(re.sub(r'_\D*', '', file_name)) for file_name in listdir(base_model_path)})
        if len(dir_list) == 0:
            numbering = 0
        else:
            dir_list.sort()
            numbering = dir_list[-1] + 1
    else:
        numbering = version

    model_path = base_model_path + '/{}'.format(numbering)
    log_path = base_log_path + '/{}'.format(numbering)

    return device, hyperparameters, model_path, log_path

def run(algorithm_class):
    parser = argparse.ArgumentParser(description='Input Argument')

    # Arguments
    parser.add_argument('-e', '--env', type=str, default='Pendulum-v0') # Environment
    parser.add_argument('-s', '--seed', type=int, default=0) # Seed
    parser.add_argument('-v', '--version', type=int, default=-1) # Model version
    parser.add_argument('-f', '--evaluate_freq', type=int, default=-1) # Save/evaluation frequency
    parser.add_argument('-w', '--worker_num', type=int, default=4) # Worker number for distributed algorithms, disabled for non-distributed algorithms
    parser.add_argument('-m', '--max_iteration', type=int, default=1000000) # Max iteration for training
    parser.add_argument('-x', '--explore_iteration', type=int, default=25000) # First some iterations can be used for random exploration with random action
    parser.add_argument('--train', action='store_true') # Decide to train
    parser.add_argument('--view', action='store_true') # Decide to view the graph of the log
    parser.add_argument('--best', action='store_true') # Decide to test with the model evaluated during training with the best score
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
    best = args.best
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
        if isinstance(algorithm, DistributedRunner):
            algorithm.train(env, seed, log_path, model_path, evaluate_freq, worker_num, max_iteration, explore_iteration, render=render)
        else:
            algorithm.train(env, seed, log_path, model_path, evaluate_freq, max_iteration, explore_iteration, render=render)

    if view:
        viewer = Viewer(log_path, 10)
        viewer.view()

    if test:
        if best:
            algorithm.load_models(model_path + '_best')
        else:
            algorithm.load_models(model_path)
        algorithm.test(env)
