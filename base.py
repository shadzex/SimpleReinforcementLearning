import torch

import random
import numpy as np

from utils import Normalizer

import json

from RL.env.env import Environment

from logger import Logger

from multiprocessing import Queue, Process

from copy import deepcopy

# Hyperparameter class
# Read hyperparameter json file and convert it to the instance of this class
class Hyperparameters(dict):
    def __init__(self):
        super(Hyperparameters, self).__init__()
        self.hyperparameters = {}

    def __repr__(self):
        print(self.hyperparameters)

    def set(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def __contains__(self, item):
        return item in self.hyperparameters.keys()

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __getattr__(self, item):
        if item in self.hyperparameters.keys():
            value = self.hyperparameters[item]

            if isinstance(value, dict):
                sub_hyperparameters = Hyperparameters()
                sub_hyperparameters.set(value)

                return sub_hyperparameters
            else:
                return value
        else:
            return None

    def __setstate__(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def __getstate__(self):
        return self.hyperparameters

    def __deepcopy__(self, memodict={}):
        new_hyperparameters = Hyperparameters()
        new_hyperparameters.set(deepcopy(self.hyperparameters))

        return new_hyperparameters

    def load(self, path):
        with open(path) as f:
            self.hyperparameters = json.load(f)

# Base Algorithm class
class BaseAlgorithm:
    def __init__(self,
                 device: str,
                 seed: int,

                 hyperparameters):

        # Experiment Configurations
        # Set training device
        if device:
            self.device = torch.device(device)
        else:
            self.device = None

        # Set random seed
        self.seed = seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Set hyperparameters
        self.hyperparameters = hyperparameters

    # Builders
    # Builder methods are for initializing networks, losses, optimizers, memories, and normalizers
    # Builders must be called inside the init method

    # Build
    def build(self):
        # Networks and Optimizers
        self.build_networks_and_optimizers()

        # Losses
        self.build_losses()

        # Others
        self.build_others()

    # Build networks and normalizers
    # Networks and normalizers must be in the same method, or networks cannot be trained
    def build_networks_and_optimizers(self):
        return

    # Build losses
    def build_losses(self):
        return

    # Build others
    def build_others(self):
        return

    # Preprocessing
    def preprocess(self, *args):
        raise NotImplementedError

    # Actual update and optimization method
    def update(self, *args):
        raise NotImplementedError

    # Change network's mode
    # From train to eval or eval to train
    def change_train_mode(self, mode):
        raise NotImplementedError

    # A method to train within instantiated class
    # To train, this method or another way such as custom function to train can be used
    def train(self, *args):
        self.change_train_mode(True)

    # A method to test within instantiated class
    def test(self, *args):
        self.change_train_mode(False)

    # Save neural network models
    def save_models(self, *args):
        raise NotImplementedError

    # Load neural network models
    def load_models(self, *args):
        raise NotImplementedError


# Base RL algorithm class
class BaseRLAlgorithm(BaseAlgorithm):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):
        super(BaseRLAlgorithm, self).__init__(device,
                                              seed,

                                              hyperparameters)

        # Environment Configurations
        self.env_info = env_info

        self.env = env_info.env
        self.state_dim = env_info.state_dim
        self.state_dim.reverse()
        self.goal_dim = env_info.goal_dim
        self.goal_dim.reverse()
        self.action_space_type = env_info.action_space_type
        self.action_num = env_info.action_num
        self.action_scale = env_info.action_scale

        self.discount_factor = hyperparameters.discount_factor

        # Others
        self.iteration = 0

        self.best_score = -np.inf

    # Overrided build function
    def build(self):
        super(BaseRLAlgorithm, self).build()

        # Memories
        self.build_memories()

        # Others

        if self.hyperparameters.normalize_state:
            self.normalization_method = self.hyperparameters.normalization_method
        else:
            self.normalization_method = None

        self.build_normalizers(self.normalization_method)

    # Build memories
    def build_memories(self):
        raise NotImplementedError

    # Build normalizers
    # Always build state and goal normalizers.
    # Free to use both
    def build_normalizers(self, method):
        self.state_normalizer = Normalizer(self.state_dim, method)
        self.goal_normalizer = Normalizer(self.goal_dim, method)

    # A method for initializing before the training begins
    # Mostly not necessary
    def init(self,
             env,
             max_iteration,
             explore_iteration,
             *args):
        self.reward_function = env.reward_function
        self.explore_iteration = explore_iteration

    # Reset function used at the beginning of the episode
    def reset(self, *args):
        raise NotImplementedError

    # Preprocessing
    def preprocess(self, state, *args):
        # shape checks are including batch
        if np.ndim(state) == 4:
            state = np.transpose(state, (0, 3, 2, 1))
            # Image data preprocessing
            if state.dtype == np.uint8:
                return state / 255.
            else:
                return state
        else:
            # Vector data preprocessing
            return state

    # Preprocessing input
    def preprocess_inputs(self, inputs, normalizer):
        inputs = np.expand_dims(inputs, axis=0)
        inputs = self.preprocess(inputs)
        inputs = normalizer.normalize(inputs)

        inputs = torch.tensor(inputs, device=self.device).float()

        return inputs

    # A method for action selection with exploration while training
    def explore(self, *args):
        raise NotImplementedError

    # Deterministic action selection method for testing learned models
    def act(self, *args):
        raise NotImplementedError

    # Calculate intrinsic reward(Optional)
    def calculate_intrinsic_reward(self, *args):
        raise NotImplementedError

    # Processing each transition at each iteration
    def process(self, *args):
        raise NotImplementedError

    # Process method for evaluation
    # Called only in evaluate function
    def process_eval(self):
        return

    # If necessary, pretraining can be done
    def pretrain(self, *args):
        raise NotImplementedError

    # A method called after each update
    def postprocess(self, *args):
        return

    # Check trainability
    def trainable(self, *args):
        raise NotImplementedError

    # Called when the whole training ends
    def terminate(self, *args):
        return

    # Evaluate the current model
    # Check if current model is better than previous model
    def evaluate(self,
                 env: Environment,
                 max_episode: int = 10,
                 print_score: bool = False):
        scores = []

        for episode in range(1, max_episode + 1):
            score = 0.

            state = env.reset()
            self.goal = env.goal
            self.reset()
            done = False

            while not done:
                action = self.act(state)

                next_state, reward, done, info = env.step(action)

                self.process_eval()

                state = next_state

                score += reward

            if print_score:
                print('Episode {} Score: {}'.format(episode, score))

            scores.append(score)

        current_score = np.mean(scores)

        if current_score >= self.best_score:
            return True

        return False

    # Actual running method
    def train(self,
              env: str,
              seed: int,
              log_path: str,
              model_path: str,
              evaluate_freq: int = -1,
              max_iteration: int = 1000000,
              explore_iteration: int = 25000,
              render: bool = True,
              *args):
        super(BaseRLAlgorithm, self).train()
        
        logger = Logger(log_path, log_path, max_iteration, 1, 10)
        logger.reset()

        self.load_models(model_path)

        env = Environment(env, seed, render=render)

        self.init(env,
                  max_iteration,
                  explore_iteration)

        score = 0.
        discounted_score = 0.

        state = env.reset()
        self.goal = env.goal
        self.reset()
        done = False

        episode = 0
        for iteration in range(max_iteration):

            if iteration < explore_iteration:
                action = env.sample()
            else:
                action = self.explore(state)

            next_state, reward, done, info = env.step(action)

            self.process(state, action, next_state, reward, done, info)

            state = next_state

            if iteration >= explore_iteration and self.trainable():
                gradient_step, train_info, hyperparameter_info = self.update()

                logger.log_train(train_info)
                logger.log_hyperparamters(hyperparameter_info)

            self.postprocess()

            logger.log()

            score += reward
            discounted_score = reward + self.discount_factor * discounted_score

            if done:
                episode += 1

                logger.log_score([0, score, discounted_score])

                score = 0.
                discounted_score = 0.

                if evaluate_freq > 0 and episode % evaluate_freq == 0:
                    update_best_model = self.evaluate(env)
                    self.save_models(model_path)
                    if update_best_model:
                        self.save_models(model_path + '_best')

                state = env.reset()
                self.goal = env.goal
                self.reset()
                done = False

        self.terminate()

        logger.save()
        logger.close()

        self.save_models(model_path)

        env.close()

    # Test method
    def test(self,
             env: str,
             max_episode: int = 10):
        super(BaseRLAlgorithm, self).test()
        
        env = Environment(env)

        self.evaluate(env, max_episode, print_score=True)

        self.terminate()

        env.close()
