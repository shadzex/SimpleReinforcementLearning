import gym
from gym.spaces import Box, Discrete

from gym.wrappers.atari_preprocessing import AtariPreprocessing

import numpy as np

def get_all_envs():
    return gym.envs.registry.all()

# Wrapper class for environments
class Environment:
    def __init__(self, name, seed=None, action_repeat=1, visual_wrapping=True, render=True):

        self.env_name = name

        self.action_repeat = action_repeat

        self._render = render

        self.seed = seed

        self.env = gym.make(name, render_mode='human' if render else None)

        self.state_dim = list(self.env.observation_space.shape)

        if len(self.state_dim) == 3 and visual_wrapping:
            self.env = AtariPreprocessing(self.env)

        if isinstance(self.env.action_space, Discrete):
            self.action_num = self.env.action_space.n
            self.action_space_type = 'discrete'
            self.action_lower_bound = None
            self.action_upper_bound = None
            self.action_scale = None
            self.action_bias = None
        elif isinstance(self.env.action_space, Box):
            self.action_num = self.env.action_space.shape[0]
            self.action_space_type = 'continuous'
            self.action_lower_bound = self.env.action_space.low[0]
            self.action_upper_bound = self.env.action_space.high[0]
            self.action_scale = float((self.action_upper_bound - self.action_lower_bound) / 2.)
            self.action_bias = float((self.action_upper_bound + self.action_lower_bound) / 2.)

        try:
            self.goal_dim = list(self.env.goal_space.shape)
        except:
            try:
                self.reset()
                self.goal_dim = list(self.goal.shape)
            except:
                self.goal_dim = [1]

    def reset(self):
        observation, info = self.env.reset(seed=self.seed)

        return observation

    def step(self, action, *args):

        state, reward, terminated, truncated, info = self.env.step(action)
        for _ in range(self.action_repeat - 1):
            state, reward_, terminated, truncated, info = self.env.step(action)
            reward += reward_

        done = terminated or truncated
        return state, reward, done, info

    # If we need actual goal
    @property
    def goal(self):
        try:
            return self.env.goal
        except:
            return np.zeros((1,))

    # If we need to access to the reward function, use this function
    def reward_function(self, state, achieved_goal, goal, action, next_state, info):
        return self.env.compute_reward(achieved_goal, goal, info)

    # get expert demonstration
    def demonstrate(self):
        return
    
    def sample(self):
        return self.env.action_space.sample()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()