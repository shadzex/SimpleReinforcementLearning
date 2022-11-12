# Asynchronous Advantage Actor Critic

import torch
from torch.distributions import Categorical, MultivariateNormal

import numpy as np

from os.path import isfile

from base import DistributedWorker, DistributedRunner
from RL.MFRL.A2C.A2C.A2C import Actor, V
from RL.env.env import Environment
from optimizer import SharedOptimizer


class A2C(DistributedWorker):
    def __init__(self,
                 worker_id: int,

                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters,

                 score_queue,
                 info_queue,
                 max_iteration: int,
                 explore_iteration: int,

                 global_actor: Actor,
                 global_critic: V,
                 optimizer: SharedOptimizer,

                 render: bool = False):
        super().__init__(worker_id,

                         device,
                         seed,

                         env_info,

                         hyperparameters,

                         score_queue,
                         explore_iteration,

                         render
                         )

        self.info_queue = info_queue

        self.max_iteration = max_iteration

        self.global_actor = global_actor
        self.global_critic = global_critic
        self.optimizer = optimizer

        # Hyperparameters

        if self.action_space_type == 'continuous':
            self.std_init = hyperparameters.std_init
            self.std = self.std_init
            self.std_min = hyperparameters.std_min
            self.std_decay = hyperparameters.std_decay
            self.std_decay_rate = hyperparameters.std_decay_rate

            self.var = torch.full((self.action_num,), self.std_init * self.std_init).to(self.device)

        self.horizon = hyperparameters.horizon

        self.normalize_state = hyperparameters.normalize_state
        self.normalization_method = hyperparameters.normalization_method
        self.normalize_reward = hyperparameters.normalize_reward
        self.reward_scale = hyperparameters.reward_scale
        self.value_coefficient = hyperparameters.value_coefficient
        self.entropy_coefficient = hyperparameters.entropy_coefficient

        self.build()

    def build_memories(self):

        self.states = []
        self.actions = []
        self.rewards = []

    def build_networks_and_optimizers(self):
        # Networks
        self.actor = Actor(self.state_dim, self.action_space_type, self.action_num, self.hyperparameters.actor).to(self.device)
        self.critic = V(self.state_dim, self.hyperparameters.critic).to(self.device)

    def reset(self):
        self.actor.load_state_dict(self.global_actor.state_dict())
        self.critic.load_state_dict(self.global_critic.state_dict())

    def explore(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        if self.action_space_type == 'discrete':
            probs = self.actor(state)
            policy = Categorical(probs=probs)
            action = policy.sample()
            action = action.item()

        else:
            mu = self.actor(state)
            cov_mat = torch.diag(self.var).unsqueeze(dim=0)

            policy = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)
            action = policy.rsample()
            action = action.tolist()[0]

        return action

    def process(self,
                state,
                action,
                next_state,
                reward: float,
                done: bool,
                info: dict):

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.last_state = next_state

        self.done = done

        self.iteration += 1

    def update(self):

        last_state = self.preprocess_inputs(self.last_state, self.state_normalizer)
        last_value = self.critic(last_state)
        last_value = last_value.item()

        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)

        r = (1.0 - self.done) * last_value
        returns = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            r = rewards[t] + self.discount_factor * r
            returns[t] = r

        # Normalizing the returns
        if self.normalize_reward:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        returns = self.reward_scale * returns

        # Train
        states = self.preprocess(states)

        self.state_normalizer.update(states)
        states = self.state_normalizer.normalize(states)

        states = torch.tensor(states, device=self.device).detach().float()
        actions = torch.tensor(actions, device=self.device).detach()
        returns = torch.tensor(returns, device=self.device).detach().float()

        values = self.critic(states).squeeze()

        if self.action_space_type == 'discrete':
            probs = self.actor(states)
            policy = Categorical(probs=probs)

        else:
            mu = self.actor(states)
            var = self.var.expand_as(mu)
            cov_mat = torch.diag_embed(var).to(self.device)
            policy = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)

        log_probs = policy.log_prob(actions)

        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()

        critic_loss = self.value_coefficient * advantages.pow(2).mean()

        entropy = -self.entropy_coefficient * policy.entropy().mean()

        loss = actor_loss + critic_loss + entropy

        self.optimizer.step(loss, [self.actor, self.critic])

        if self.action_space_type == 'continuous' and self.iteration % self.std_decay_rate == 0:
            self.std = max(self.std - self.std_decay, self.std_min)
            self.var = torch.full((self.action_num,), self.std * self.std).to(self.device)

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

        train_info = {'actor_loss': actor_loss.item(),
                      'critic_loss': critic_loss.item()}

        hyperparameter_info = {'lr': self.optimizer.get_lr(),
                               }

        return 1, train_info, hyperparameter_info

    def trainable(self, *args):
        return self.iteration % self.horizon == 0 or self.done

    def run(self):
        env = Environment(self.env, self.seed, render=self.render)

        score = 0.
        discounted_score = 0.

        state = env.reset()
        self.reset()
        done = False

        for iteration in range(self.max_iteration):

            if iteration < self.explore_iteration:
                action = env.sample()
            else:
                action = self.explore(state)

            next_state, reward, done, info = env.step(action)

            self.process(state, action, next_state, reward, done, info)

            state = next_state

            if iteration >= self.explore_iteration and self.trainable():
                gradient_step, train_info, hyperparameter_info = self.update()

                self.info_queue.put((train_info, hyperparameter_info))
            else:
                self.info_queue.put((None, None))

            score += reward
            discounted_score = reward + self.discount_factor * discounted_score

            if done:
                self.score_queue.put([self.worker_id, score, discounted_score])

                score = 0.
                discounted_score = 0.

                state = env.reset()
                self.reset()
                done = False

        env.close()


class A3C(DistributedRunner):
    def __init__(self,
                 # Experimental Configurations
                 device: str,
                 seed: int,

                 # Environmental Configurations
                 env_info,

                 hyperparameters):
        super().__init__('cpu',
                         seed,

                         env_info,

                         hyperparameters)

        self.build()

    def build_memories(self):
        return

    def build_networks_and_optimizers(self):
        # Networks
        self.actor = Actor(self.state_dim, self.action_space_type, self.action_num, self.hyperparameters.actor)
        self.critic = V(self.state_dim, self.hyperparameters.critic)

        self.actor.share_memory()
        self.critic.share_memory()

        self.optimizer = SharedOptimizer([self.actor, self.critic], hyperparameters=self.hyperparameters.optimizer)

    def act(self, state):
        state = self.preprocess_inputs(state, self.state_normalizer)

        if self.action_space_type == 'discrete':
            probs = self.actor(state)

            action = torch.argmax(probs).item()
        else:
            action = self.actor(state)

            action = action.tolist()[0]

        return action

    def init(self,
             seed,
             log_path,
             worker_num,
             max_iteration,
             explore_iteration,
             render):
        super(A3C, self).init(seed,
                              log_path,
                              worker_num,
                              max_iteration,
                              explore_iteration,
                              render)

        self.workers = [A2C(worker_id,

                            self.device,
                            seed,

                            self.env_info,

                            self.hyperparameters,

                            self.score_queue,
                            self.info_queue,
                            max_iteration,
                            explore_iteration,

                            self.actor,
                            self.critic,

                            render if not worker_id else False) for worker_id in range(worker_num)]

    def run(self,
            env,
            max_iteration,
            ):
        for worker in self.workers:
            worker.start()

        self.logger.start()

    def close(self,
              model_path):
        for worker in self.workers:
            worker.join()

        super(A3C, self).close(model_path)

    def save_models(self,
                    path: str):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict()
                    }, path)

    def load_models(self,
                    path: str,
                    test: bool = False):
        # Pre-build

        if isfile(path):
            checkpoint = torch.load(path)

            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])

if __name__ == '__main__':
    from runner import run_distributed
    run_distributed(A3C)