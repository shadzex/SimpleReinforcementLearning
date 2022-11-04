# Logger
# Log scores, losses, and hyperparameters

import numpy as np

from os.path import isfile

import pickle

from tqdm import tqdm

class Logger:
    def __init__(self, load_path, save_path, max_iteration, worker_num, episode_for_log=100):

        self.worker_num = worker_num
        self.episode_for_log = episode_for_log

        # 로그 결과가 주어진 파일 명에 저장되었는지 확인하고, 있는 경우 해당 파일에서 불러옴
        self.save_path = save_path
        iteration, init_storage = self.load(load_path)

        if init_storage == None:
            self.scores = {i: {'scores': [],
                               'discounted_scores': [],
                               'max_score': float('-inf'),
                               'avg_scores': [],
                               'avg_discounted_scores': [],
                               'avg_score_for_episode': [],
                               'sum_score': 0.,
                               'sum_discounted_score': 0.,
                               'episode': 0} for i in range(worker_num)}
            self.info = {}
            self.test_scores = []
            self.hyperparameters = {}
        else:
            self.scores = init_storage[0]
            self.info = init_storage[1]
            self.test_scores = init_storage[2]
            self.hyperparameters = init_storage[3]

        self.iteration = iteration

        self.max_iteration = max_iteration

    def reset(self):

        self.iteration = self.get_iteration()

        self.pbar = tqdm(total=self.max_iteration + self.iteration, desc='Train Progress', initial=self.iteration)


        return self.iteration

    def log(self, iteration=1):

        self.pbar.update(iteration)

        self.iteration += iteration

    def log_score(self, data):
        id, score, discounted_score = data

        self.scores[id]['scores'].append(score)
        self.scores[id]['discounted_scores'].append(discounted_score)
        if score >= self.scores[id]['max_score']:
            self.scores[id]['max_score'] = score

        self.scores[id]['episode'] += 1

        self.scores[id]['sum_score'] += score
        self.scores[id]['sum_discounted_score'] += discounted_score

        self.scores[id]['avg_scores'].append(self.scores[id]['sum_score'] / self.scores[id]['episode'])
        self.scores[id]['avg_discounted_scores'].append(self.scores[id]['sum_discounted_score'] / self.scores[id]['episode'])
        self.scores[id]['avg_score_for_episode'].append(np.mean(self.scores[id]['scores'][-self.episode_for_log:]))

    def log_evaluation(self, score):
        self.test_scores.append(score)

    def log_train(self, info):
        if len(self.info) == 0:
            self.info = {i: {'name': '', 'values': [], 'avg_values': [], 'sum_value': 0., 'iteration': 0} for i in range(len(info))}

        for i, (key, value) in enumerate(info.items()):
            if self.info[i]['name'] == '':
                self.info[i]['name'] = key

            if value != None:
                self.info[i]['values'].append(value)

                self.info[i]['iteration'] += 1

                self.info[i]['sum_value'] += value

                self.info[i]['avg_values'].append(self.info[i]['sum_value'] / self.info[i]['iteration'])

    def log_hyperparamters(self, info):
        self.hyperparameters = info

    def log_info(self, train_info, hyperparameter_info):
        self.log_train(train_info)
        self.log_hyperparamters(hyperparameter_info)

    def get_iteration(self):
        return self.iteration

    def save(self):
        with open(self.save_path, 'wb') as file:
            log = {
                'iteration': self.iteration,
                'score_storage': self.scores,
                'info_storage': self.info,
                'test_storage': self.test_scores,
                'hyperparameter_storage': self.hyperparameters
            }

            pickle.dump(log, file)

    def load(self, load_path):
        if isfile(load_path):
            with open(load_path, "rb") as file:
                log = pickle.load(file)

                iteration = log['iteration']
                score_storage = log['score_storage']
                info_storage = log['info_storage']
                test_storage = log['test_storage']
                hyperparameter_storage = log['hyperparameter_storage']

            return iteration, [score_storage, info_storage, test_storage, hyperparameter_storage]

        return 0, None

    def close(self):
        self.pbar.close()