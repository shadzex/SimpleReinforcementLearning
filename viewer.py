# View the result of the training

import matplotlib.pyplot as plt

import pickle

class Viewer:
    def __init__(self, file_name, episode_for_log, ids=None):
        with open(file_name, "rb") as file:
            log = pickle.load(file)

            self.score_storage = log['score_storage']
            self.info_storage = log['info_storage']
            self.hyperparameter_storage = log['hyperparameter_storage']

        self.episode_for_log = episode_for_log

        self.ids = ids

    def view(self):

        worker_num = len(self.score_storage)
        train_info_num = len(self.info_storage)

        if self.ids == None:
            self.ids = [i for i in range(worker_num)]

        fig = plt.figure(figsize=(12, 8))

        score_axes = []
        train_axes = []

        for i in range(len(self.ids)):
            ax = fig.add_subplot(len(self.ids), 2, i * 2 + 1)
            score_axes.append(ax)

        for i in range(train_info_num):
            ax = fig.add_subplot(train_info_num + 1, 2, (i + 1) * 2)
            train_axes.append(ax)

        hyperparameter_ax = fig.add_subplot(train_info_num + 1, 2, (train_info_num + 1) * 2)
        hyperparameter_ax.axis('off')

        ax_id = 0
        for id, data in self.score_storage.items():

            scores = data['scores']
            max_score = data['max_score']
            avg_scores = data['avg_scores']
            avg_discounted_scores = data['avg_discounted_scores']
            avg_scores_for_episode = data['avg_score_for_episode']

            if id in self.ids:
                score_axes[ax_id].plot(list(range(1, len(scores) + 1)), scores, 'b-', label='Score')
                score_axes[ax_id].plot(list(range(1, len(avg_scores) + 1)), avg_scores, 'r-',
                                       label='Average Score')
                score_axes[ax_id].plot(list(range(1, len(avg_discounted_scores) + 1)), avg_scores, 'y-',
                                       label='Average Discounted Score')
                score_axes[ax_id].plot(list(range(1, len(avg_scores_for_episode) + 1)),
                                       avg_scores_for_episode, 'm-',
                                       label='Average Score For {} Episodes'.format(self.episode_for_log))

                score_axes[ax_id].set_xlabel('max_score: {}'.format(max_score))
                score_axes[ax_id].set_title('Worker {}'.format(id))
                score_axes[ax_id].legend(loc='upper left', prop={'size': 6})

                ax_id += 1

        for i, (key, data) in enumerate(self.info_storage.items()):

            name = data['name']
            values = data['values']
            avg_values = data['avg_values']

            train_axes[i].plot(list(range(1, len(values) + 1)), values, 'y-', label='Value')
            train_axes[i].plot(list(range(1, len(avg_values) + 1)), avg_values, 'k-', label='Average Value')

            if name == '':
                name = 'Default{}'.format(i)

            train_axes[i].set_title(name)

            train_axes[i].legend(loc='upper left', prop={'size': 6})

        hyperparameter_text_box = hyperparameter_ax.text(0.002, 0.00001, '', fontsize=12)
        hyperparameter_text_box.set_clip_on(False)

        if self.hyperparameter_storage != None:
            text = ''.join(['{}: {}\n'.format(key, value) for key, value in self.hyperparameter_storage.items()])
            hyperparameter_text_box.set_text(text)

        fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.6)

        plt.show()

