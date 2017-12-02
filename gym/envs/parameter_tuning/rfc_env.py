import gym
import random
from gym import spaces
import numpy as np
from keras.datasets import cifar10, mnist, cifar100
from keras.utils import np_utils
from itertools import cycle
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def multi_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


class RandomForestEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, natural=False):
        self.action_space = spaces.Tuple((
            spaces.Box(10, 500, 1),  # n_est
            spaces.Box(4, 50, 1),  # max_depth
            spaces.Box(10, 50, 1)  # min_sample_leaves
        ))

        # observation features, in order: num of instances, num of labels,
        # number of filter in part A / B of neural net, num of neurons in
        # output layer, validation accuracy after training with given
        # parameters
        self.observation_space = spaces.Box(-1e5, 1e5, 4)

        # Start the first game
        self._reset()

    def _step(self, action):
        """
        Perform some action in the environment
        """
        # assert self.action_space.contains(action)

        n_est, max_depth, num_leaves = action

        # map ranges of inputs
        max_depth = int(max_depth[0])
        n_est = int(n_est[0])
        num_leaves = int(num_leaves[0])

        names = ["max_depth", "n_est", "num_leaves"]
        values = [max_depth, n_est, num_leaves]

        # for n, v in zip(names, values):
        #     print(n, v)

        X_train, y_train, X_valid, y_valid = self.data

        clf = RandomForestClassifier(n_estimators=n_est, n_jobs=-1,
                                     min_samples_leaf=num_leaves, max_depth=max_depth, random_state=42)

        # train model for one epoch_idx
        clf.fit(X_train, y_train)

        pred = clf.predict(X_valid)

        self.merror = multi_error(y_valid, pred)

        acc = accuracy_score(y_valid, pred)

        self.tacc = accuracy_score(y_train, clf.predict(X_train))

        # # save best validation
        # if acc > self.best_val:
        self.best_val = acc

        self.previous_acc = acc

        self.epoch_idx = self.epoch_idx + 1

        done = self.epoch_idx == 20

        if acc < .6:
            """ maybe not set to a very large value; if you get something nice,
            but then diverge, maybe it is not too bad
            """
            reward = -1.0
        else:
            reward = acc

            # as number of labels increases, learning problem becomes
            # more difficult for fixed dataset size. In order to avoid
            # for the agent to ignore more complex datasets, on which
            # accuracy is low and concentrate on simple cases which bring bulk
            # of reward, I normalize by number of labels in dataset

            reward = reward * self.nb_classes

            # formula below encourages higher best validation

            reward = reward + reward ** 2

        return self._get_obs(), reward, done, {}

    def _render(self, mode="human", close=False):
        if close:
            return
        print(">> Step ", self.epoch_idx, "best validation:", self.best_val)

    def _get_obs(self):
        """
        Observe the environment. Is usually used after the step is taken
        """
        # observation as per observation space
        return np.array([self.nb_classes,
                         self.nb_inst,
                         self.merror,
                         self.tacc])

    def data_mix(self):

            # randomly choose dataset
        dataset = random.choice(['mnist', 'cifar10', 'cifar100'])

        # tmp
        dataset = 'mnist'

        n_labels = 10

        if dataset == "mnist":
            data = mnist.load_data()

        if dataset == "cifar10":
            data = cifar10.load_data()

        if dataset == "cifar100":
            data = cifar100.load_data()
            n_labels = 100

        # Choose dataset size. This affects regularization needed
        r = np.random.rand()

        # not using full dataset to make regularization more important and
        # speed up testing a little bit
        data_size = int(2000 * (1 - r) + 40000 * r)

        # I do not use test data for validation, but last 10000 instances in dataset
        # so that trained models can be compared to results in literature
        (CX, CY), (CXt, CYt) = data

        if dataset == "mnist":
            CX = np.expand_dims(CX, axis=1)

        data = CX[:data_size], CY[:data_size], CX[-10000:], CY[-10000:]

        return data, n_labels

    def _reset(self):
        data, nb_classes = self.data_mix()
        X, Y, Xv, Yv = data

        # input square image dimensions
        img_rows, img_cols = X.shape[2], X.shape[1]
        img_channels = X.shape[-1]
        #
        X = X.reshape(X.shape[0], img_rows * img_cols * img_channels)
        Xv = Xv.reshape(Xv.shape[0], img_rows * img_cols * img_channels)

        # save number of classes and instances
        self.nb_classes = nb_classes
        self.nb_inst = len(X)

        # convert class vectors to binary class matrices
        # Y = Y.reshape(-1, 1)
        # Yv = Y.reshape(-1, 1)

        X = X.astype('float32')
        Xv = Xv.astype('float32')
        X /= 255
        Xv /= 255

        self.data = (X, Y, Xv, Yv)
        self.model = RandomForestClassifier(random_state=42)

        # initial accuracy values
        self.best_val = 0.0
        self.previous_acc = 0.0
        self.merror = 0.0
        self.tacc = 0.0
        self.epoch_idx = 0

        return self._get_obs()
