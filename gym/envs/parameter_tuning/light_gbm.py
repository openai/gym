from __future__ import print_function
import gym
import random
from gym import spaces
import numpy as np
from keras.datasets import cifar10, mnist, cifar100
from keras.utils import np_utils
from itertools import cycle
import math
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


def multi_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


def multi_logloss(y_true, y_pred):
    return np.mean([-math.log(y_pred[i][y]) for i, y in enumerate(y_true)])


class LightGBM(gym.Env):

    metadata = {"render.modes": ["human"]}

    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'metric': {'l2', 'auc'},
    #     'num_leaves': 31,
    #     'learning_rate': 0.05,
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'verbose': 0
    #     }

    def __init__(self, natural=False):
        self.action_space = spaces.Tuple((
            spaces.Box(10, 500, 1),  # n_est
            spaces.Box(-5.0, 0.0, 1),  # lr
            spaces.Box(10, 50, 1)  # num_leaves
        ))

        # observation features, in order: num of instances, num of labels,
        # number of filter in part A / B of neural net, num of neurons in
        # output layer, validation accuracy after training with given
        # parameters
        self.observation_space = spaces.Box(-1e5, 1e5, 5)

        # Start the first game
        self._reset()

    def _step(self, action):
        """
        Perform some action in the environment
        """
        # assert self.action_space.contains(action)

        n_est, lr, num_leaves = action

        # map ranges of inputs
        lr = (10.0 ** lr[0]).astype('float32')
        n_est = int(n_est[0])
        num_leaves = int(num_leaves[0])

        """
        names = ["lr", "n_est", "num_leaves"]
        values = [lr, n_est, num_leaves]

        for n,v in zip(names, values):
            print(n,v)
        """

        X_train, y_train, X_valid, y_valid = self.data

        gbm = LGBMClassifier(n_estimators=n_est,
                             num_leaves=num_leaves, learning_rate=lr, silent=True)

        # train model for one epoch_idx
        print(X_train.shape, X_valid.shape)
        gbm.fit(X_train, y_train, eval_set=[
                (X_valid, y_valid)], early_stopping_rounds=5, verbose=True)

        fpred = gbm.predict_proba(X_valid)
        pred = gbm.predict(X_valid)

        self.merror = multi_error(y_valid, pred)

        self.mloss = multi_logloss(y_valid, fpred)

        acc = accuracy_score(y_valid, pred)

        self.tacc = accuracy_score(y_train, gbm.predict(X_train))
        # self.assertAlmostEqual(ret, gbm.evals_result_[
        #                        'valid_0']['multi_logloss'][gbm.best_iteration_ - 1], places=5)
        # _, acc = self.model.evaluate(Xv, Yv)

        # save best validation
        if acc > self.best_val:
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
            reward = self.best_val

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
                         self.mloss,
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
        self.model = LGBMClassifier()

        # initial accuracy values
        self.best_val = 0.0
        self.previous_acc = 0.0
        self.merror = 0.0
        self.mloss = 0.0
        self.tacc = 0.0
        self.epoch_idx = 0

        return self._get_obs()
