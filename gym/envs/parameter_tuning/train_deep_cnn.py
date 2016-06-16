from __future__ import print_function
import gym
import random
from gym import spaces
import numpy as np
from keras.datasets import cifar10, mnist, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import WeightRegularizer
from keras import backend as K

from itertools import cycle
import math


class CNNClassifierTraining(gym.Env):
    """Environment where agent learns to select training parameters and
    architecture of a deep convolutional neural network

    Training parameters that the agent can adjust are learning
    rate, learning rate decay, momentum, batch size, L1 / L2 regularization.

    Agent can select up to 5 cnn layers and up to 2 fc layers.

    Agent is provided with feedback on validation accuracy, as well as on
    the size of a dataset.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, natural=False):
        """
        Initialize environment
        """

        # I use array of len 1 to store constants (otherwise there were some errors)
        self.action_space = spaces.Tuple((
            spaces.Box(-5.0, 0.0, 1),  # learning rate
            spaces.Box(-7.0, -2.0, 1),  # decay
            spaces.Box(-5.0, 0.0, 1),  # momentum
            spaces.Box(2, 8, 1),  # batch size
            spaces.Box(-6.0, 1.0, 1),  # l1 reg
            spaces.Box(-6.0, 1.0, 1),  # l2 reg
            spaces.Box(0.0, 1.0, (5, 2)),  # convolutional layer parameters
            spaces.Box(0.0, 1.0, (2, 2)),  # fully connected layer parameters
        ))

        # observation features, in order: num of instances, num of labels,
        # validation accuracy after training with given parameters
        self.observation_space = spaces.Box(-1e5, 1e5, 2)  # validation accuracy

        # Start the first game
        self._reset()

    def _step(self, action):
        """
        Perform some action in the environment
        """
        assert self.action_space.contains(action)

        lr, decay, momentum, batch_size, l1, l2, convs, fcs = action

        # map ranges of inputs
        lr = (10.0 ** lr[0]).astype('float32')
        decay = (10.0 ** decay[0]).astype('float32')
        momentum = (10.0 ** momentum[0]).astype('float32')

        batch_size = int(2 ** batch_size[0])

        l1 = (10.0 ** l1[0]).astype('float32')
        l2 = (10.0 ** l2[0]).astype('float32')

        """
        names = ["lr", "decay", "mom", "batch", "l1", "l2"]
        values = [lr, decay, momentum, batch_size, l1, l2]

        for n,v in zip(names, values):
            print(n,v)
        """

        diverged, acc = self.train_blueprint(lr, decay, momentum, batch_size, l1, l2, convs, fcs)

        # save best validation. If diverged, acc is zero
        if acc > self.best_val:
            self.best_val = acc

        self.previous_acc = acc

        self.epoch_idx += 1
        done = self.epoch_idx == 10

        reward = self.best_val

        # as for number of labels increases, learning problem becomes
        # more difficult for fixed dataset size. In order to avoid
        # for the agent to ignore more complex datasets, on which
        # accuracy is low and concentrate on simple cases which bring bulk
        # of reward, reward is normalized by number of labels in dataset
        reward *= self.nb_classes

        # formula below encourages higher best validation
        reward += reward ** 2

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
        return np.array([self.nb_inst,
                         self.previous_acc])

    def data_mix(self):

        # randomly choose dataset
        dataset = random.choice(['mnist', 'cifar10', 'cifar100'])  #

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

        self.generate_data()

        # initial accuracy values
        self.best_val = 0.0
        self.previous_acc = 0.0
        self.epoch_idx = 0

        return self._get_obs()

    def generate_data(self):
        self.data, self.nb_classes = self.data_mix()
        # zero index corresponds to training inputs
        self.nb_inst = len(self.data[0])

    def train_blueprint(self, lr, decay, momentum, batch_size, l1, l2, convs, fcs):

        X, Y, Xv, Yv = self.data
        nb_classes = self.nb_classes

        reg = WeightRegularizer()

        # a hack to make regularization variable
        reg.l1 = K.variable(0.0)
        reg.l2 = K.variable(0.0)

        # input square image dimensions
        img_rows, img_cols = X.shape[-1], X.shape[-1]
        img_channels = X.shape[1]

        # convert class vectors to binary class matrices
        Y = np_utils.to_categorical(Y, nb_classes)
        Yv = np_utils.to_categorical(Yv, nb_classes)

        # here definition of the model happens
        model = Sequential()

        has_convs = False
        # create all convolutional layers
        for val, use in convs:

            # Size of convolutional layer
            cnvSz = int(val * 127) + 1

            if use < 0.5:
                continue
            has_convs = True
            model.add(Convolution2D(cnvSz, 3, 3, border_mode='same',
                                    input_shape=(img_channels, img_rows, img_cols),
                                    W_regularizer=reg,
                                    b_regularizer=reg))
            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))
            # model.add(Dropout(0.25))

        if has_convs:
            model.add(Flatten())
        else:
            model.add(Flatten(input_shape=(img_channels, img_rows, img_cols)))  # avoid excetpions on no convs

        # create all fully connected layers
        for val, use in fcs:

            if use < 0.5:
                continue

            # choose fully connected layer size
            densesz = int(1023 * val) + 1

            model.add(Dense(densesz,
                            W_regularizer=reg,
                            b_regularizer=reg))
            model.add(Activation('relu'))
            # model.add(Dropout(0.5))

        model.add(Dense(nb_classes,
                        W_regularizer=reg,
                        b_regularizer=reg))
        model.add(Activation('softmax'))

        # let's train the model using SGD + momentum (how original).
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        X = X.astype('float32')
        Xv = Xv.astype('float32')
        X /= 255
        Xv /= 255

        model = model
        sgd = sgd
        reg = reg

        # set parameters of training step

        sgd.lr.set_value(lr)
        sgd.decay.set_value(decay)
        sgd.momentum.set_value(momentum)

        reg.l1.set_value(l1)
        reg.l2.set_value(l2)

        # train model for one epoch_idx
        H = model.fit(X, Y,
                      batch_size=int(batch_size),
                      nb_epoch=10,
                      shuffle=True)

        diverged = math.isnan(H.history['loss'][-1])
        acc = 0.0

        if not diverged:
            _, acc = model.evaluate(Xv, Yv)

        return diverged, acc
