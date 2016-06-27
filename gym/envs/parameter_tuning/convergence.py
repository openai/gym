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


class ConvergenceControl(gym.Env):
    """Environment where agent learns to tune parameters of training
    DURING the training of the neural network to improve its convergence /
    performance on the validation set.

    Parameters can be tuned after every epoch. Parameters tuned are learning
    rate, learning rate decay, momentum, batch size, L1 / L2 regularization.

    Agent is provided with feedback on validation accuracy, as well as on
    the size of dataset and number of classes, and some coarse description of
    architecture being optimized.

    The most close publication that I am aware of that tries to solve similar
    environment is

    http://research.microsoft.com/pubs/259048/daniel2016stepsizecontrol.pdf

    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, natural=False):
        """
        Initialize environment
        """

        # I use array of len 1 to store constants (otherwise there were some errors)
        self.action_space = spaces.Tuple((
                                          spaces.Box(-5.0,0.0, 1), # learning rate
                                          spaces.Box(-7.0,-2.0, 1), # decay
                                          spaces.Box(-5.0,0.0, 1), # momentum
                                          spaces.Box(2, 8, 1), # batch size
                                          spaces.Box(-6.0,1.0, 1), # l1 reg
                                          spaces.Box(-6.0,1.0, 1), # l2 reg
                                           ))

        # observation features, in order: num of instances, num of labels,
        # number of filter in part A / B of neural net, num of neurons in
        # output layer, validation accuracy after training with given
        # parameters
        self.observation_space = spaces.Box(-1e5,1e5, 6) # validation accuracy

        # Start the first game
        self._reset()

    def _step(self, action):
        """
        Perform some action in the environment
        """
        assert self.action_space.contains(action)

        lr, decay, momentum, batch_size, l1, l2 = action;


        # map ranges of inputs
        lr = (10.0 ** lr[0]).astype('float32')
        decay = (10.0 ** decay[0]).astype('float32')
        momentum = (10.0 ** momentum[0]).astype('float32')

        batch_size = int( 2 ** batch_size[0] )

        l1 = (10.0 ** l1[0]).astype('float32')
        l2 = (10.0 ** l2[0]).astype('float32')

        """
        names = ["lr", "decay", "mom", "batch", "l1", "l2"]
        values = [lr, decay, momentum, batch_size, l1, l2]

        for n,v in zip(names, values):
            print(n,v)
        """

        X,Y,Xv,Yv = self.data

        # set parameters of training step

        self.sgd.lr.set_value(lr)
        self.sgd.decay.set_value(decay)
        self.sgd.momentum.set_value(momentum)

        self.reg.l1.set_value(l1)
        self.reg.l2.set_value(l2)

        # train model for one epoch_idx
        H = self.model.fit(X, Y,
                      batch_size=int(batch_size),
                      nb_epoch=1,
                      shuffle=True)

        _, acc = self.model.evaluate(Xv,Yv)

        # save best validation
        if acc > self.best_val:
            self.best_val = acc

        self.previous_acc = acc;

        self.epoch_idx = self.epoch_idx + 1

        diverged = math.isnan( H.history['loss'][-1] )
        done = self.epoch_idx == 20 or diverged

        if diverged:
            """ maybe not set to a very large value; if you get something nice,
            but then diverge, maybe it is not too bad
            """
            reward = -100.0
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

        print(">> Step ",self.epoch_idx,"best validation:", self.best_val)

    def _get_obs(self):
        """
        Observe the environment. Is usually used after the step is taken
        """
        # observation as per observation space
        return np.array([self.nb_classes,
                         self.nb_inst,
                         self.convAsz,
                         self.convBsz,
                         self.densesz,
                         self.previous_acc])

    def data_mix(self):

        # randomly choose dataset
        dataset = random.choice(['mnist', 'cifar10', 'cifar100'])#

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
        data_size = int( 2000 * (1-r) + 40000 * r )

        # I do not use test data for validation, but last 10000 instances in dataset
        # so that trained models can be compared to results in literature
        (CX, CY), (CXt, CYt) = data

        if dataset == "mnist":
            CX = np.expand_dims(CX, axis=1)

        data = CX[:data_size], CY[:data_size], CX[-10000:], CY[-10000:];

        return data, n_labels

    def _reset(self):

        reg = WeightRegularizer()

        # a hack to make regularization variable
        reg.l1 = K.variable(0.0)
        reg.l2 = K.variable(0.0)


        data, nb_classes = self.data_mix()
        X, Y, Xv, Yv = data

        # input square image dimensions
        img_rows, img_cols = X.shape[-1], X.shape[-1]
        img_channels = X.shape[1]
        # save number of classes and instances
        self.nb_classes = nb_classes
        self.nb_inst = len(X)

        # convert class vectors to binary class matrices
        Y = np_utils.to_categorical(Y, nb_classes)
        Yv = np_utils.to_categorical(Yv, nb_classes)

        # here definition of the model happens
        model = Sequential()

        # double true for icnreased probability of conv layers
        if random.choice([True, True, False]):

            # Choose convolution #1
            self.convAsz = random.choice([32,64,128])

            model.add(Convolution2D(self.convAsz, 3, 3, border_mode='same',
                                    input_shape=(img_channels, img_rows, img_cols),
                                    W_regularizer = reg,
                                    b_regularizer = reg))
            model.add(Activation('relu'))

            model.add(Convolution2D(self.convAsz, 3, 3,
                                    W_regularizer = reg,
                                    b_regularizer = reg))
            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            # Choose convolution size B (if needed)
            self.convBsz = random.choice([0,32,64])

            if self.convBsz > 0:
                model.add(Convolution2D(self.convBsz, 3, 3, border_mode='same',
                                        W_regularizer = reg,
                                        b_regularizer = reg))
                model.add(Activation('relu'))

                model.add(Convolution2D(self.convBsz, 3, 3,
                                        W_regularizer = reg,
                                        b_regularizer = reg))
                model.add(Activation('relu'))

                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))

            model.add(Flatten())

        else:
            model.add(Flatten(input_shape=(img_channels, img_rows, img_cols)))
            self.convAsz = 0
            self.convBsz = 0

        # choose fully connected layer size
        self.densesz = random.choice([256,512,762])

        model.add(Dense(self.densesz,
                                W_regularizer = reg,
                                b_regularizer = reg))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(nb_classes,
                                W_regularizer = reg,
                                b_regularizer = reg))
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

        self.data = (X,Y,Xv,Yv)
        self.model = model
        self.sgd = sgd

        # initial accuracy values
        self.best_val = 0.0
        self.previous_acc = 0.0

        self.reg = reg
        self.epoch_idx = 0

        return self._get_obs()
