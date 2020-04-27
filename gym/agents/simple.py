# -*- coding: utf-8 -*-
import numpy
from gym.agents.core import BaseAgent
from numpy.random import random


class RandomAgent(BaseAgent):
    """A random agent that takes random decisions."""

    def __init__(self, action_space):
        super().__init__(action_space)

    def act(self, observation, reward, done):
        self.reward += reward
        if done:
            return None
        self.actions += 1

        return self.action_space.sample()


class LazyEpsilonGreedy(BaseAgent):
    """ Easy epsilon greedy strategy.

    Pick the action with highest avg reward.
    """

    def __init__(self, action_space, epsilon=0.1):
        self.epsilon = epsilon
        self.last_action = None
        # for each action we collect total reward and total time such
        # action has been chosen
        self.action_history = {}
        super().__init__(action_space)

    def act(self, observation, reward, done):
        self.reward += reward
        if done:
            return None
        self.actions += 1

        # update what we know about past action
        if self.last_action is not None:
            if self.last_action not in self.action_history:
                self.action_history[self.last_action] = [0, 0]
            avg_reward = self.action_history[self.last_action][0]
            tot_action = self.action_history[self.last_action][1]
            avg_reward = (avg_reward * tot_action + reward) / (tot_action + 1)
            self.action_history[self.last_action][0] = avg_reward
            self.action_history[self.last_action][1] += 1

        action = None
        if random() < self.epsilon or self.last_action is None:
            action = self.action_space.sample()
        else:
            current_max = numpy.float('-inf')
            for a, (m, _) in self.action_history.items():
                if m > current_max:
                    action = a
                    current_max = m
        self.last_action = action
        return action


class TicTacToeStupidSolver(BaseAgent):
    """A simple tictactoe solver."""

    def __init__(self, action_space, whoami):
        self.whoami = whoami
        super().__init__(action_space)

    def act(self, observation, reward, done):
        self.reward += reward
        if done:
            return None
        self.actions += 1

        a = numpy.argwhere(observation == -1)
        valid_actions = a[:, 0] * observation.shape[0] + a[:, 1]
        a = numpy.argwhere(observation == self.whoami)
        past_actions = a[:, 0] * observation.shape[0] + a[:, 1]

        # this is very stupid, but for the sake of the example
        v = []
        for pa in past_actions:
            for delta in [-1, 1, 3, -3, 6, -6]:
                v.append(pa + delta)
        v = set(v) & set(list(valid_actions))
        if len(v) > 0:
            numpy.random.choice(list(v))
        return numpy.random.choice(valid_actions)
