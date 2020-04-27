# -*- coding: utf-8 -*-


class BaseAgent(object):
    """It describes an agent reference class. Other agents should inherit from
    this one, mainly implementing `act`.

    The main API methods that users of this class need to know are:

        act
        reset

    Take a look to `RandomAgent`. More complex examples may need to override
    also `reset`.
    """

    def __init__(self, action_space, name=None):
        self.action_space = action_space
        if name is None:
            self.name = self.__class__.__name__
        self.reward = 0
        self.actions = 0

    def act(self, observation, reward, done):
        # self.reward += reward
        # self.actions += 1
        raise NotImplementedError()

    def reset(self):
        # reset may be more complex than this
        self.reward = 0
        self.actions = 0

    def __repr__(self):
        return '{}: {} actions taken, reward: {}'.format(self.name, self.actions, self.reward)
