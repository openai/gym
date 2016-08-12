import numpy as np
import gym
from gym import spaces

# Constants
NUM_ACTIONS = 43
ALLOWED_ACTIONS = [
    [0, 10, 11],                                # 0 - Basic
    [0, 10, 11, 13, 14, 15],                    # 1 - Corridor
    [0, 14, 15],                                # 2 - DefendCenter
    [0, 14, 15],                                # 3 - DefendLine
    [13, 14, 15],                               # 4 - HealthGathering
    [13, 14, 15],                               # 5 - MyWayHome
    [0, 14, 15],                                # 6 - PredictPosition
    [10, 11],                                   # 7 - TakeCover
    [x for x in range(NUM_ACTIONS) if x != 33], # 8 - Deathmatch
]

__all__ = [ 'DiscreteMinimal', 'Discrete7', 'Discrete17', 'DiscreteFull',
            'BoxMinimal', 'Box7', 'Box17', 'BoxFull']

# Helper functions

class DiscreteToHighLow(object):
    """ Acts as a filter between HighLow and Discrete (only one button can be pressed at a time) """
    def __init__(self, high_low_space, allowed_actions=None):
        assert isinstance(high_low_space, spaces.HighLow)
        self.actions = allowed_actions if allowed_actions is not None else list(range(high_low_space.shape))
        # +1 to take into account noop at beginning
        self.mapping = {(i + 1): self.actions[i] for i in range(len(self.actions))}
        self.n = high_low_space.shape
        self.discrete_n = len(self.actions) + 1

    def __call__(self, act):
        action_list = [0] * self.n
        if act in self.mapping:
            action_list[self.mapping[act]] = 1
        return action_list

class BoxToHighLow(object):
    """ Acts as a filter between HighLow and Box (values are value based on HighLow precision level) """
    def __init__(self, high_low_space, allowed_actions=None):
        assert isinstance(high_low_space, spaces.HighLow)
        self.actions = allowed_actions if allowed_actions is not None else list(range(high_low_space.shape))
        self.low = [high_low_space.matrix[i, 0] for i in self.actions]
        self.high = [high_low_space.matrix[i, 1] for i in self.actions]
        self.precision = [high_low_space.matrix[i, 2] for i in range(high_low_space.shape)]
        self.box = spaces.Box(np.array(self.low), np.array(self.high))
        self.n = high_low_space.shape

    def __call__(self, box_array):
        action_list = [0] * self.n
        for ind, i in enumerate(self.actions):
            action_list[i] = round(box_array[ind], self.precision[i])
        return action_list

# Wrappers

class DiscreteMinimal(gym.Wrapper):
    """
        Converts HighLow action space to Discrete with NOOP and only the level's allowed actions
        Discrete only supports one action at a given time

        Basic:              NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT
        Corridor:           NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
        DefendCenter        NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
        DefendLine:         NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
        HealthGathering:    NOOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
        MyWayHome:          NOOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
        PredictPosition:    NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
        TakeCover:          NOOP, MOVE_RIGHT, MOVE_LEFT
        Deathmatch:         NOOP, ALL COMMANDS (Deltas are limited to [0,1] range and will not work properly)
    """
    def __init__(self, env):
        allowed_actions = ALLOWED_ACTIONS[self._unwrapped.level]
        self.action_filter = DiscreteToHighLow(self.action_space, allowed_actions)
        self.action_space = spaces.Discrete(self.action_filter.discrete_n)
    def _step(self, action):
        return self.env._step(self.action_filter(action))

class Discrete7(gym.Wrapper):
    """
        Converts HighLow action space to Discrete with the 8 minimum actions required to complete all levels
        Actions are NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, SELECT_NEXT_WEAPON
        Discrete only supports one action at a given time
    """
    def __init__(self, env):
        allowed_actions = [0, 10, 11, 13, 14, 15, 31]
        self.action_filter = DiscreteToHighLow(self.action_space, allowed_actions)
        self.action_space = spaces.Discrete(self.action_filter.discrete_n)
    def _step(self, action):
        return self.env._step(self.action_filter(action))

class Discrete17(gym.Wrapper):
    """
        Converts HighLow action space to Discrete with the 18 most used actions
        Actions are NOOP, ATTACK, JUMP, CROUCH, TURN180, RELOAD, SPEED, STRAFE, MOVE_RIGHT, MOVE_LEFT, MOVE_BACKWARD
                    MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, LOOK_UP, LOOK_DOWN, SELECT_NEXT_WEAPON, SELECT_PREV_WEAPON
        Discrete only supports one action at a given time
    """
    def __init__(self, env):
        allowed_actions = [0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32]
        self.action_filter = DiscreteToHighLow(self.action_space, allowed_actions)
        self.action_space = spaces.Discrete(self.action_filter.discrete_n)
    def _step(self, action):
        return self.env._step(self.action_filter(action))

class DiscreteFull(gym.Wrapper):
    """
        Converts HighLow action space to Discrete with the all available actions
        Discrete only supports one action at a given time
    """
    def __init__(self, env):
        self.action_filter = DiscreteToHighLow(self.action_space)
        self.action_space = spaces.Discrete(self.action_filter.discrete_n)
    def _step(self, action):
        return self.env._step(self.action_filter(action))

class BoxMinimal(gym.Wrapper):
    """
        Converts HighLow action space to Box with only the level's allowed actions
        Values are automatically rounded to the nearest integer
        Box supports multiple simultaneous actions

        Basic:              ATTACK, MOVE_RIGHT, MOVE_LEFT
        Corridor:           ATTACK, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
        DefendCenter        ATTACK, TURN_RIGHT, TURN_LEFT
        DefendLine:         ATTACK, TURN_RIGHT, TURN_LEFT
        HealthGathering:    MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
        MyWayHome:          MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
        PredictPosition:    ATTACK, TURN_RIGHT, TURN_LEFT
        TakeCover:          MOVE_RIGHT, MOVE_LEFT
        Deathmatch:         ALL COMMANDS
    """
    def __init__(self, env):
        allowed_actions = ALLOWED_ACTIONS[self._unwrapped.level]
        self.action_filter = BoxToHighLow(self.action_space, allowed_actions)
        self.action_space = self.action_filter.box
    def _step(self, action):
        return self.env._step(self.action_filter(action))

class Box7(gym.Wrapper):
    """
        Converts HighLow action space to Box with the 8 minimum actions required to complete all levels
        Values are automatically rounded to the nearest integer
        Actions are NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, SELECT_NEXT_WEAPON
        Box supports multiple simultaneous actions
    """
    def __init__(self, env):
        allowed_actions = [0, 10, 11, 13, 14, 15, 31]
        self.action_filter = BoxToHighLow(self.action_space, allowed_actions)
        self.action_space = self.action_filter.box
    def _step(self, action):
        return self.env._step(self.action_filter(action))

class Box17(gym.Wrapper):
    """
        Converts HighLow action space to Box with the 18 most used actions
        Values are automatically rounded to the nearest integer
        Actions are NOOP, ATTACK, JUMP, CROUCH, TURN180, RELOAD, SPEED, STRAFE, MOVE_RIGHT, MOVE_LEFT, MOVE_BACKWARD
                    MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, LOOK_UP, LOOK_DOWN, SELECT_NEXT_WEAPON, SELECT_PREV_WEAPON
        Box supports multiple simultaneous actions
    """
    def __init__(self, env):
        allowed_actions = [0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32]
        self.action_filter = BoxToHighLow(self.action_space, allowed_actions)
        self.action_space = self.action_filter.box
    def _step(self, action):
        return self.env._step(self.action_filter(action))

class BoxFull(gym.Wrapper):
    """
        Converts HighLow action space to Box with the all available actions
        Values are automatically rounded to the nearest integer
        Box supports multiple simultaneous actions
    """
    def __init__(self, env):
        self.action_filter = BoxToHighLow(self.action_space)
        self.action_space = self.action_filter.box
    def _step(self, action):
        return self.env._step(self.action_filter(action))
