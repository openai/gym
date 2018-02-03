import distutils.version
import os
import sys
import warnings

from gym import error
from gym.utils import reraise
from gym.version import VERSION as __version__

from gym.core import Env, Space, Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper
from gym.envs import make, spec
from gym import wrappers, logger

spaces = None
import gym.spaces
from gym import spaces

def undo_logger_setup():
    warnings.warn("gym.undo_logger_setup is deprecated. gym no longer modifies the global logging configuration")

__all__ = ["Env", "Space", "Wrapper", "make", "spec", "wrappers"]
