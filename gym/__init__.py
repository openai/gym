import distutils.version
import os
import sys
import warnings

from gym import error
from gym.utils import reraise
from gym.version import VERSION as __version__

from gym.core import Env, GoalEnv, Space, Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper
from gym.envs import make, spec
from gym import logger

def undo_logger_setup():
    warnings.warn("gym.undo_logger_setup is deprecated. gym no longer modifies the global logging configuration")

# Upon one acccess to gym.spaces.foo (or a manually-called import
# gym.spaces), gym.spaces will be imported and override the stub
# object.
class Spaces(object):
    def __getattr__(self, k):
        warnings.warn('DEPRECATION WARNING: to improve load times, gym no longer automatically loads gym.spaces. Please run "import gym.spaces" to load gym.spaces on your own. This warning will turn into an error in a future version of gym.')
        import gym.spaces
        return getattr(gym.spaces, k)
spaces = Spaces()

class Wrappers(object):
    def __getattr__(self, k):
        warnings.warn('DEPRECATION WARNING: to improve load times, gym no longer automatically loads gym.wrappers. Please run "import gym.wrappers" to load gym.wrappers on your own. This warning will turn into an error in a future version of gym.')
        import gym.wrappers
        return getattr(gym.wrappers, k)
wrappers = Wrappers()

__all__ = ["Env", "Space", "Wrapper", "make", "spec", "wrappers", "spaces"]
