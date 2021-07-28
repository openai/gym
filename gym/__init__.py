import distutils.version
import os
import sys
import warnings

from gym import error, logger, vector, wrappers
from gym.core import (
    ActionWrapper,
    Env,
    GoalEnv,
    ObservationWrapper,
    RewardWrapper,
    Wrapper,
)
from gym.envs import make, register, spec
from gym.spaces import Space
from gym.version import VERSION as __version__

__all__ = ["Env", "Space", "Wrapper", "make", "spec", "register"]
