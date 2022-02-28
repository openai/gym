from gym import error
from gym.version import VERSION as __version__

from gym.core import (
    Env,
    Wrapper,
    ObservationWrapper,
    ActionWrapper,
    RewardWrapper,
)
from gym.spaces import Space
from gym.envs import make, spec, register
from gym import logger
from gym import vector
from gym import wrappers
import os


__all__ = ["Env", "Space", "Wrapper", "make", "spec", "register"]

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

try:
    import gym_notices.notices as notices
    import sys

    # print version warning if necessary
    notice = notices.notices.get(__version__)
    if notice:
        print(notice, file=sys.stderr)

except Exception:  # nosec
    pass
