from gym import error
import urllib.request
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

__all__ = ["Env", "Space", "Wrapper", "make", "spec", "register"]


try:
    with urllib.request.urlopen("https://raw.githubusercontent.com/ \
            Farama-Foundation/gym-notices/main/notices.txt") as f:
        html = f.read().decode("utf-8")
        lines = html.split("\n")
        for line in lines:
            if line.startswith(__version__):
                print(line.split(' : ')[1])
except Exception: 
    pass
