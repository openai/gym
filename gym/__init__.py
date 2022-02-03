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
import os

__all__ = ["Env", "Space", "Wrapper", "make", "spec", "register"]

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

try:  # Motivation for snippet explained here: https://github.com/Farama-Foundation/gym-notices
    with urllib.request.urlopen(  # nosec
        "https://raw.githubusercontent.com/ \
            Farama-Foundation/gym-notices/main/notices.txt",
        timeout=4,
    ) as f:
        html = f.read().decode("utf-8")
        lines = html.split("\n")
        for line in lines:
            if line.startswith(__version__):
                print(line.split(" : ")[1])
except Exception:  # nosec
    pass
