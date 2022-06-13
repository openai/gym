"""Root __init__ of the gym module setting the __all__ of gym modules."""
# isort: skip_file
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from gym import error  # noqa: E402
from gym.version import VERSION as __version__  # noqa: E402

from gym.core import (  # noqa: E402
    Env,
    Wrapper,
    ObservationWrapper,
    ActionWrapper,
    RewardWrapper,
)
from gym.spaces import Space  # noqa: E402
from gym.envs import make, spec, register  # noqa: E402
from gym import logger  # noqa: E402
from gym import vector  # noqa: E402
from gym import wrappers  # noqa: E402
import sys  # noqa: E402

__all__ = ["Env", "Space", "Wrapper", "make", "spec", "register"]

# Initializing pygame initializes audio connections through SDL. SDL uses alsa by default on all Linux systems
# SDL connecting to alsa frequently create these giant lists of warnings every time you import an environment using
#   pygame
# DSP is far more benign (and should probably be the default in SDL anyways)

if sys.platform.startswith("linux"):
    os.environ["SDL_AUDIODRIVER"] = "dsp"

try:
    import gym_notices.notices as notices

    # print version warning if necessary
    notice = notices.notices.get(__version__)
    if notice:
        print(notice, file=sys.stderr)

except Exception:  # nosec
    pass
