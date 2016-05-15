import distutils.version
import logging
import sys

from gym import error
from gym.configuration import logger_setup, undo_logger_setup
from gym.utils import reraise

logger = logging.getLogger(__name__)

# Do this before importing any other gym modules, as most of them import some
# dependencies themselves.
def sanity_check_dependencies():
    import numpy
    import requests
    import six

    if distutils.version.StrictVersion(numpy.__version__) < distutils.version.StrictVersion('1.10.4'):
        logger.warn("You have 'numpy' version %s installed, but 'gym' requires at least 1.10.4. HINT: upgrade via 'pip install -U numpy'.", numpy.__version__)

    if distutils.version.StrictVersion(requests.__version__) < distutils.version.StrictVersion('2.0'):
        logger.warn("You have 'requests' version %s installed, but 'gym' requires at least 2.0. HINT: upgrade via 'pip install -U requests'.", requests.__version__)

# We automatically configure a logger with a simple stderr handler. If
# you'd rather customize logging yourself, run undo_logger_setup.
#
# (Note: this needs to happen before importing the rest of gym, since
# we may print a warning at load time.)
logger_setup(logger)
del logger_setup

sanity_check_dependencies()

from gym.core import Env, Space
from gym.envs import make, spec
from gym.scoreboard.api import upload

__all__ = ["Env", "Space", "make", "spec", "upload"]
