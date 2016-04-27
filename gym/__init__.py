import logging
import sys

from gym.core import Env, Space
from gym.configuration import logger_setup, undo_logger_setup
from gym.envs import make, spec
from gym.scoreboard.api import upload

logger = logging.getLogger(__name__)

# We automatically configure a logger with a simple stderr handler. If
# you'd rather customize logging yourself, run undo_logger_setup.
logger_setup(logger)
del logger_setup

__all__ = ["Env", "Space", "make", "spec", "upload"]
