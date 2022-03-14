import sys
import warnings
from typing import Optional, Type

from gym.utils import colorize

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

min_level = 30


warnings.simplefilter("once", DeprecationWarning)


def set_level(level: int) -> None:
    """
    Set logging threshold on current logger.
    """
    global min_level
    min_level = level


def debug(msg: str, *args: object):
    if min_level <= DEBUG:
        print(f"DEBUG: {msg % args}", file=sys.stderr)


def info(msg: str, *args: object):
    if min_level <= INFO:
        print(f"INFO: {msg % args}", file=sys.stderr)


def warn(
    msg: str,
    *args: object,
    category: Optional[Type[Warning]] = None,
    stacklevel: int = 1,
):
    if min_level <= WARN:
        warnings.warn(
            colorize(f"WARN: {msg % args}", "yellow"),
            category=category,
            stacklevel=stacklevel + 1,
        )


def deprecation(msg: str, *args: object):
    warn(msg, *args, category=DeprecationWarning, stacklevel=2)


def error(msg: str, *args: object):
    if min_level <= ERROR:
        print(colorize(f"ERROR: {msg % args}", "red"), file=sys.stderr)


# DEPRECATED:
setLevel = set_level
