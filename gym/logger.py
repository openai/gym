"""Set of functions for logging messages."""
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


def set_level(level: int):
    """Set logging threshold on current logger."""
    global min_level
    min_level = level


def debug(msg: str, *args: object):
    """Logs a debug message to the user."""
    if min_level <= DEBUG:
        print(f"DEBUG: {msg % args}", file=sys.stderr)


def info(msg: str, *args: object):
    """Logs an info message to the user."""
    if min_level <= INFO:
        print(f"INFO: {msg % args}", file=sys.stderr)


def warn(
    msg: str,
    *args: object,
    category: Optional[Type[Warning]] = None,
    stacklevel: int = 1,
):
    """Raises a warning to the user if the min_level <= WARN.

    Args:
        msg: The message to warn the user
        *args: Additional information to warn the user
        category: The category of warning
        stacklevel: The stack level to raise to
    """
    if min_level <= WARN:
        warnings.warn(
            colorize(f"WARN: {msg % args}", "yellow"),
            category=category,
            stacklevel=stacklevel + 1,
        )


def deprecation(msg: str, *args: object):
    """Logs a deprecation warning to users."""
    warn(msg, *args, category=DeprecationWarning, stacklevel=2)


def error(msg: str, *args: object):
    """Logs an error message if min_level <= ERROR in red on the sys.stderr."""
    if min_level <= ERROR:
        print(colorize(f"ERROR: {msg % args}", "red"), file=sys.stderr)


# DEPRECATED:
setLevel = set_level
