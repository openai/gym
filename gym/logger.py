import sys
import warnings

from gym.utils import colorize

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

MIN_LEVEL = 30


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    global MIN_LEVEL
    MIN_LEVEL = level


def debug(msg, *args):
    if MIN_LEVEL <= DEBUG:
        print("%s: %s" % ("DEBUG", msg % args), file=sys.stderr)


def info(msg, *args):
    if MIN_LEVEL <= INFO:
        print("%s: %s" % ("INFO", msg % args), file=sys.stderr)


def warn(msg, *args, category=None, stacklevel=1):
    if MIN_LEVEL <= WARN:
        warnings.warn(
            colorize("%s: %s" % ("WARN", msg % args), "yellow"),
            category=category,
            stacklevel=stacklevel + 1,
        )


def deprecation(msg, *args):
    warn(msg, *args, category=DeprecationWarning, stacklevel=2)


def error(msg, *args):
    if MIN_LEVEL <= ERROR:
        print(colorize("%s: %s" % ("ERROR", msg % args), "red"), file=sys.stderr)


# DEPRECATED:
setLevel = set_level
