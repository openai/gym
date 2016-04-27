import hashlib
import numpy as np
import logging
import os
import random
import struct
import sys

import gym

logger = logging.getLogger(__name__)

root_logger = logging.getLogger()
requests_logger = logging.getLogger('requests')

# Set up the default handler
formatter = logging.Formatter('[%(asctime)s] %(message)s')
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)

# We need to take in the gym logger explicitly since this is called
# at initialization time.
def logger_setup(gym_logger):
    root_logger.addHandler(handler)
    gym_logger.setLevel(logging.INFO)
    # When set to INFO, this will print out the hostname of every
    # connection it makes.
    # requests_logger.setLevel(logging.WARN)

def undo_logger_setup():
    """Undoes the automatic logging setup done by OpenAI Gym. You should call
    this function if you want to manually configure logging
    yourself. Typical usage would involve putting something like the
    following at the top of your script:

    gym.undo_logger_setup()
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stderr))
    """
    root_logger.removeHandler(handler)
    gym.logger.setLevel(logging.NOTSET)
    requests_logger.setLevel(logging.NOTSET)

def seed(a=None):
    """Seeds the 'random' and 'numpy.random' generators. By default,
    Python seeds these with the system time. Call this if you are
    using multiple processes.

    Notes:
        SECURITY SENSITIVE: a bug here would allow people to generate fake results. Please let us know if you find one :).

    Args:
        a (Optional[int, str]): None or no argument seeds from an operating system specific randomness source. If an int or str passed, then all of bits are used.
    """
    # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
    if a is None:
        a = bigint_from_bytes(os.urandom(32))

    if isinstance(a, str):
        a = a.encode('utf8')
        a += hashlib.sha512(a).digest()
        a = bigint_from_bytes(a)

    # Actually seed the generators
    random.seed(a)
    np.random.seed(int_list_from_bigint(a))

    return a

# TODO: don't hardcode sizeof_int here
def bigint_from_bytes(bytes):
    sizeof_int = 4
    padding = sizeof_int - len(bytes) % sizeof_int
    bytes += '\0' * padding
    int_count = len(bytes) / sizeof_int
    unpacked = struct.unpack("{}I".format(int_count), bytes)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum

def int_list_from_bigint(bigint):
    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints
