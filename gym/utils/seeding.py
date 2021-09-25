import hashlib
import numpy as np
import os
import random as _random
import struct
import sys

from gym import error


def np_random(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise error.Error(
            "Seed must be a non-negative integer or omitted, not {}".format(seed)
        )

    rng = np.random.default_rng(seed)
    return rng, seed
