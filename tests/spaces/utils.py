from typing import List

import numpy as np

from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space, Text

TESTING_FUNDAMENTAL_SPACES = [
    Discrete(3),
    Discrete(3, start=-1),
    Box(low=0.0, high=1.0),
    Box(low=0.0, high=np.inf, shape=(2, 2)),
    Box(low=np.array([-10.0, 0.0]), high=np.array([10.0, 10.0]), dtype=np.float64),
    Box(low=-np.inf, high=0.0, shape=(2, 1)),
    Box(low=0.0, high=np.inf, shape=(2, 1)),
    MultiDiscrete([2, 2]),
    MultiDiscrete([[2, 3], [3, 2]]),
    MultiBinary(8),
    MultiBinary([2, 3]),
    Text(6),
    Text(min_length=3, max_length=6),
    Text(6, charset="abcdef"),
]
TESTING_FUNDAMENTAL_SPACES_IDS = [f"{space}" for space in TESTING_FUNDAMENTAL_SPACES]


TESTING_SPACES: List[Space] = TESTING_FUNDAMENTAL_SPACES  # + TESTING_COMPOSITE_SPACES
TESTING_SPACES_IDS = TESTING_FUNDAMENTAL_SPACES_IDS  # + TESTING_COMPOSITE_SPACES_IDS
