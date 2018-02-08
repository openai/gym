import pytest
import numpy as np
import copy
import random
from gym.spaces import Box, Dict, Tuple


def test_compatibility():
    # create all testable spaces
    spaces = [
        Box(-1.0, 1.0, (20, ), np.float32),
        Box(-1.0, 1.0, (40, ), np.float32),
        Box(-1.0, 1.0, (20, 20), np.float32),
        Box(-1.0, 1.0, (20, 24), np.float32),
        Dict({'A': Box(-1.0, 1.0, (20, ), np.float32), 'B': Box(-1.0, 1.0, (20, ), np.float32)}),
        Dict({'A': Box(-1.0, 1.0, (20, ), np.float32), 'B': Box(-1.0, 1.0, (40, ), np.float32)}),
        Dict({'A': Box(-1.0, 1.0, (40, ), np.float32), 'B': Box(-1.0, 1.0, (20, ), np.float32)}),
        Tuple([Box(-1.0, 1.0, (20, ), np.float32), Box(-1.0, 1.0, (20, ), np.float32)]),
        Tuple([Box(-1.0, 1.0, (40, ), np.float32), Box(-1.0, 1.0, (20, ), np.float32)]),
        Tuple([Box(-1.0, 1.0, (20, ), np.float32), Box(-1.0, 1.0, (40, ), np.float32)]),
    ]

    # iterate and compare all combinations
    spaces_range = range(len(spaces))
    for x in spaces_range:
        space1 = spaces[x]
        for y in spaces_range:
            # create copy of Space with random bounds
            space2 = randomize_space_bounds(spaces[y])

            expected_compatible = x == y
            actual_compatible = space1.compatible(space2)
            assert expected_compatible == actual_compatible

def randomize_space_bounds(space):
    # copy space
    space = copy.copy(space)

    # check if space contain subspaces
    if hasattr(space, "spaces"):
        # compare each sub-Space
        subspaces = space.spaces
        if hasattr(subspaces, "keys"):
            iterable = subspaces.keys()
        else:
            iterable = range(len(subspaces))
        for k in iterable:
            space.spaces[k] = randomize_space_bounds(space.spaces[k])

    # randomize bounds
    if hasattr(space, "low"):
        space.low = random.uniform(-5.0, 5.0)
    if hasattr(space, "high"):
        space.high = random.uniform(-5.0, 5.0)

    return space
