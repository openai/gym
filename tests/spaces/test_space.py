from functools import partial

import pytest

from gym import Space
from gym.spaces import utils

TESTING_SPACE = Space()


@pytest.mark.parametrize(
    "func",
    [
        TESTING_SPACE.sample,
        partial(TESTING_SPACE.contains, None),
        partial(utils.flatdim, TESTING_SPACE),
        partial(utils.flatten, TESTING_SPACE, None),
        partial(utils.flatten_space, TESTING_SPACE),
        partial(utils.unflatten, TESTING_SPACE, None),
    ],
)
def test_not_implemented_errors(func):
    with pytest.raises(NotImplementedError):
        func()
