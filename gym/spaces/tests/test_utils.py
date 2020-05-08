from collections import OrderedDict
import numpy as np
import pytest

from gym.spaces import utils
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary, Dict


@pytest.mark.parametrize(["space", "flatdim"], [
    (Discrete(3), 3),
    (Box(low=0., high=np.inf, shape=(2, 2)), 4),
    (Tuple([Discrete(5), Discrete(10)]), 15),
    (Tuple([Discrete(5), Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32)]), 7),
    (Tuple((Discrete(5), Discrete(2), Discrete(2))), 9),
    (MultiDiscrete([2, 2, 100]), 3),
    (MultiBinary(10), 10),
    (Dict({"position": Discrete(5),
           "velocity": Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32)}), 7),
])
def test_flatdim(space, flatdim):
    dim = utils.flatdim(space)
    assert dim == flatdim, "Expected {} to equal {}".format(dim, flatdim)


@pytest.mark.parametrize("space", [
    Discrete(3),
    Box(low=0., high=np.inf, shape=(2, 2)),
    Tuple([Discrete(5), Discrete(10)]),
    Tuple([Discrete(5), Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32)]),
    Tuple((Discrete(5), Discrete(2), Discrete(2))),
    MultiDiscrete([2, 2, 100]),
    MultiBinary(10),
    Dict({"position": Discrete(5),
          "velocity": Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32)}),
    ])
def test_flatten_space_boxes(space):
    flat_space = utils.flatten_space(space)
    assert isinstance(flat_space, Box), "Expected {} to equal {}".format(type(flat_space), Box)
    flatdim = utils.flatdim(space)
    (single_dim, ) = flat_space.shape
    assert single_dim == flatdim, "Expected {} to equal {}".format(single_dim, flatdim)


@pytest.mark.parametrize("space", [
    Discrete(3),
    Box(low=0., high=np.inf, shape=(2, 2)),
    Tuple([Discrete(5), Discrete(10)]),
    Tuple([Discrete(5), Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32)]),
    Tuple((Discrete(5), Discrete(2), Discrete(2))),
    MultiDiscrete([2, 2, 100]),
    MultiBinary(10),
    Dict({"position": Discrete(5),
          "velocity": Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32)}),
    ])
def test_flat_space_contains_flat_points(space):
    some_samples = [space.sample() for _ in range(10)]
    flattened_samples = [utils.flatten(space, sample) for sample in some_samples]
    flat_space = utils.flatten_space(space)
    for i, flat_sample in enumerate(flattened_samples):
        assert flat_sample in flat_space,\
            'Expected sample #{} {} to be in {}'.format(i, flat_sample, flat_space)


@pytest.mark.parametrize("space", [
    Discrete(3),
    Box(low=0., high=np.inf, shape=(2, 2)),
    Tuple([Discrete(5), Discrete(10)]),
    Tuple([Discrete(5), Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32)]),
    Tuple((Discrete(5), Discrete(2), Discrete(2))),
    MultiDiscrete([2, 2, 100]),
    MultiBinary(10),
    Dict({"position": Discrete(5),
          "velocity": Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32)}),
    ])
def test_flatten_dim(space):
    sample = utils.flatten(space, space.sample())
    (single_dim, ) = sample.shape
    flatdim = utils.flatdim(space)
    assert single_dim == flatdim, "Expected {} to equal {}".format(single_dim, flatdim)


@pytest.mark.parametrize("space", [
    Discrete(3),
    Box(low=0., high=np.inf, shape=(2, 2)),
    Tuple([Discrete(5), Discrete(10)]),
    Tuple([Discrete(5), Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32)]),
    Tuple((Discrete(5), Discrete(2), Discrete(2))),
    MultiDiscrete([2, 2, 100]),
    MultiBinary(10),
    Dict({"position": Discrete(5),
          "velocity": Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32)}),
])
def test_flatten_roundtripping(space):
    some_samples = [space.sample() for _ in range(10)]
    flattened_samples = [utils.flatten(space, sample) for sample in some_samples]
    roundtripped_samples = [utils.unflatten(space, sample) for sample in flattened_samples]
    for i, (original, roundtripped) in enumerate(zip(some_samples, roundtripped_samples)):
        assert compare_nested(original, roundtripped), \
            'Expected sample #{} {} to equal {}'.format(i, original, roundtripped)


def compare_nested(left, right):
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.allclose(left, right)
    elif isinstance(left, OrderedDict) and isinstance(right, OrderedDict):
        res = len(left) == len(right)
        for ((left_key, left_value), (right_key, right_value)) in zip(left.items(), right.items()):
            if not res:
                return False
            res = left_key == right_key and compare_nested(left_value, right_value)
        return res
    elif isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        res = len(left) == len(right)
        for (x, y) in zip(left, right):
            if not res:
                return False
            res = compare_nested(x, y)
        return res
    else:
        return left == right
