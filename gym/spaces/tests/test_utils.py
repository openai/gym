from collections import OrderedDict

import numpy as np
import pytest

from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple, utils


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

'''
Expecteded flattened types are based off:
1. The type that the space is hardcoded as(ie. multi_discrete=np.int64, discrete=np.int64, multi_binary=np.int8)
2. The type that the space is instantiated with(ie. box=np.float32 by default unless instantiated with a different type)
3. The smallest type that the composite space(tuple, dict) can be represented as. In flatten, this is determined 
   internally by numpy when np.concatenate is called. 
'''
@pytest.mark.parametrize(["original_space", "expected_flattened_dtype"], [
    (Discrete(3), np.int64),
    (Box(low=0., high=np.inf, shape=(2, 2)), np.float32),
    (Box(low=0., high=np.inf, shape=(2, 2), dtype=np.float16), np.float16),
    (Tuple([Discrete(5), Discrete(10)]), np.int64),
    (Tuple([Discrete(5), Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float32)]), np.float64),
    (Tuple((Discrete(5), Discrete(2), Discrete(2))), np.int64),
    (MultiDiscrete([2, 2, 100]), np.int64),
    (MultiBinary(10), np.int8),
    (Dict({"position": Discrete(5),
           "velocity": Box(low=np.array([0, 0]), high=np.array([1, 5]), dtype=np.float16)}), np.float64),
])
def test_dtypes(original_space, expected_flattened_dtype):
    flattened_space = utils.flatten_space(original_space)

    original_sample = original_space.sample()
    flattened_sample = utils.flatten(original_space, original_sample)
    unflattened_sample = utils.unflatten(original_space, flattened_sample)

    assert flattened_space.contains(flattened_sample), "Expected flattened_space to contain flattened_sample"
    assert flattened_space.dtype == expected_flattened_dtype, "Expected flattened_space's dtype to equal " \
                                                              "{}".format(expected_flattened_dtype)

    assert flattened_sample.dtype == flattened_space.dtype, "Expected flattened_space's dtype to equal " \
                                                            "flattened_sample's dtype "

    compare_sample_types(original_space, original_sample, unflattened_sample)


def compare_sample_types(original_space, original_sample, unflattened_sample):
    if isinstance(original_space, Discrete):
        assert isinstance(unflattened_sample, int), "Expected unflattened_sample to be an int. unflattened_sample: " \
                                                    "{} original_sample: {}".format(unflattened_sample, original_sample)
    elif isinstance(original_space, Tuple):
        for index in range(len(original_space)):
            compare_sample_types(original_space.spaces[index], original_sample[index], unflattened_sample[index])
    elif isinstance(original_space, Dict):
        for key, space in original_space.spaces.items():
            compare_sample_types(space, original_sample[key], unflattened_sample[key])
    else:
        assert unflattened_sample.dtype == original_sample.dtype, "Expected unflattened_sample's dtype to equal " \
                                                                  "original_sample's dtype. unflattened_sample: " \
                                                                  "{} original_sample: {}".format(unflattened_sample,
                                                                                                  original_sample)
