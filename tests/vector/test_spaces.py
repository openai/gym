import copy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gym.spaces import Box, Dict, MultiDiscrete, Space, Tuple
from gym.vector.utils.spaces import batch_space, iterate
from tests.vector.utils import CustomSpace, assert_rng_equal, custom_spaces, spaces

expected_batch_spaces_4 = [
    Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
    Box(low=0.0, high=10.0, shape=(4, 1), dtype=np.float64),
    Box(
        low=np.array(
            [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        ),
        high=np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        ),
        dtype=np.float64,
    ),
    Box(
        low=np.array(
            [
                [[-1.0, 0.0], [0.0, -1.0]],
                [[-1.0, 0.0], [0.0, -1.0]],
                [[-1.0, 0.0], [0.0, -1]],
                [[-1.0, 0.0], [0.0, -1.0]],
            ]
        ),
        high=np.ones((4, 2, 2)),
        dtype=np.float64,
    ),
    Box(low=0, high=255, shape=(4,), dtype=np.uint8),
    Box(low=0, high=255, shape=(4, 32, 32, 3), dtype=np.uint8),
    MultiDiscrete([2, 2, 2, 2]),
    Box(low=-2, high=2, shape=(4,), dtype=np.int64),
    Tuple((MultiDiscrete([3, 3, 3, 3]), MultiDiscrete([5, 5, 5, 5]))),
    Tuple(
        (
            MultiDiscrete([7, 7, 7, 7]),
            Box(
                low=np.array([[0.0, -1.0], [0.0, -1.0], [0.0, -1.0], [0.0, -1]]),
                high=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
                dtype=np.float64,
            ),
        )
    ),
    Box(
        low=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        high=np.array([[10, 12, 16], [10, 12, 16], [10, 12, 16], [10, 12, 16]]),
        dtype=np.int64,
    ),
    Box(low=0, high=1, shape=(4, 19), dtype=np.int8),
    Dict(
        {
            "position": MultiDiscrete([23, 23, 23, 23]),
            "velocity": Box(low=0.0, high=1.0, shape=(4, 1), dtype=np.float64),
        }
    ),
    Dict(
        {
            "position": Dict(
                {
                    "x": MultiDiscrete([29, 29, 29, 29]),
                    "y": MultiDiscrete([31, 31, 31, 31]),
                }
            ),
            "velocity": Tuple(
                (
                    MultiDiscrete([37, 37, 37, 37]),
                    Box(low=0, high=255, shape=(4,), dtype=np.uint8),
                )
            ),
        }
    ),
]

expected_custom_batch_spaces_4 = [
    Tuple((CustomSpace(), CustomSpace(), CustomSpace(), CustomSpace())),
    Tuple(
        (
            Tuple((CustomSpace(), CustomSpace(), CustomSpace(), CustomSpace())),
            Box(low=0, high=255, shape=(4,), dtype=np.uint8),
        )
    ),
]


@pytest.mark.parametrize(
    "space,expected_batch_space_4",
    list(zip(spaces, expected_batch_spaces_4)),
    ids=[space.__class__.__name__ for space in spaces],
)
def test_batch_space(space, expected_batch_space_4):
    batch_space_4 = batch_space(space, n=4)
    assert batch_space_4 == expected_batch_space_4


@pytest.mark.parametrize(
    "space,expected_batch_space_4",
    list(zip(custom_spaces, expected_custom_batch_spaces_4)),
    ids=[space.__class__.__name__ for space in custom_spaces],
)
def test_batch_space_custom_space(space, expected_batch_space_4):
    batch_space_4 = batch_space(space, n=4)
    assert batch_space_4 == expected_batch_space_4


@pytest.mark.parametrize(
    "space,batch_space",
    list(zip(spaces, expected_batch_spaces_4)),
    ids=[space.__class__.__name__ for space in spaces],
)
def test_iterate(space, batch_space):
    items = batch_space.sample()
    iterator = iterate(batch_space, items)
    i = 0
    for i, item in enumerate(iterator):
        assert item in space
    assert i == 3


@pytest.mark.parametrize(
    "space,batch_space",
    list(zip(custom_spaces, expected_custom_batch_spaces_4)),
    ids=[space.__class__.__name__ for space in custom_spaces],
)
def test_iterate_custom_space(space, batch_space):
    items = batch_space.sample()
    iterator = iterate(batch_space, items)
    i = 0
    for i, item in enumerate(iterator):
        assert item in space
    assert i == 3


@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
@pytest.mark.parametrize("n", [4, 5], ids=[f"n={n}" for n in [4, 5]])
@pytest.mark.parametrize(
    "base_seed", [123, 456], ids=[f"seed={base_seed}" for base_seed in [123, 456]]
)
def test_rng_different_at_each_index(space: Space, n: int, base_seed: int):
    """
    Tests that the rng values produced at each index are different
    to prevent if the rng is copied for each subspace
    """
    space.seed(base_seed)

    batched_space = batch_space(space, n)
    assert space.np_random is not batched_space.np_random
    assert_rng_equal(space.np_random, batched_space.np_random)

    batched_sample = batched_space.sample()
    sample = list(iterate(batched_space, batched_sample))
    assert not all(np.all(element == sample[0]) for element in sample), sample


@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
@pytest.mark.parametrize("n", [1, 2, 5], ids=[f"n={n}" for n in [1, 2, 5]])
@pytest.mark.parametrize(
    "base_seed", [123, 456], ids=[f"seed={base_seed}" for base_seed in [123, 456]]
)
def test_deterministic(space: Space, n: int, base_seed: int):
    """Tests the batched spaces are deterministic by using a copied version"""
    # Copy the spaces and check that the np_random are not reference equal
    space_a = space
    space_a.seed(base_seed)
    space_b = copy.deepcopy(space_a)
    assert_rng_equal(space_a.np_random, space_b.np_random)
    assert space_a.np_random is not space_b.np_random

    # Batch the spaces and check that the np_random are not reference equal
    space_a_batched = batch_space(space_a, n)
    space_b_batched = batch_space(space_b, n)
    assert_rng_equal(space_a_batched.np_random, space_b_batched.np_random)
    assert space_a_batched.np_random is not space_b_batched.np_random
    # Create that the batched space is not reference equal to the origin spaces
    assert space_a.np_random is not space_a_batched.np_random

    # Check that batched space a and b random number generator are not effected by the original space
    space_a.sample()
    space_a_batched_sample = space_a_batched.sample()
    space_b_batched_sample = space_b_batched.sample()
    for a_sample, b_sample in zip(
        iterate(space_a_batched, space_a_batched_sample),
        iterate(space_b_batched, space_b_batched_sample),
    ):
        if isinstance(a_sample, tuple):
            assert len(a_sample) == len(b_sample)
            for a_subsample, b_subsample in zip(a_sample, b_sample):
                assert_array_equal(a_subsample, b_subsample)
        else:
            assert_array_equal(a_sample, b_sample)
