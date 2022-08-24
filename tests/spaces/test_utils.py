from itertools import zip_longest
from typing import Optional

import numpy as np
import pytest

import gym
from gym.spaces import Box, Graph, utils
from gym.utils.env_checker import data_equivalence
from tests.spaces.utils import TESTING_SPACES, TESTING_SPACES_IDS

TESTING_SPACES_EXPECTED_FLATDIMS = [
    # Discrete
    3,
    3,
    # Box
    1,
    4,
    2,
    2,
    2,
    # Multi-discrete
    4,
    10,
    # Multi-binary
    8,
    6,
    # Text
    6,
    6,
    6,
    # Tuple
    9,
    7,
    10,
    6,
    None,
    # Dict
    7,
    8,
    17,
    None,
    # Graph
    None,
    None,
    None,
    # Sequence
    None,
    None,
    None,
]


@pytest.mark.parametrize(
    ["space", "flatdim"],
    zip_longest(TESTING_SPACES, TESTING_SPACES_EXPECTED_FLATDIMS),
    ids=TESTING_SPACES_IDS,
)
def test_flatdim(space: gym.spaces.Space, flatdim: Optional[int]):
    """Checks that the flattened dims of the space is equal to an expected value."""
    if space.is_np_flattenable:
        dim = utils.flatdim(space)
        assert dim == flatdim, f"Expected {dim} to equal {flatdim}"
    else:
        with pytest.raises(
            ValueError,
        ):
            utils.flatdim(space)


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_flatten_space(space):
    """Test that the flattened spaces are a box and have the `flatdim` shape."""
    flat_space = utils.flatten_space(space)

    if space.is_np_flattenable:
        assert isinstance(flat_space, Box)
        (single_dim,) = flat_space.shape
        flatdim = utils.flatdim(space)

        assert single_dim == flatdim
    elif isinstance(flat_space, Graph):
        assert isinstance(space, Graph)

        (node_single_dim,) = flat_space.node_space.shape
        node_flatdim = utils.flatdim(space.node_space)
        assert node_single_dim == node_flatdim

        if flat_space.edge_space is not None:
            (edge_single_dim,) = flat_space.edge_space.shape
            edge_flatdim = utils.flatdim(space.edge_space)
            assert edge_single_dim == edge_flatdim
    else:
        assert isinstance(
            space, (gym.spaces.Tuple, gym.spaces.Dict, gym.spaces.Sequence)
        )


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_flatten(space):
    """Test that a flattened sample have the `flatdim` shape."""
    flattened_sample = utils.flatten(space, space.sample())

    if space.is_np_flattenable:
        assert isinstance(flattened_sample, np.ndarray)
        (single_dim,) = flattened_sample.shape
        flatdim = utils.flatdim(space)

        assert single_dim == flatdim
    else:
        assert isinstance(flattened_sample, (tuple, dict, Graph))


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_flat_space_contains_flat_points(space):
    """Test that the flattened samples are contained within the flattened space."""
    flattened_samples = [utils.flatten(space, space.sample()) for _ in range(10)]
    flat_space = utils.flatten_space(space)

    for flat_sample in flattened_samples:
        assert flat_sample in flat_space


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_flatten_roundtripping(space):
    """Tests roundtripping with flattening and unflattening are equal to the original sample."""
    samples = [space.sample() for _ in range(10)]

    flattened_samples = [utils.flatten(space, sample) for sample in samples]
    unflattened_samples = [
        utils.unflatten(space, sample) for sample in flattened_samples
    ]

    for original, roundtripped in zip(samples, unflattened_samples):
        assert data_equivalence(original, roundtripped)
