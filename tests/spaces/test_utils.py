from itertools import zip_longest

import numpy as np
import pytest

from gym.spaces import Box, Graph, GraphInstance, utils
from gym.utils.env_checker import data_equivalence
from tests.spaces.utils import TESTING_SPACES, TESTING_SPACES_IDS

TESTING_SPACES_EXPECTED_FLATDIMS = [
    3,
    3,
    1,
    4,
    2,
    2,
    2,
    4,
    10,
    8,
    6,
    10,
    10,
    10,
    15,
    7,
    10,
    6,
    7,
    17,
    None,
    None,
    None,
]


@pytest.mark.parametrize(
    ["space", "flatdim"],
    zip_longest(TESTING_SPACES, TESTING_SPACES_EXPECTED_FLATDIMS),
    ids=TESTING_SPACES_IDS,
)
def test_flatdim(space, flatdim):
    """Checks that the flatten dims of the space is equal to an expected value."""
    if not isinstance(space, Graph):
        dim = utils.flatdim(space)
        assert dim == flatdim, f"Expected {dim} to equal {flatdim}"


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_flatten_space(space):
    """Test that the flattened spaces are a box and have the `flatdim` shape."""
    flat_space = utils.flatten_space(space)

    if isinstance(flat_space, Box):
        (single_dim,) = flat_space.shape
        flatdim = utils.flatdim(space)

        assert single_dim == flatdim
    elif isinstance(flat_space, Graph):
        assert isinstance(space, Graph)

        (node_single_dim,) = flat_space.node_space.shape
        node_flatdim = utils.flatdim(space.node_space)
        assert node_single_dim == node_flatdim

        if space.edge_space is not None:
            (edge_single_dim,) = flat_space.edge_space.shape
            edge_flatdim = utils.flatdim(space.edge_space)
            assert edge_single_dim == edge_flatdim
    else:
        raise Exception(f"Unknown flattened space: {type(flat_space)}")


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_flatten(space):
    """Test that a flattened sample have the `flatdim` shape."""
    flattened_sample = utils.flatten(space, space.sample())

    if isinstance(flattened_sample, np.ndarray):
        (single_dim,) = flattened_sample.shape
        flatdim = utils.flatdim(space)

        assert single_dim == flatdim
    elif isinstance(flattened_sample, GraphInstance):
        assert isinstance(space, Graph)

        node_flatdim = utils.flatdim(space.node_space)
        assert flattened_sample.nodes.shape[1] == node_flatdim

        if space.edge_space is not None:
            edge_flatdim = utils.flatdim(space.edge_space)
            assert flattened_sample.edges.shape[1] == edge_flatdim
    else:
        raise Exception(f"Unknown sample type: {type(flattened_sample)}")


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
