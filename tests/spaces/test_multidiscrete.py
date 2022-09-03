import pytest

from gym.spaces import Discrete, MultiDiscrete
from gym.utils.env_checker import data_equivalence


def test_multidiscrete_as_tuple():
    # 1D multi-discrete
    space = MultiDiscrete([3, 4, 5])

    assert space.shape == (3,)
    assert space[0] == Discrete(3)
    assert space[0:1] == MultiDiscrete([3])
    assert space[0:2] == MultiDiscrete([3, 4])
    assert space[:] == space and space[:] is not space

    # 2D multi-discrete
    space = MultiDiscrete([[3, 4, 5], [6, 7, 8]])

    assert space.shape == (2, 3)
    assert space[0, 1] == Discrete(4)
    assert space[0] == MultiDiscrete([3, 4, 5])
    assert space[0:1] == MultiDiscrete([[3, 4, 5]])
    assert space[0:2, :] == MultiDiscrete([[3, 4, 5], [6, 7, 8]])
    assert space[:, 0:1] == MultiDiscrete([[3], [6]])
    assert space[0:2, 0:2] == MultiDiscrete([[3, 4], [6, 7]])
    assert space[:] == space and space[:] is not space
    assert space[:, :] == space and space[:, :] is not space


def test_multidiscrete_subspace_reproducibility():
    # 1D multi-discrete
    space = MultiDiscrete([100, 200, 300])
    space.seed()

    assert data_equivalence(space[0].sample(), space[0].sample())
    assert data_equivalence(space[0:1].sample(), space[0:1].sample())
    assert data_equivalence(space[0:2].sample(), space[0:2].sample())
    assert data_equivalence(space[:].sample(), space[:].sample())
    assert data_equivalence(space[:].sample(), space.sample())

    # 2D multi-discrete
    space = MultiDiscrete([[300, 400, 500], [600, 700, 800]])
    space.seed()

    assert data_equivalence(space[0, 1].sample(), space[0, 1].sample())
    assert data_equivalence(space[0].sample(), space[0].sample())
    assert data_equivalence(space[0:1].sample(), space[0:1].sample())
    assert data_equivalence(space[0:2, :].sample(), space[0:2, :].sample())
    assert data_equivalence(space[:, 0:1].sample(), space[:, 0:1].sample())
    assert data_equivalence(space[0:2, 0:2].sample(), space[0:2, 0:2].sample())
    assert data_equivalence(space[:].sample(), space[:].sample())
    assert data_equivalence(space[:, :].sample(), space[:, :].sample())
    assert data_equivalence(space[:, :].sample(), space.sample())


def test_multidiscrete_length():
    space = MultiDiscrete(nvec=[3, 2, 4])
    assert len(space) == 3

    space = MultiDiscrete(nvec=[[2, 3], [3, 2]])
    with pytest.warns(
        UserWarning,
        match="Getting the length of a multi-dimensional MultiDiscrete space.",
    ):
        assert len(space) == 2
