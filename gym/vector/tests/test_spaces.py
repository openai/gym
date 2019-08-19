import pytest
import numpy as np

from gym.spaces import Box, MultiDiscrete, Tuple, Dict
from gym.vector.tests.utils import spaces

from gym.vector.utils.spaces import _BaseGymSpaces, batch_space

expected_batch_spaces_4 = [
    Box(low=-1., high=1., shape=(4,), dtype=np.float64),
    Box(low=0., high=10., shape=(4, 1), dtype=np.float32),
    Box(low=np.array([[-1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.]]),
        high=np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]), dtype=np.float32),
    Box(low=np.array([[[-1., 0.], [0., -1.]], [[-1., 0.], [0., -1.]], [[-1., 0.], [0., -1]],
        [[-1., 0.], [0., -1.]]]), high=np.ones((4, 2, 2)), dtype=np.float32),
    Box(low=0, high=255, shape=(4,), dtype=np.uint8),
    Box(low=0, high=255, shape=(4, 32, 32, 3), dtype=np.uint8),
    MultiDiscrete([2, 2, 2, 2]),
    Tuple((MultiDiscrete([3, 3, 3, 3]), MultiDiscrete([5, 5, 5, 5]))),
    Tuple((MultiDiscrete([7, 7, 7, 7]), Box(low=np.array([[0., -1.], [0., -1.], [0., -1.], [0., -1]]),
        high=np.array([[1., 1.], [1., 1.], [1., 1.], [1., 1.]]), dtype=np.float32))),
    Box(low=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        high=np.array([[10, 12, 16], [10, 12, 16], [10, 12, 16], [10, 12, 16]]), dtype=np.int64),
    Box(low=0, high=1, shape=(4, 19), dtype=np.int8),
    Dict({
        'position': MultiDiscrete([23, 23, 23, 23]),
        'velocity': Box(low=0., high=1., shape=(4, 1), dtype=np.float32)
    }),
    Dict({
        'position': Dict({'x': MultiDiscrete([29, 29, 29, 29]), 'y': MultiDiscrete([31, 31, 31, 31])}),
        'velocity': Tuple((MultiDiscrete([37, 37, 37, 37]), Box(low=0, high=255, shape=(4,), dtype=np.uint8)))
    })
]

@pytest.mark.parametrize('space,expected_batch_space_4', list(zip(spaces,
    expected_batch_spaces_4)), ids=[space.__class__.__name__ for space in spaces])
def test_batch_space(space, expected_batch_space_4):
    batch_space_4 = batch_space(space, n=4)
    assert batch_space_4 == expected_batch_space_4
