from nose2 import tools
import numpy as np
from gym.spaces import Discrete, Tuple, Box
from gym.utils.space_utils import concatenated_input_dim, flatten_input, flatten_spaces
from gym.utils.space_utils import space_shapes, concatenated_input


@tools.params((Discrete(3), [(3,)]),
              (Tuple([Discrete(5), Discrete(10)]), [(5,), (10,)]))
def test_space_shapes(space, expected_shapes):
    actual_shapes = space_shapes(flatten_spaces(space))
    assert actual_shapes == expected_shapes


@tools.params((Discrete(5), 3, [np.array([0, 0, 0, 1, 0])]),
              (Tuple([Box(-10, 10, (2, 3)), Discrete(4)]), ([[2, 3, 4], [1, 2, 3]], 2),
               [[[2, 3, 4], [1, 2, 3]], [0, 0, 1, 0]]))
def test_flatten_input(space, observation, expected_input):
    actual_input = flatten_input(observation, space)
    for actual, expected in zip(actual_input, expected_input):
        np.testing.assert_array_equal(actual, expected)


@tools.params((Discrete(5), 3, 5),
              (Tuple([Box(-10, 10, (2, 3)), Discrete(4)]), ([[2, 3, 4], [1, 2, 3]], 2), 10))
def test_concatenated_input(space, observation, expected_shape):
    actual_shape_dim = concatenated_input_dim(space)
    assert (actual_shape_dim == expected_shape)
    flat_input = concatenated_input(observation, space)
    assert (flat_input.shape[0] == expected_shape)
