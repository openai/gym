import numpy as np
import pytest

from gym.spaces import Box

# Todo, move Box unique tests from test_spaces.py to test_box.py


@pytest.mark.parametrize(
    "box,expected_shape",
    [
        (Box(low=np.zeros(2), high=np.ones(2)), (2,)),
        (Box(low=np.zeros((2, 1)), high=np.ones((2, 1))), (2, 1)),
        (Box(low=0, high=1, shape=(5, 2)), (5, 2)),
        (Box(low=0, high=1), (1,)),
    ],
)
def test_box_shape_inference(box, expected_shape):
    assert box.shape == expected_shape
    assert box.sample().shape == expected_shape
