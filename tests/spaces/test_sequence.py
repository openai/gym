import re

import numpy as np
import pytest

import gym.spaces


def test_sample():
    """Tests the sequence sampling works as expects and the errors are correctly raised."""
    space = gym.spaces.Sequence(gym.spaces.Box(0, 1))

    # Test integer mask length
    for length in range(4):
        sample = space.sample(mask=(length, None))
        assert sample in space
        assert len(sample) == length

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Expects the length mask to be greater than or equal to zero, actual value: -1"
        ),
    ):
        space.sample(mask=(-1, None))

    # Test np.array mask length
    sample = space.sample(mask=(np.array([5]), None))
    assert sample in space
    assert len(sample) == 5

    sample = space.sample(mask=(np.array([3, 4, 5]), None))
    assert sample in space
    assert len(sample) in [3, 4, 5]

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Expects the shape of the length mask to be 1-dimensional, actual shape: (2, 2)"
        ),
    ):
        space.sample(mask=(np.array([[2, 2], [2, 2]]), None))

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Expects all values in the length_mask to be greater than or equal to zero, actual values: [ 1  2 -1]"
        ),
    ):
        space.sample(mask=(np.array([1, 2, -1]), None))

    # Test with an invalid length
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Expects the type of length_mask to an integer or a np.ndarray, actual type: <class 'str'>"
        ),
    ):
        space.sample(mask=("abc", None))
