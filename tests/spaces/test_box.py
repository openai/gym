import re

import numpy as np
import pytest

from gym.spaces import Box

# Todo, move Box unique tests from test_spaces.py to test_box.py


@pytest.mark.parametrize(
    "box,expected_shape",
    [
        (
            Box(low=np.zeros(2), high=np.ones(2)),
            (2,),
        ),  # Test with same 1-dim low and high shape
        (
            Box(low=np.zeros((2, 1)), high=np.ones((2, 1))),
            (2, 1),
        ),  # Test with same multi-dim low and high shape
        (
            Box(low=0, high=1, shape=(5, 2)),
            (5, 2),
        ),  # Test with scalar low high and different shape
        (Box(low=0, high=1), (1,)),  # Test with int and int
        (Box(low=0.0, high=1.0), (1,)),  # Test with float and float
        (Box(low=np.zeros(1)[0], high=np.ones(1)[0]), (1,)),
        (Box(low=0.0, high=1), (1,)),  # Test with float and int
        (Box(low=0, high=np.int32(1)), (1,)),  # Test with python int and numpy int32
        (Box(low=0, high=np.ones(3)), (3,)),  # Test with array and scalar
        (Box(low=np.zeros(3), high=1.0), (3,)),  # Test with array and scalar
    ],
)
def test_box_shape_inference(box, expected_shape):
    assert box.shape == expected_shape
    assert box.sample().shape == expected_shape


@pytest.mark.parametrize(
    "value,valid",
    [
        (1, True),
        (1.0, True),
        (np.int32(1), True),
        (np.float32(1.0), True),
        (np.zeros(2, dtype=np.float32), True),
        (np.zeros((2, 2), dtype=np.float32), True),
        (np.inf, True),
        (np.nan, True),  # This is a weird side
        (True, False),
        (np.bool8(True), False),
        (1 + 1j, False),
        (np.complex128(1 + 1j), False),
        ("string", False),
    ],
)
def test_box_values(value, valid):
    if valid:
        with pytest.warns(None) as warnings:
            Box(low=value, high=value)
        assert len(warnings.list) == 0, tuple(warning.message for warning in warnings)
    else:
        with pytest.raises(
            ValueError,
            match=r"expect their types to be np\.ndarray, an integer or a float",
        ):
            Box(low=value, high=value)


@pytest.mark.parametrize(
    "low,high,kwargs,error,message",
    [
        (
            0,
            1,
            {"dtype": None},
            AssertionError,
            "Box dtype must be explicitly provided, cannot be None.",
        ),
        (
            0,
            1,
            {"shape": (None,)},
            AssertionError,
            "Expect all shape elements to be an integer, actual type: (<class 'NoneType'>,)",
        ),
        (
            0,
            1,
            {
                "shape": (
                    1,
                    None,
                )
            },
            AssertionError,
            "Expect all shape elements to be an integer, actual type: (<class 'int'>, <class 'NoneType'>)",
        ),
        (
            0,
            1,
            {
                "shape": (
                    np.int64(1),
                    None,
                )
            },
            AssertionError,
            "Expect all shape elements to be an integer, actual type: (<class 'numpy.int64'>, <class 'NoneType'>)",
        ),
        (
            None,
            None,
            {},
            ValueError,
            "Box shape is inferred from low and high, expect their types to be np.ndarray, an integer or a float, actual type low: <class 'NoneType'>, high: <class 'NoneType'>",
        ),
        (
            0,
            None,
            {},
            ValueError,
            "Box shape is inferred from low and high, expect their types to be np.ndarray, an integer or a float, actual type low: <class 'int'>, high: <class 'NoneType'>",
        ),
        (
            np.zeros(3),
            np.ones(2),
            {},
            AssertionError,
            "high.shape doesn't match provided shape, high.shape: (2,), shape: (3,)",
        ),
    ],
)
def test_box_errors(low, high, kwargs, error, message):
    with pytest.raises(error, match=f"^{re.escape(message)}$"):
        Box(low=low, high=high, **kwargs)
