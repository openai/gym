import re
import warnings

import numpy as np
import pytest

import gym.error
from gym.spaces import Box
from gym.spaces.box import get_inf


@pytest.mark.parametrize(
    "box,expected_shape",
    [
        (  # Test with same 1-dim low and high shape
            Box(low=np.zeros(2), high=np.ones(2), dtype=np.int32),
            (2,),
        ),
        (  # Test with same multi-dim low and high shape
            Box(low=np.zeros((2, 1)), high=np.ones((2, 1)), dtype=np.int32),
            (2, 1),
        ),
        (  # Test with scalar low high and different shape
            Box(low=0, high=1, shape=(5, 2)),
            (5, 2),
        ),
        (Box(low=0, high=1), (1,)),  # Test with int and int
        (Box(low=0.0, high=1.0), (1,)),  # Test with float and float
        (Box(low=np.zeros(1)[0], high=np.ones(1)[0]), (1,)),
        (Box(low=0.0, high=1), (1,)),  # Test with float and int
        (Box(low=0, high=np.int32(1)), (1,)),  # Test with python int and numpy int32
        (Box(low=0, high=np.ones(3)), (3,)),  # Test with array and scalar
        (Box(low=np.zeros(3), high=1.0), (3,)),  # Test with array and scalar
    ],
)
def test_shape_inference(box, expected_shape):
    """Test that the shape inference is as expected."""
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
        (np.nan, True),  # This is a weird case that we allow
        (True, False),
        (np.bool8(True), False),
        (1 + 1j, False),
        (np.complex128(1 + 1j), False),
        ("string", False),
    ],
)
def test_low_high_values(value, valid: bool):
    """Test what `low` and `high` values are valid for `Box` space."""
    if valid:
        with warnings.catch_warnings(record=True) as caught_warnings:
            Box(low=value, high=value)
        assert len(caught_warnings) == 0, tuple(
            warning.message for warning in caught_warnings
        )
    else:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "expect their types to be np.ndarray, an integer or a float"
            ),
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
def test_init_errors(low, high, kwargs, error, message):
    """Test all constructor errors."""
    with pytest.raises(error, match=f"^{re.escape(message)}$"):
        Box(low=low, high=high, **kwargs)


def test_dtype_check():
    """Tests the Box contains function with different dtypes."""
    # Related Issues:
    # https://github.com/openai/gym/issues/2357
    # https://github.com/openai/gym/issues/2298

    space = Box(0, 1, (), dtype=np.float32)

    # casting will match the correct type
    assert np.array(0.5, dtype=np.float32) in space

    # float16 is in float32 space
    assert np.array(0.5, dtype=np.float16) in space

    # float64 is not in float32 space
    assert np.array(0.5, dtype=np.float64) not in space


@pytest.mark.parametrize(
    "space",
    [
        Box(low=0, high=np.inf, shape=(2,), dtype=np.int32),
        Box(low=0, high=np.inf, shape=(2,), dtype=np.float32),
        Box(low=0, high=np.inf, shape=(2,), dtype=np.int64),
        Box(low=0, high=np.inf, shape=(2,), dtype=np.float64),
        Box(low=-np.inf, high=0, shape=(2,), dtype=np.int32),
        Box(low=-np.inf, high=0, shape=(2,), dtype=np.float32),
        Box(low=-np.inf, high=0, shape=(2,), dtype=np.int64),
        Box(low=-np.inf, high=0, shape=(2,), dtype=np.float64),
        Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.int32),
        Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.int64),
        Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
        Box(low=0, high=np.inf, shape=(2, 3), dtype=np.int32),
        Box(low=0, high=np.inf, shape=(2, 3), dtype=np.float32),
        Box(low=0, high=np.inf, shape=(2, 3), dtype=np.int64),
        Box(low=0, high=np.inf, shape=(2, 3), dtype=np.float64),
        Box(low=-np.inf, high=0, shape=(2, 3), dtype=np.int32),
        Box(low=-np.inf, high=0, shape=(2, 3), dtype=np.float32),
        Box(low=-np.inf, high=0, shape=(2, 3), dtype=np.int64),
        Box(low=-np.inf, high=0, shape=(2, 3), dtype=np.float64),
        Box(low=-np.inf, high=np.inf, shape=(2, 3), dtype=np.int32),
        Box(low=-np.inf, high=np.inf, shape=(2, 3), dtype=np.float32),
        Box(low=-np.inf, high=np.inf, shape=(2, 3), dtype=np.int64),
        Box(low=-np.inf, high=np.inf, shape=(2, 3), dtype=np.float64),
        Box(low=np.array([-np.inf, 0]), high=np.array([0.0, np.inf]), dtype=np.int32),
        Box(low=np.array([-np.inf, 0]), high=np.array([0.0, np.inf]), dtype=np.float32),
        Box(low=np.array([-np.inf, 0]), high=np.array([0.0, np.inf]), dtype=np.int64),
        Box(low=np.array([-np.inf, 0]), high=np.array([0.0, np.inf]), dtype=np.float64),
    ],
)
def test_infinite_space(space):
    """
    To test spaces that are passed in have only 0 or infinite bounds because `space.high` and `space.low`
     are both modified within the init, we check for infinite when we know it's not 0
    """

    assert np.all(
        space.low < space.high
    ), f"Box low bound ({space.low}) is not lower than the high bound ({space.high})"

    space.seed(0)
    sample = space.sample()

    # check if space contains sample
    assert (
        sample in space
    ), f"Sample ({sample}) not inside space according to `space.contains()`"

    # manually check that the sign of the sample is within the bounds
    assert np.all(
        np.sign(sample) <= np.sign(space.high)
    ), f"Sign of sample ({sample}) is less than space upper bound ({space.high})"
    assert np.all(
        np.sign(space.low) <= np.sign(sample)
    ), f"Sign of sample ({sample}) is more than space lower bound ({space.low})"

    # check that int bounds are bounded for everything
    # but floats are unbounded for infinite
    if np.any(space.high != 0):
        assert (
            space.is_bounded("above") is False
        ), "inf upper bound supposed to be unbounded"
    else:
        assert (
            space.is_bounded("above") is True
        ), "non-inf upper bound supposed to be bounded"

    if np.any(space.low != 0):
        assert (
            space.is_bounded("below") is False
        ), "inf lower bound supposed to be unbounded"
    else:
        assert (
            space.is_bounded("below") is True
        ), "non-inf lower bound supposed to be bounded"

    if np.any(space.low != 0) or np.any(space.high != 0):
        assert space.is_bounded("both") is False
    else:
        assert space.is_bounded("both") is True

    # check for dtype
    assert (
        space.high.dtype == space.dtype
    ), f"High's dtype {space.high.dtype} doesn't match `space.dtype`'"
    assert (
        space.low.dtype == space.dtype
    ), f"Low's dtype {space.high.dtype} doesn't match `space.dtype`'"

    with pytest.raises(
        ValueError, match="manner is not in {'below', 'above', 'both'}, actual value:"
    ):
        space.is_bounded("test")


def test_legacy_state_pickling():
    legacy_state = {
        "dtype": np.dtype("float32"),
        "_shape": (5,),
        "low": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "high": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "bounded_below": np.array([True, True, True, True, True]),
        "bounded_above": np.array([True, True, True, True, True]),
        "_np_random": None,
    }

    b = Box(-1, 1, ())
    assert "low_repr" in b.__dict__ and "high_repr" in b.__dict__
    del b.__dict__["low_repr"]
    del b.__dict__["high_repr"]
    assert "low_repr" not in b.__dict__ and "high_repr" not in b.__dict__

    b.__setstate__(legacy_state)
    assert b.low_repr == "0.0"
    assert b.high_repr == "1.0"


def test_get_inf():
    """Tests that get inf function works as expected, primarily for coverage."""
    assert get_inf(np.float32, "+") == np.inf
    assert get_inf(np.float16, "-") == -np.inf
    with pytest.raises(
        TypeError, match=re.escape("Unknown sign *, use either '+' or '-'")
    ):
        get_inf(np.float32, "*")

    assert get_inf(np.int16, "+") == 32765
    assert get_inf(np.int8, "-") == -126
    with pytest.raises(
        TypeError, match=re.escape("Unknown sign *, use either '+' or '-'")
    ):
        get_inf(np.int32, "*")

    with pytest.raises(
        ValueError,
        match=re.escape("Unknown dtype <class 'numpy.complex128'> for infinite bounds"),
    ):
        get_inf(np.complex_, "+")


def test_sample_mask():
    """Box cannot have a mask applied."""
    space = Box(0, 1)
    with pytest.raises(
        gym.error.Error,
        match=re.escape("Box.sample cannot be provided a mask, actual value: "),
    ):
        space.sample(mask=np.array([0, 1, 0], dtype=np.int8))
