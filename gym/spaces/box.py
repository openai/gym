from __future__ import annotations

from typing import Tuple, SupportsFloat, Union, Type, Optional, Sequence

import numpy as np

from .space import Space
from gym import logger


def _short_repr(arr: np.ndarray) -> str:
    """Create a shortened string representation of a numpy array.

    If arr is a multiple of the all-ones vector, return a string representation of the multiplier.
    Otherwise, return a string representation of the entire array.
    """
    if arr.size != 0 and np.min(arr) == np.max(arr):
        return str(np.min(arr))
    return str(arr)


class Box(Space[np.ndarray]):
    """
    A (possibly unbounded) box in R^n. Specifically, a Box represents the
    Cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo).

    There are two common use cases:

    * Identical bound for each dimension::
        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(3, 4)

    * Independent bound for each dimension::
        >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        Box(2,)

    """

    def __init__(
        self,
        low: Union[SupportsFloat, np.ndarray],
        high: Union[SupportsFloat, np.ndarray],
        shape: Optional[Sequence[int]] = None,
        dtype: Type = np.float32,
        seed: Optional[int] = None,
    ):
        assert dtype is not None, "dtype must be explicitly provided. "
        self.dtype = np.dtype(dtype)

        # determine shape if it isn't provided directly
        if shape is not None:
            shape = tuple(shape)
        elif not np.isscalar(low):
            shape = low.shape  # type: ignore
        elif not np.isscalar(high):
            shape = high.shape  # type: ignore
        else:
            raise ValueError(
                "shape must be provided or inferred from the shapes of low or high"
            )
        assert isinstance(shape, tuple)

        # Capture the boundedness information before replacing np.inf with get_inf
        _low = np.full(shape, low, dtype=float) if np.isscalar(low) else low
        self.bounded_below = -np.inf < _low
        _high = np.full(shape, high, dtype=float) if np.isscalar(high) else high
        self.bounded_above = np.inf > _high

        low = _broadcast(low, dtype, shape, inf_sign="-")  # type: ignore
        high = _broadcast(high, dtype, shape, inf_sign="+")  # type: ignore

        assert isinstance(low, np.ndarray)
        assert low.shape == shape, "low.shape doesn't match provided shape"
        assert isinstance(high, np.ndarray)
        assert high.shape == shape, "high.shape doesn't match provided shape"

        self._shape: Tuple[int, ...] = shape

        low_precision = get_precision(low.dtype)
        high_precision = get_precision(high.dtype)
        dtype_precision = get_precision(self.dtype)
        if min(low_precision, high_precision) > dtype_precision:  # type: ignore
            logger.warn(f"Box bound precision lowered by casting to {self.dtype}")
        self.low = low.astype(self.dtype)
        self.high = high.astype(self.dtype)

        self.low_repr = _short_repr(self.low)
        self.high_repr = _short_repr(self.high)

        super().__init__(self.shape, self.dtype, seed)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self._shape

    def is_bounded(self, manner: str = "both") -> bool:
        below = bool(np.all(self.bounded_below))
        above = bool(np.all(self.bounded_above))
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def sample(self) -> np.ndarray:
        """
        Generates a single random sample inside of the Box.

        In creating a sample of the box, each coordinate is sampled according to
        the form of the interval:

        * [a, b] : uniform distribution
        * [a, oo) : shifted exponential distribution
        * (-oo, b] : shifted negative exponential distribution
        * (-oo, oo) : normal distribution
        """
        high = self.high if self.dtype.kind == "f" else self.high.astype("int64") + 1
        sample = np.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = self.np_random.normal(size=unbounded[unbounded].shape)

        sample[low_bounded] = (
            self.np_random.exponential(size=low_bounded[low_bounded].shape)
            + self.low[low_bounded]
        )

        sample[upp_bounded] = (
            -self.np_random.exponential(size=upp_bounded[upp_bounded].shape)
            + self.high[upp_bounded]
        )

        sample[bounded] = self.np_random.uniform(
            low=self.low[bounded], high=high[bounded], size=bounded[bounded].shape
        )
        if self.dtype.kind == "i":
            sample = np.floor(sample)

        return sample.astype(self.dtype)

    def contains(self, x) -> bool:
        if not isinstance(x, np.ndarray):
            logger.warn("Casting input x to numpy array.")
            x = np.asarray(x, dtype=self.dtype)

        return bool(
            np.can_cast(x.dtype, self.dtype)
            and x.shape == self.shape
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n: Sequence[SupportsFloat]) -> list[np.ndarray]:
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self) -> str:
        return f"Box({self.low_repr}, {self.high_repr}, {self.shape}, {self.dtype})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Box)
            and (self.shape == other.shape)
            and np.allclose(self.low, other.low)
            and np.allclose(self.high, other.high)
        )


def get_inf(dtype, sign: str) -> SupportsFloat:
    """Returns an infinite that doesn't break things.
    `dtype` must be an `np.dtype`
    `bound` must be either `min` or `max`
    """
    if np.dtype(dtype).kind == "f":
        if sign == "+":
            return np.inf
        elif sign == "-":
            return -np.inf
        else:
            raise TypeError(f"Unknown sign {sign}, use either '+' or '-'")
    elif np.dtype(dtype).kind == "i":
        if sign == "+":
            return np.iinfo(dtype).max - 2
        elif sign == "-":
            return np.iinfo(dtype).min + 2
        else:
            raise TypeError(f"Unknown sign {sign}, use either '+' or '-'")
    else:
        raise ValueError(f"Unknown dtype {dtype} for infinite bounds")


def get_precision(dtype) -> SupportsFloat:
    if np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).precision
    else:
        return np.inf


def _broadcast(
    value: Union[SupportsFloat, np.ndarray],
    dtype,
    shape: tuple[int, ...],
    inf_sign: str,
) -> np.ndarray:
    """handle infinite bounds and broadcast at the same time if needed"""
    if np.isscalar(value):
        value = get_inf(dtype, inf_sign) if np.isinf(value) else value  # type: ignore
        value = np.full(shape, value, dtype=dtype)
    else:
        assert isinstance(value, np.ndarray)
        if np.any(np.isinf(value)):
            # create new array with dtype, but maintain old one to preserve np.inf
            temp = value.astype(dtype)
            temp[np.isinf(value)] = get_inf(dtype, inf_sign)
            value = temp
    return value
