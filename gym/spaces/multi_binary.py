from __future__ import annotations

from typing import Optional, Union, Sequence
import numpy as np
from .space import Space


class MultiBinary(Space[np.ndarray]):
    """
    An n-shape binary space.

    The argument to MultiBinary defines n, which could be a number or a `list` of numbers.

    Example Usage:

    >> self.observation_space = spaces.MultiBinary(5)

    >> self.observation_space.sample()

        array([0, 1, 0, 1, 0], dtype=int8)

    >> self.observation_space = spaces.MultiBinary([3, 2])

    >> self.observation_space.sample()

        array([[0, 0],
               [0, 1],
               [1, 1]], dtype=int8)

    """

    def __init__(
        self, n: Union[np.ndarray, Sequence[int], int], seed: Optional[int] = None
    ):
        if isinstance(n, (Sequence, np.ndarray)):
            self.n = input_n = tuple(int(i) for i in n)
        else:
            self.n = n = int(n)
            input_n = (n,)

        assert (np.asarray(input_n) > 0).all(), "n (counts) have to be positive"

        super().__init__(input_n, np.int8, seed)

    @property
    def shape(self) -> tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self._shape  # type: ignore

    def sample(self) -> np.ndarray:
        return self.np_random.integers(low=0, high=2, size=self.n, dtype=self.dtype)

    def contains(self, x) -> bool:
        if isinstance(x, Sequence):
            x = np.array(x)  # Promote list to array for contains check
        if self.shape != x.shape:
            return False
        return ((x == 0) | (x == 1)).all()

    def to_jsonable(self, sample_n) -> list:
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n) -> list:
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self) -> str:
        return f"MultiBinary({self.n})"

    def __eq__(self, other) -> bool:
        return isinstance(other, MultiBinary) and self.n == other.n
