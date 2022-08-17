"""Implementation of a space that consists of binary np.ndarrays of a fixed shape."""
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from gym.spaces.space import Space
from gym.utils import seeding


class MultiBinary(Space[np.ndarray]):
    """An n-shape binary space.

    Elements of this space are binary arrays of a shape that is fixed during construction.

    Example Usage::

        >>> observation_space = MultiBinary(5)
        >>> observation_space.sample()
            array([0, 1, 0, 1, 0], dtype=int8)
        >>> observation_space = MultiBinary([3, 2])
        >>> observation_space.sample()
            array([[0, 0],
                [0, 1],
                [1, 1]], dtype=int8)
    """

    def __init__(
        self,
        n: Union[np.ndarray, Sequence[int], int],
        seed: Optional[Union[int, seeding.RandomNumberGenerator]] = None,
    ):
        """Constructor of :class:`MultiBinary` space.

        Args:
            n: This will fix the shape of elements of the space. It can either be an integer (if the space is flat)
                or some sort of sequence (tuple, list or np.ndarray) if there are multiple axes.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space.
        """
        if isinstance(n, (Sequence, np.ndarray)):
            self.n = input_n = tuple(int(i) for i in n)
            assert (np.asarray(input_n) > 0).all()  # n (counts) have to be positive
        else:
            self.n = n = int(n)
            input_n = (n,)
            assert (np.asarray(input_n) > 0).all()  # n (counts) have to be positive

        super().__init__(input_n, np.int8, seed)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self._shape  # type: ignore

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def sample(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Generates a single random sample from this space.

        A sample is drawn by independent, fair coin tosses (one toss per binary variable of the space).

        Args:
            mask: An optional np.ndarray to mask samples with expected shape of ``space.shape``.
                Where mask == 0 then the samples will be 0.

        Returns:
            Sampled values from space
        """
        if mask is not None:
            assert isinstance(
                mask, np.ndarray
            ), f"The expected type of the mask is np.ndarray, actual type: {type(mask)}"
            assert (
                mask.dtype == np.int8
            ), f"The expected dtype of the mask is np.int8, actual dtype: {mask.dtype}"
            assert (
                mask.shape == self.shape
            ), f"The expected shape of the mask is {self.shape}, actual shape: {mask.shape}"
            assert np.all(
                np.logical_or(mask == 0, mask == 1)
            ), f"All values of a mask should be 0 or 1, actual values: {mask}"

            return mask * self.np_random.integers(
                low=0, high=2, size=self.n, dtype=self.dtype
            )

        return self.np_random.integers(low=0, high=2, size=self.n, dtype=self.dtype)

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, Sequence):
            x = np.array(x)  # Promote list to array for contains check
        if self.shape != x.shape:
            return False
        return ((x == 0) | (x == 1)).all()

    def to_jsonable(self, sample_n) -> list:
        """Convert a batch of samples from this space to a JSONable data type."""
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n) -> list:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return f"MultiBinary({self.n})"

    def __eq__(self, other) -> bool:
        """Check whether `other` is equivalent to this instance."""
        return isinstance(other, MultiBinary) and self.n == other.n
