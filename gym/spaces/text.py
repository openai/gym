"""Implementation of a space that represents textual strings."""
from typing import Any, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np

from gym.spaces.space import Space
from gym.utils import seeding

alphanumeric: FrozenSet[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


class Text(Space[str]):
    r"""A space representing a string comprised of characters from a given charset.

    Example::
        >>> # {"", "B5", "hello", ...}
        >>> Text(5)
        >>> # {"0", "42", "0123456789", ...}
        >>> import string
        >>> Text(min_length = 1,
        ...      max_length = 10,
        ...      charset = string.digits)
    """

    def __init__(
        self,
        max_length: int,
        *,
        min_length: int = 0,
        charset: Union[Set[str], str] = alphanumeric,
        seed: Optional[Union[int, seeding.RandomNumberGenerator]] = None,
    ):
        r"""Constructor of :class:`Text` space.

        Both bounds for text length are inclusive.

        Args:
            min_length (int): Minimum text length (in characters).
            max_length (int): Maximum text length (in characters).
            charset (Union[set, SupportsIndex]): Character set, defaults to the lower and upper english alphabet plus latin digits.
            seed: The seed for sampling from the space.
        """
        assert np.issubdtype(
            type(min_length), np.integer
        ), f"Expects the min_length to be an integer, actual type: {type(min_length)}"
        assert np.issubdtype(
            type(max_length), np.integer
        ), f"Expects the max_length to be an integer, actual type: {type(max_length)}"
        assert (
            0 <= min_length
        ), f"Minimum text length must be non-negative, actual value: {min_length}"
        assert (
            min_length <= max_length
        ), f"The min_length must be less than or equal to the max_length, min_length: {min_length}, max_length: {max_length}"

        self.min_length: int = int(min_length)
        self.max_length: int = int(max_length)
        self.charset: FrozenSet[str] = frozenset(charset)
        self._charlist: List[str] = list(charset)
        self._charset_str: str = "".join(sorted(self._charlist))

        # As the shape is dynamic (between min_length and max_length) then None
        super().__init__(dtype=str, seed=seed)

    def sample(
        self, mask: Optional[Tuple[Optional[int], Optional[np.ndarray]]] = None
    ) -> str:
        """Generates a single random sample from this space with by default a random length between `min_length` and `max_length` and sampled from the `charset`.

        Args:
            mask: An optional tuples of length and mask for the text.
                The length is expected to be between the `min_length` and `max_length` otherwise a random integer between `min_length` and `max_length` is selected.
                For the mask, we expect a numpy array of length of the charset passed with dtype == np.int8

        Returns:
            A sampled string from the space
        """
        if mask is not None:
            length, charlist_mask = mask
            if length is not None:
                assert self.min_length <= length <= self.max_length

            if charlist_mask is not None:
                assert isinstance(charlist_mask, np.ndarray)
                assert charlist_mask.dtype is np.int8
                assert charlist_mask.shape == (len(self._charlist),)
        else:
            length, charlist_mask = None, None

        if length is None:
            length = self.np_random.randint(self.min_length, self.max_length + 1)

        if charlist_mask is None:
            string = self.np_random.choice(self._charlist, size=length)
        else:
            masked_charlist = self._charlist[np.where(mask)[0]]
            string = self.np_random.choice(masked_charlist, size=length)

        return "".join(string)

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, str):
            if self.min_length <= len(x) <= self.max_length:
                return all(c in self.charset for c in x)
        return False

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return (
            f"Text({self.min_length}, {self.max_length}, charset={self._charset_str})"
        )

    def __eq__(self, other) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Text)
            and self.min_length == other.min_length
            and self.max_length == other.max_length
            and self.charset == other.charset
        )
