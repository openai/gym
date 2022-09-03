"""Implementation of a space that represents textual strings."""
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple, Union

import numpy as np

from gym.spaces.space import Space

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
        min_length: int = 1,
        charset: Union[Set[str], str] = alphanumeric,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        r"""Constructor of :class:`Text` space.

        Both bounds for text length are inclusive.

        Args:
            min_length (int): Minimum text length (in characters). Defaults to 1 to prevent empty strings.
            max_length (int): Maximum text length (in characters).
            charset (Union[set], str): Character set, defaults to the lower and upper english alphabet plus latin digits.
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

        self._char_set: FrozenSet[str] = frozenset(charset)
        self._char_list: Tuple[str, ...] = tuple(charset)
        self._char_index: Dict[str, np.int32] = {
            val: np.int32(i) for i, val in enumerate(tuple(charset))
        }
        self._char_str: str = "".join(sorted(tuple(charset)))

        # As the shape is dynamic (between min_length and max_length) then None
        super().__init__(dtype=str, seed=seed)

    def sample(
        self, mask: Optional[Tuple[Optional[int], Optional[np.ndarray]]] = None
    ) -> str:
        """Generates a single random sample from this space with by default a random length between `min_length` and `max_length` and sampled from the `charset`.

        Args:
            mask: An optional tuples of length and mask for the text.
                The length is expected to be between the `min_length` and `max_length` otherwise a random integer between `min_length` and `max_length` is selected.
                For the mask, we expect a numpy array of length of the charset passed with `dtype == np.int8`.
                If the charlist mask is all zero then an empty string is returned no matter the `min_length`

        Returns:
            A sampled string from the space
        """
        if mask is not None:
            assert isinstance(
                mask, tuple
            ), f"Expects the mask type to be a tuple, actual type: {type(mask)}"
            assert (
                len(mask) == 2
            ), f"Expects the mask length to be two, actual length: {len(mask)}"
            length, charlist_mask = mask

            if length is not None:
                assert np.issubdtype(
                    type(length), np.integer
                ), f"Expects the Text sample length to be an integer, actual type: {type(length)}"
                assert (
                    self.min_length <= length <= self.max_length
                ), f"Expects the Text sample length be between {self.min_length} and {self.max_length}, actual length: {length}"

            if charlist_mask is not None:
                assert isinstance(
                    charlist_mask, np.ndarray
                ), f"Expects the Text sample mask to be an np.ndarray, actual type: {type(charlist_mask)}"
                assert (
                    charlist_mask.dtype == np.int8
                ), f"Expects the Text sample mask to be an np.ndarray, actual dtype: {charlist_mask.dtype}"
                assert charlist_mask.shape == (
                    len(self.character_set),
                ), f"expects the Text sample mask to be {(len(self.character_set),)}, actual shape: {charlist_mask.shape}"
                assert np.all(
                    np.logical_or(charlist_mask == 0, charlist_mask == 1)
                ), f"Expects all masks values to 0 or 1, actual values: {charlist_mask}"
        else:
            length, charlist_mask = None, None

        if length is None:
            length = self.np_random.integers(self.min_length, self.max_length + 1)

        if charlist_mask is None:
            string = self.np_random.choice(self.character_list, size=length)
        else:
            valid_mask = charlist_mask == 1
            valid_indexes = np.where(valid_mask)[0]
            if len(valid_indexes) == 0:
                if self.min_length == 0:
                    string = ""
                else:
                    # Otherwise the string will not be contained in the space
                    raise ValueError(
                        f"Trying to sample with a minimum length > 0 ({self.min_length}) but the character mask is all zero meaning that no character could be sampled."
                    )
            else:
                string = "".join(
                    self.character_list[index]
                    for index in self.np_random.choice(valid_indexes, size=length)
                )

        return "".join(string)

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, str):
            if self.min_length <= len(x) <= self.max_length:
                return all(c in self.character_set for c in x)
        return False

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return (
            f"Text({self.min_length}, {self.max_length}, characters={self.characters})"
        )

    def __eq__(self, other) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Text)
            and self.min_length == other.min_length
            and self.max_length == other.max_length
            and self.character_set == other.character_set
        )

    @property
    def character_set(self) -> FrozenSet[str]:
        """Returns the character set for the space."""
        return self._char_set

    @property
    def character_list(self) -> Tuple[str, ...]:
        """Returns a tuple of characters in the space."""
        return self._char_list

    def character_index(self, char: str) -> np.int32:
        """Returns a unique index for each character in the space's character set."""
        return self._char_index[char]

    @property
    def characters(self) -> str:
        """Returns a string with all Text characters."""
        return self._char_str

    @property
    def is_np_flattenable(self) -> bool:
        """The flattened version is an integer array for each character, padded to the max character length."""
        return True
