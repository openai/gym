"""Implementation of a space that represents textual strings."""
from typing import Optional, Union, Set, SupportsIndex

import numpy as np

from gym.spaces.space import Space
from gym.utils import seeding

alphanumeric = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# TODO: Add support for:
#  * lexicons
#  * regular expressions
#     - for sample() and contains()
#  * tokenization

class Text(Space[str]):
    r"""A space representing a string comprised of characters from a given charset.
    Example::
        >>> # {"", "B5", "hello", ...}
        >>> Text(5)
        >>> # {"01", "0123456789", ...}
        >>> Text(min_length = 2,
        ...      max_length = 10,
        ...      charset = string.digits)
    """
    
    def __init__(
        self,
        max_length: int,
        *,
        min_length: int = 0,
        charset: Union[set, SupportsIndex] = alphanumeric,
        seed: Optional[Union[int, seeding.RandomNumberGenerator]] = None,
    ):
        r"""Constructor of :class:`Text` space.
        Both bounds for text length are inclusive.
        Args:
            min_length (int): Minimum text length (in characters).
            max_length (int): Maximum text length (in characters).
            charset (Union[set, SupportsIndex]): Character set.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the ``Dict`` space.
        """
        assert isinstance(min_length, (int, np.integer))
        assert isinstance(max_length, (int, np.integer))
        assert min_length >= 0, "Minimum text length must be non-negative"
        assert max_length >= min_length, "Maximum text length must be greater than or equal to minimum text length"
        self.min_length = min_length
        self.max_length = max_length
        self.charset = set(charset)
        self._charlist = list(charset)
        self._charset_str = "".join(sorted(self._charlist))
        super().__init__((), str, seed)
        
    def sample(self) -> str:
        """Generates a single random sample from this space.
        Returns:
            A sampled string from the space
        """
        length = np.random.randint(self.min_length, self.max_length + 1)
        string = np.random.choice(self._charlist, size = length)
        string = "".join(string)
        return string
    
    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, str) and self.min_length <= len(x) <= self.max_length:
            for c in x:
                if c not in self.charset:
                    return False
            return True
        else:
            return False
        
    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return "Text(%d, %d, charset=%s)" % (
            self.min_length,
            self.max_length,
            self._charset_str)

    def __eq__(self, other) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Text)
            and self.min_length == other.min_length
            and self.max_length == other.max_length
            and self.charset == other.charset
        )
        
