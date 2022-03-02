from typing import Optional

import numpy as np
from .space import Space


class Discrete(Space[int]):
    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.

    A start value can be optionally specified to shift the range
    to :math:`\{ a, a+1, \\dots, a+n-1 \}`.

    Example::

        >>> Discrete(2)            # {0, 1}
        >>> Discrete(3, start=-1)  # {-1, 0, 1}

    """

    def __init__(self, n: int, seed: Optional[int] = None, start: int = 0):
        assert n > 0, "n (counts) have to be positive"
        assert isinstance(start, (int, np.integer))
        self.n = int(n)
        self.start = int(start)
        super().__init__((), np.int64, seed)

    def sample(self) -> int:
        return int(self.start + self.np_random.integers(self.n))

    def contains(self, x) -> bool:
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
            x.dtype.char in np.typecodes["AllInteger"] and x.shape == ()
        ):
            as_int = int(x)  # type: ignore
        else:
            return False
        return self.start <= as_int < self.start + self.n

    def __repr__(self) -> str:
        if self.start != 0:
            return "Discrete(%d, start=%d)" % (self.n, self.start)
        return "Discrete(%d)" % self.n

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Discrete)
            and self.n == other.n
            and self.start == other.start
        )

    def __setstate__(self, state):
        super().__setstate__(state)

        # Don't mutate the original state
        state = dict(state)

        # Allow for loading of legacy states.
        # See https://github.com/openai/gym/pull/2470
        if "start" not in state:
            state["start"] = 0

        # Update our state
        self.__dict__.update(state)
