import numpy as np
from .space import Space


class Discrete(Space):
    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.

    A start value can be optionally specified to shift the range
    to :math:`\{ a, a+1, \\dots, a+n-1 \}`.

    Example::

        >>> Discrete(2)
        >>> Discrete(3, start=-1)  # {-1, 0, 1}

    """

    def __init__(self, n, seed=None, start=0):
        assert n >= 0 and isinstance(start, (int, np.integer))
        self.n = n
        self.start = int(start)
        super().__init__((), np.int64, seed)

    def sample(self):
        return self.start + self.np_random.randint(self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
            x.dtype.char in np.typecodes["AllInteger"] and x.shape == ()
        ):
            as_int = int(x)
        else:
            return False
        return self.start <= as_int < self.start + self.n

    def __repr__(self):
        if self.start != 0:
            return "Discrete(%d, start=%d)" % (self.n, self.start)
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return (
            isinstance(other, Discrete)
            and self.n == other.n
            and self.start == other.start
        )
