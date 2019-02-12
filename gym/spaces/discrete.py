import numpy as np

from .space import Space


class Discrete(Space):
    r"""A discrete space in :math:`\{ 0, 1, \dots, n-1 \}`. 
    
    Example::
    
        >>> Discrete(2)
        
    """
    def __init__(self, n):
        assert n >= 0
        self.n = n
        super(Discrete, self).__init__((), np.int64)

    def sample(self):
        return self.np_random.randint(self.n)

    @property
    def flat_dim(self):
        return int(self.n)

    def flatten(self, x):
        # One-hot representation
        onehot = np.zeros(self.n)
        onehot[x] = 1.0
        return onehot

    def unflatten(self, x):
        # Extract index from one-hot representation
        return int(np.nonzero(x)[0][0])

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, x):
        return isinstance(x, Discrete) and x.n == self.n
