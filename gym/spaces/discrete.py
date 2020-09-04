import numpy as np
from scipy.stats import rv_discrete
from .space import Space


class Discrete(Space):
    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`. 

    Example::

        >>> Discrete(2)

    """
    def __init__(self, n):
        assert n >= 0
        self.n = n
        super(Discrete, self).__init__((), np.int64)

    def sample(self):
        return self.np_random.randint(self.n)

    def sample_with_proba(self, p):
        # p is the distribution of 0, 1, ..., n-1
        d = rv_discrete(values=(np.arange(self.n), p))
        return d.rvs()

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.char in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return 0 <= as_int < self.n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, Discrete) and self.n == other.n


from collections.abc import Iterable

class FiniteSet(Discrete):
    r"""A discrete space in :math:`\{ a,b,c ... z \}`. 
    Example::
        >>> space = FiniteSet('news')
        >>> space.sample()
    """
    def __init__(self, actions):
        # the members have been ordered in actions, indexed by integers from 0
        assert isinstance(actions, Iterable)
        self.__actions = tuple(actions)
        super(FiniteSet, self).__init__(len(actions))
        
    @property
    def actions(self):
        return self.__actions

    def sample(self):
        k = super(FiniteSet, self).sample()
        return self[k]
    
    def sample_with_proba(self, proba=None):
        k = super(FiniteSet, self).sample_with_proba(proba)
        return self[k]


    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.char in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return x in self.actions
        return 0 <= as_int < self.n

    def __repr__(self):
        return "FiniteSet(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, FiniteSet) and set(self.actions) == set(other.actions)

    def __getitem__(self, k):
        return self.actions[k]
