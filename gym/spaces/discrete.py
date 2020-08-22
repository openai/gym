import numpy as np
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


from sklearn.preprocessing import OneHotEncoder

from collections.abc import Iterable

class FiniteSet(Discrete):
    r"""A discrete space in :math:`\{ a,b,c ... z \}`. 
    Example::
        >>> space = FiniteSet('news')
        >>> space.sample()
    """
    def __init__(self, actions):
        assert isinstance(actions, Iterable)
        self.__actions = tuple(actions)
        super(FiniteSet, self).__init__(len(actions))
        
    @property
    def actions(self):
        return self.__actions

    def sample(self):
        return self.np_random.choice(self.actions)

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
        return isinstance(other, FiniteSet) and self.n == other.n

    def __getitem__(self, k):
        return self.actions[k]

    def onehot_encode(self, k=None):
        if k is None:
            X = [[a] for a in self.actions]
            self.onehot_encoder = OneHotEncoder().fit(X)
            self.onehot = self.onehot_encoder.transform(X).toarray()
        else:
            if not hasattr(self, 'onehot_encoder'):
                self.onehot_encode()
            if isinstance(k, int):
                return self.onehot[k]
            elif isinstance(k, (list, tuple, np.ndarray)):
                return self.onehot_encoder.transform([[_] for _ in k]).toarray()
            else:
                return self.onehot_encoder.transform([[k]]).toarray()[0]
