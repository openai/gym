import numpy as np

import gym, time
from gym.spaces import prng

class Discrete(gym.Space):
    """
    {0,1,...,n-1}

    Example usage:
    self.observation_space = spaces.Discrete(2)
    """
    def __init__(self, n):
        self.n = n
    def sample(self):
        return prng.np_random.randint(self.n)
    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n
    def __repr__(self):
        return "Discrete({})".format(self.n)
    def __eq__(self, other):
        return self.n == other.n

class Categorical(Discrete):
    """A discrete space with named elements. Element indices are made available
    as attributes (as long as they're legal Python names). e.g.
    >>> shapes = Categorical(['circle', 'triangle', 'rectangle'])
    >>> shapes.circle
    0
    """
    def __init__(self, named_members):
        self.n = len(named_members)
        self.named_members = list(named_members)
    def __repr__(self):
        return "Categorical({!r})".format(self.named_members)
    def __getattr__(self, attr):
        try:
            return self.named_members.index(attr)
        except IndexError:
            raise AttributeError("{} has no attribute {}".format(self, attr))
