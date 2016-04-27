import numpy as np
from gym import Space

class Discrete(Space):
    """
    {0,1,...,n-1}
    """
    def __init__(self, n):
        self.n = n
    def sample(self):
        return np.random.randint(self.n)
    def contains(self, x):
        return isinstance(x, int) and x >= 0 and x < self.n
    def __repr__(self):
        return "Discrete(%d)" % self.n
    def __eq__(self, other):
        return self.n == other.n
