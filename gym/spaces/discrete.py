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
        # prng is terminating VizDoom with Fatal Error - Address not mapped to object (signal 11)
        # 82 is for HexEnv (9x9), which is the environment causing the Vizdoom crash
        if 82 == self.n: return int(round(time.time(), -1) % self.n)
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
        return "Discrete(%d)" % self.n
    def __eq__(self, other):
        return self.n == other.n
