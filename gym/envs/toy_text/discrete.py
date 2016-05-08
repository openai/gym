from gym import Env
from gym import spaces
import numpy as np

def categorical_sample(prob_n):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()


class DiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)

    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS


    """
    def __init__(self, nS, nA, P, isd):
        self.action_space = spaces.Discrete(nA)
        self.observation_space = spaces.Discrete(nS)
        self.nA = nA
        self.P = P
        self.isd = isd
        self.lastaction=None # for rendering

    @property
    def nS(self):
        return self.observation_space.n

    def _reset(self):
        self.s = categorical_sample(self.isd)
        return self.s

    def _step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions])
        p, s, r, d= transitions[i]
        self.s = s
        self.lastaction=a
        return (s, r, d, {"prob" : p})
