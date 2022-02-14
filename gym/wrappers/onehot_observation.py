import gym
import numpy as np


def one_hot(a, size):
    b = np.zeros((size))
    b[a] = 1
    return b

class ProcessObsInputEnv(gym.ObservationWrapper):
    """
    This wrapper handles inputs from `Discrete` and `Box` observation space.
    If the `env.observation_space` is of `Discrete` type, 
    it returns the one-hot encoding of the state
    """
    def __init__(self, env):
        super().__init__(env)
        self.n = None
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.n = self.env.observation_space.n
            self.observation_space = gym.spaces.Box(0, 1, (self.n,))

    def observation(self, obs):
        if self.n:
            return one_hot(np.array(obs), self.n)
        return obs
