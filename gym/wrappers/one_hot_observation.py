import gym
import numpy as np


def one_hot(a: np.array, size: int) -> np.array:
    b = np.zeros((size), dtype=np.float32)
    b[a] = 1
    return b


class OneHotObservation(gym.ObservationWrapper):
    """
    This wrapper one hot encodes the `Discrete` observation space.
    For example, `Discrete(3)` observation space will be converted
    to `Box(0, 1, (3,))` observation space, where
    `(1, 0, 0)`, `(0, 1, 0)` and `(0, 0, 1)` corresponds to the three
    discrete states

    Example::

        >>> import gym
        >>> env = gym.make('Taxi-v3')
        >>> env = gym.wrappers.OneHotObservation(env)
        >>> env.reset()
        >>> env.observation_space
        Box(0, 1, (500,))

    Args:
        env (Env): environment
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(self.env.observation_space, gym.spaces.Discrete):
            raise ValueError(
                "This wrapper can only apply to the Discrete observation space"
            )
        self.n = self.env.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, (self.n,))

    def observation(self, obs):
        return one_hot(np.array(obs), self.n)
