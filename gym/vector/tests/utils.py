import numpy as np
import gym
import time

from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict

spaces = [
    Box(low=np.array(-1.), high=np.array(1.), dtype=np.float64),
    Box(low=np.array([0.]), high=np.array([10.]), dtype=np.float32),
    Box(low=np.array([-1., 0., 0.]), high=np.array([1., 1., 1.]), dtype=np.float32),
    Box(low=np.array([[-1., 0.], [0., -1.]]), high=np.ones((2, 2)), dtype=np.float32),
    Box(low=0, high=255, shape=(), dtype=np.uint8),
    Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8),
    Discrete(2),
    Tuple((Discrete(3), Discrete(5))),
    Tuple((Discrete(7), Box(low=np.array([0., -1.]), high=np.array([1., 1.]), dtype=np.float32))),
    MultiDiscrete([11, 13, 17]),
    MultiBinary(19),
    Dict({
        'position': Discrete(23),
        'velocity': Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float32)
    }),
    Dict({
        'position': Dict({'x': Discrete(29), 'y': Discrete(31)}),
        'velocity': Tuple((Discrete(37), Box(low=0, high=255, shape=(), dtype=np.uint8)))
    })
]

HEIGHT, WIDTH = 64, 64

class UnittestSlowEnv(gym.Env):
    def __init__(self, slow_reset=0.3, episode_length=1):
        super(UnittestSlowEnv, self).__init__()
        self.slow_reset = slow_reset
        self.episode_length = episode_length
        self._num_steps = 0

        self.observation_space = Box(low=0, high=255,
            shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.action_space = Box(low=0., high=1., shape=(), dtype=np.float32)

    def reset(self):
        self._num_steps = 0
        if self.slow_reset > 0:
            time.sleep(self.slow_reset)
        return self.observation_space.sample()

    def step(self, action):
        if (action < 0.):
            raise ValueError('The duration must be positive, got '
                '`{0}`.'.format(action))
        time.sleep(action)
        self._num_steps += 1
        observation = self.observation_space.sample()
        reward, done = np.random.rand(), (self._num_steps >= self.episode_length)
        return observation, reward, done, {}

def make_env(env_name, seed):
    def _make():
        env = gym.make(env_name)
        env.seed(seed)
        return env
    return _make

def make_slow_env(slow_reset, seed, episode_length=1):
    def _make():
        env = UnittestSlowEnv(slow_reset=slow_reset,
            episode_length=episode_length)
        env.seed(seed)
        return env
    return _make
