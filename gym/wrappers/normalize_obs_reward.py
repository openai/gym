import numpy as np
import gym


class RunningMeanVar(object):
    r"""Estimates sample mean and variance by using `Chan's method`_. 

    It supports for both scalar and multi-dimensional data, however, the input is
    expected to be batched. The first dimension is always treated as batch dimension.

    .. note::

        For better precision, we handle the data with `np.float64`.

    .. warning::

        To use estimated moments for standardization, remember to keep the precision `np.float64`
        and calculated as ..math:`\frac{x - \mu}{\sqrt{\sigma^2 + 10^{-8}}}`. 

    Example:

        >>> f = RunningMeanVar(shape=())
        >>> f([1, 2])
        >>> f([3])
        >>> f([4])
        >>> f.mean
        2.499937501562461
        >>> f.var
        1.2501499923440393

    .. _Chan's method:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    """
    def __init__(self, shape):
        self.shape = shape
      import gym
  self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.N = 1e-8  # numerical stability for variance term, and 1e-4 is for std

    def __call__(self, x):
        r"""Update the mean and variance given an additional batched data. 

        Args:
            x (object): additional batched data.
        """
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim == len(self.shape) + 1, f'expected {len(self.shape) + 1}, got {x.ndim}'

        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_N = x.shape[0]

        new_N = self.N + batch_N
        delta = batch_mean - self.mean
        new_mean = self.mean + delta*(batch_N/new_N)
        M_A = self.N*self.var
        M_B = batch_N*batch_var
        M_X = M_A + M_B + (delta**2)*((self.N*batch_N)/new_N)
        new_var = M_X/new_N

        self.mean = new_mean
        self.var = new_var
        self.N = new_N

    @property
    def n(self):
        r"""Returns the total number of samples so far. """
        return int(self.N)


class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env, clip=5., constant_moments=None):
        super().__init__(env)
        self.clip = clip
        self.constant_moments = constant_moments
        self.eps = 1e-8
        if constant_moments is None:
            self.obs_moments = RunningMeanVar(shape=env.observation_space.shape)
        else:
            self.constant_mean, self.constant_var = constant_moments
            
    def observation(self, observation):
        if self.constant_moments is None:
            self.obs_moments([observation])
            mean = self.obs_moments.mean
            std = np.sqrt(self.obs_moments.var + self.eps)
        else:
            mean = self.constant_mean
            std = np.sqrt(self.constant_var + self.eps)
        observation = np.clip((observation - mean)/std, -self.clip, self.clip)
        return observation


class NormalizeReward(gym.RewardWrapper):
    def __init__(self, env, clip=10., gamma=0.99, constant_var=None):
        super().__init__(env)
        self.clip = clip
        assert gamma > 0.0 and gamma < 1.0, 'we do not allow discounted factor as 1.0. See docstring for details. '
        self.gamma = gamma
        self.constant_var = constant_var
        self.eps = 1e-8
        if constant_var is None:
            self.reward_moments = RunningMeanVar(shape=())
        
        # Buffer to save discounted returns from each environment
        self.all_returns = 0.0
        
    def reset(self):
        # Reset returns buffer
        self.all_returns = 0.0
        return super().reset()
    
    def step(self, action):
        observation, reward, done, info = super().step(action)
        # Set discounted return buffer as zero if episode terminates
        if done:
            self.all_returns = 0.0
        return observation, reward, done, info
    
    def reward(self, reward):
        if self.constant_var is None:
            self.all_returns = reward + self.gamma*self.all_returns
            self.reward_moments([self.all_returns])
            std = np.sqrt(self.reward_moments.var + self.eps)
        else:
            std = np.sqrt(self.constant_var + self.eps)
        # Do NOT subtract from mean, but only divided by std
        reward = np.clip(reward/std, -self.clip, self.clip)
        return reward
