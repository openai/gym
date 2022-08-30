"""Set of wrappers for normalizing actions and observations."""
import numpy as np

import gym


# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation(gym.core.Wrapper):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


class NormalizeReward(gym.core.Wrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma + rews
        rews = self.normalize(rews)
        dones = np.logical_or(terminateds, truncateds)
        self.returns[dones] = 0.0
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)
