import numpy as np
import gym

# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class Normalize(gym.core.Wrapper):
    def __init__(
        self,
        env,
        norm_obs=True,
        norm_return=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,
    ):
        super(Normalize, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = (
                RunningMeanStd(shape=self.single_observation_space.shape)
                if norm_obs
                else None
            )
        else:
            self.obs_rms = (
                RunningMeanStd(shape=self.observation_space.shape) if norm_obs else None
            )
        self.return_rms = RunningMeanStd(shape=()) if norm_return else None
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        if not self.is_vector_env:
            obs, rews, dones = np.array([obs]), np.array([rews]), np.array([dones])
        self.returns = self.returns * self.gamma + rews
        obs = self._obfilt(obs)
        if self.return_rms:
            self.return_rms.update(self.returns)
            rews = np.clip(
                rews / np.sqrt(self.return_rms.var + self.epsilon),
                -self.clip_reward,
                self.clip_reward,
            )
        self.returns[dones] = 0.0
        if not self.is_vector_env:
            return obs[0], rews[0], dones[0], infos
        return obs, rews, dones, infos

    def _obfilt(self, obs):
        if self.obs_rms:
            self.obs_rms.update(obs)
            obs = np.clip(
                (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            )
            return obs
        else:
            return obs

    def reset(self):
        obs = self.env.reset()
        if self.is_vector_env:
            obs = self._obfilt(obs)
            return obs
        else:
            obs = self._obfilt(np.array([obs]))[0]
            return obs
