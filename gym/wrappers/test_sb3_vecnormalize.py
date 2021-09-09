from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnvWrapper
import gym
import numpy as np


class DummyRewardEnv(gym.Env):
    metadata = {}

    def __init__(self, return_reward_idx=0):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0]), high=np.array([1.0])
        )
        self.returned_rewards = [0, 1, 2, 3, 4]
        self.return_reward_idx = return_reward_idx
        self.t = self.return_reward_idx

    def step(self, action):
        self.t += 1
        return np.array([self.t]), self.t, self.t == len(self.returned_rewards), {}

    def reset(self):
        self.t = self.return_reward_idx
        return np.array([self.t])


def make_env(return_reward_idx):
    def thunk():
        env = DummyRewardEnv(return_reward_idx)
        return env

    return thunk


class OriginalBaselinesRunningMeanStd(object):
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


class OriginalBaselinesVecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(
        self,
        venv,
        ob=True,
        ret=True,
        clipob=10.0,
        cliprew=10.0,
        gamma=0.99,
        epsilon=1e-8,
        use_tf=False,
    ):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = (
            OriginalBaselinesRunningMeanStd(shape=self.observation_space.shape)
            if ob
            else None
        )
        self.ret_rms = OriginalBaselinesRunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(
                rews / np.sqrt(self.ret_rms.var + self.epsilon),
                -self.cliprew,
                self.cliprew,
            )
        self.ret[news] = 0.0
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob,
                self.clipob,
            )
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)


env_fns = [make_env(0), make_env(1)]

print("SB3's VecNormalize")
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

envs = DummyVecEnv(env_fns)
envs = VecNormalize(envs)
envs.reset()
print(envs.obs_rms.mean)
obs, reward, done, _ = envs.step(
    [envs.action_space.sample(), envs.action_space.sample()]
)
print(envs.obs_rms.mean)

print("OriginalBaselinesVecNormalize")
envs = DummyVecEnv(env_fns)
envs = OriginalBaselinesVecNormalize(envs)
envs.reset()
print(envs.ob_rms.mean)
obs, reward, done, _ = envs.step(
    [envs.action_space.sample(), envs.action_space.sample()]
)
print(envs.ob_rms.mean)
