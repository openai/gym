import pytest
import numpy as np

import gym
from gym.wrappers import NormalizeObservation, NormalizeReward


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_normalize_observation(env_id):
    env = gym.make(env_id)
    wrapped_env = NormalizeObservation(gym.make(env_id))
    unbiased = []

    env.seed(0)
    wrapped_env.seed(0)

    obs = env.reset()
    wrapped_obs = wrapped_env.reset()
    unbiased.append(obs)

    for t in range(env.spec.max_episode_steps):
        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)
        wrapped_obs, _, wrapped_done, _ = wrapped_env.step(action)
        unbiased.append(obs)

        mean = np.mean(unbiased, 0)
        var = np.var(unbiased, 0)
        assert np.allclose(wrapped_env.obs_moments.mean, mean, atol=1e-5)
        assert np.allclose(wrapped_env.obs_moments.var, var, atol=1e-4)

        assert done == wrapped_done
        if done:
            break


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
@pytest.mark.parametrize('gamma', [0.5, 0.99])
def test_normalize_reward(env_id, gamma):
    env = gym.make(env_id)
    wrapped_env = NormalizeReward(gym.make(env_id), gamma=gamma)
    unbiased = []

    env.seed(0)
    wrapped_env.seed(0)

    for n in range(10):
        obs = env.reset()
        wrapped_obs = wrapped_env.reset()
        G = 0.0
        for t in range(env.spec.max_episode_steps):
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            _, wrapped_reward, wrapped_done, _ = wrapped_env.step(action)
            assert done == wrapped_done

            G = reward + gamma*G
            unbiased.append(G)

            if done:
                break

            mean = np.mean(unbiased, 0)
            var = np.var(unbiased, 0)
            assert wrapped_env.all_returns == G

            assert np.allclose(wrapped_env.reward_moments.mean, mean, atol=1e-4)
            assert np.allclose(wrapped_env.reward_moments.var, var, atol=1e-3)
