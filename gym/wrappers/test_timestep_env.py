import pytest

import numpy as np

import gym
from gym.wrappers import TimeStepEnv


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_timestep_env(env_id):
    env = gym.make(env_id)
    wrapped_env = TimeStepEnv(gym.make(env_id))

    env.seed(0)
    wrapped_env.seed(0)

    obs = env.reset()
    timestep = wrapped_env.reset()
    assert timestep.first()
    assert np.allclose(timestep.observation, obs)

    for t in range(env.spec.max_episode_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        timestep = wrapped_env.step(action)
        assert np.allclose(timestep.observation, obs)
        assert timestep.reward == reward
        assert timestep.done == done
        assert timestep.info == info
        if done:
            assert timestep.last()
            if 'TimeLimit.truncated' in info and info['TimeLimit.truncated']:
                assert timestep.time_limit()
            else:
                assert timestep.terminal()
            break
        else:
            assert timestep.mid()
