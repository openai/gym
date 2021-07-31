import pytest

import numpy as np

import gym
from gym.wrappers import TransformReward


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v0"])
def test_transform_reward(env_id):
    # use case #1: scale
    scales = [0.1, 200]
    for scale in scales:
        env = gym.make(env_id)
        wrapped_env = TransformReward(gym.make(env_id), lambda r: scale * r)
        action = env.action_space.sample()

        env.seed(0)
        env.reset()
        wrapped_env.seed(0)
        wrapped_env.reset()

        _, reward, _, _ = env.step(action)
        _, wrapped_reward, _, _ = wrapped_env.step(action)

        assert wrapped_reward == scale * reward
    del env, wrapped_env

    # use case #2: clip
    min_r = -0.0005
    max_r = 0.0002
    env = gym.make(env_id)
    wrapped_env = TransformReward(gym.make(env_id), lambda r: np.clip(r, min_r, max_r))
    action = env.action_space.sample()

    env.seed(0)
    env.reset()
    wrapped_env.seed(0)
    wrapped_env.reset()

    _, reward, _, _ = env.step(action)
    _, wrapped_reward, _, _ = wrapped_env.step(action)

    assert abs(wrapped_reward) < abs(reward)
    assert wrapped_reward == -0.0005 or wrapped_reward == 0.0002
    del env, wrapped_env

    # use case #3: sign
    env = gym.make(env_id)
    wrapped_env = TransformReward(gym.make(env_id), lambda r: np.sign(r))

    env.seed(0)
    env.reset()
    wrapped_env.seed(0)
    wrapped_env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        _, wrapped_reward, done, _ = wrapped_env.step(action)
        assert wrapped_reward in [-1.0, 0.0, 1.0]
        if done:
            break
    del env, wrapped_env
