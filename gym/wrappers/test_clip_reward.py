import pytest

import gym
from gym.wrappers import ClipReward


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0', 'MountainCar-v0'])
def test_clip_reward(env_id):
    env = gym.make(env_id)
    wrapped_env = ClipReward(env, -0.0005, 0.0002)

    env.reset()
    wrapped_env.reset()

    action = env.action_space.sample()

    _, reward, _, _ = env.step(action)
    _, wrapped_reward, _, _ = wrapped_env.step(action)

    assert abs(wrapped_reward) < abs(reward)
    assert wrapped_reward == -0.0005 or wrapped_reward == 0.0002
