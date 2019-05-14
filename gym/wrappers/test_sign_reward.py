import pytest

import gym
from gym.wrappers import SignReward


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_sign_reward(env_id):
    env = gym.make(env_id)
    wrapped_env = SignReward(env)

    env.reset()
    wrapped_env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        _, wrapped_reward, done, _ = wrapped_env.step(action)
        assert wrapped_reward in [-1.0, 0.0, 1.0]
        if done:
            break
