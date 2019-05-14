import pytest

import gym
from gym.wrappers import ScaleReward


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
@pytest.mark.parametrize('scale', [0.1, 200])
def test_scale_reward(env_id, scale):
    env = gym.make(env_id)

    action = env.action_space.sample()

    env.seed(0)
    env.reset()
    _, reward, _, _ = env.step(action)

    wrapped_env = ScaleReward(env, scale)
    env.seed(0)
    wrapped_env.reset()
    _, wrapped_reward, _, _ = wrapped_env.step(action)

    assert wrapped_reward == scale*reward
