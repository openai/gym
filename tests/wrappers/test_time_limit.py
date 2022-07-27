import pytest

import gym
from gym.envs.classic_control.pendulum import PendulumEnv
from gym.wrappers import TimeLimit


def test_time_limit_reset_info():
    env = gym.make("CartPole-v1", disable_env_checker=True)
    env = TimeLimit(env)
    ob_space = env.observation_space
    obs = env.reset()
    assert ob_space.contains(obs)
    del obs
    obs = env.reset(return_info=False)
    assert ob_space.contains(obs)
    del obs
    obs, info = env.reset(return_info=True)
    assert ob_space.contains(obs)
    assert isinstance(info, dict)


@pytest.mark.parametrize("double_wrap", [False, True])
def test_time_limit_wrapper(double_wrap):
    # The pendulum env does not terminate by default
    # so we are sure termination is only due to timeout
    env = PendulumEnv()
    max_episode_length = 20
    env = TimeLimit(env, max_episode_length)
    if double_wrap:
        # TimeLimit wrapper should not overwrite
        # the TimeLimit.truncated key
        # if it was already set
        env = TimeLimit(env, max_episode_length)
    env.reset()
    done = False
    n_steps = 0
    info = {}
    while not done:
        n_steps += 1
        _, _, done, info = env.step(env.action_space.sample())

    assert n_steps == max_episode_length
    assert "TimeLimit.truncated" in info
    assert info["TimeLimit.truncated"] is True


@pytest.mark.parametrize("double_wrap", [False, True])
def test_termination_on_last_step(double_wrap):
    # Special case: termination at the last timestep
    # but not due to timeout
    env = PendulumEnv()

    def patched_step(_action):
        return env.observation_space.sample(), 0.0, True, {}

    env.step = patched_step

    max_episode_length = 1
    env = TimeLimit(env, max_episode_length)
    if double_wrap:
        env = TimeLimit(env, max_episode_length)
    env.reset()
    _, _, done, info = env.step(env.action_space.sample())
    assert done is True
    assert "TimeLimit.truncated" in info
    assert info["TimeLimit.truncated"] is False
