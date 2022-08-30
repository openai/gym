import pytest

import gym
from gym.envs.classic_control.pendulum import PendulumEnv
from gym.wrappers import TimeLimit


def test_time_limit_reset_info():
    env = gym.make("CartPole-v1", disable_env_checker=True)
    env = TimeLimit(env)
    ob_space = env.observation_space
    obs, info = env.reset()
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
        env = TimeLimit(env, max_episode_length)
    env.reset()
    terminated, truncated = False, False
    n_steps = 0
    info = {}
    while not (terminated or truncated):
        n_steps += 1
        _, _, terminated, truncated, info = env.step(env.action_space.sample())

    assert n_steps == max_episode_length
    assert truncated


@pytest.mark.parametrize("double_wrap", [False, True])
def test_termination_on_last_step(double_wrap):
    # Special case: termination at the last timestep
    # Truncation due to timeout also happens at the same step

    env = PendulumEnv()

    def patched_step(_action):
        return env.observation_space.sample(), 0.0, True, False, {}

    env.step = patched_step

    max_episode_length = 1
    env = TimeLimit(env, max_episode_length)
    if double_wrap:
        env = TimeLimit(env, max_episode_length)
    env.reset()
    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    assert terminated is True
    assert truncated is True
