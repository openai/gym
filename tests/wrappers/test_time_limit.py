import pytest

import gym
from gym.envs.classic_control import CartPoleEnv
from gym.envs.classic_control.pendulum import PendulumEnv
from gym.wrappers import TimeLimit
from tests.testing_env import TestingEnv
from tests.wrappers.utils import has_wrapper


def test_elapsed_steps():
    env = gym.make("CartPole-v1", disable_env_checker=True)
    assert has_wrapper(env, TimeLimit)
    assert isinstance(env, TimeLimit)

    env.reset(seed=0)
    assert env.elapsed_steps == 0
    env.step(env.action_space.sample())
    assert env.elapsed_steps == 1


@pytest.mark.parametrize("double_wrap", [False, True])
def test_override_info(double_wrap, max_episode_length=10):
    # The pendulum env does not terminate by default, so we are sure termination is only due to timeout
    env = TimeLimit(CartPoleEnv(), max_episode_length)

    # TimeLimit wrapper should not overwrite the TimeLimit.truncated key if it was already set
    if double_wrap:
        env = TimeLimit(env, max_episode_length)

    env.reset(seed=0)
    done, n_steps, info = False, 0, {}
    while not done:
        _, _, done, info = env.step(env.action_space.sample())
        n_steps += 1

    assert n_steps == max_episode_length
    assert "TimeLimit.truncated" in info
    assert info["TimeLimit.truncated"] is True


@pytest.mark.parametrize("double_wrap", [False, True])
def test_termination_on_last_step(double_wrap, max_episode_length=1):
    # Special case: termination at the last timestep but not due to timeout
    env = PendulumEnv()
    env.step = lambda _: (env.observation_space.sample(), 0, True, {})
    env = TimeLimit(env, max_episode_length)

    if double_wrap:
        env = TimeLimit(env, max_episode_length)

    env.reset()
    _, _, done, info = env.step(env.action_space.sample())

    assert done is True
    assert "TimeLimit.truncated" in info
    assert info["TimeLimit.truncated"] is False


def test_env_failure():
    """On environment failures, then time limit shouldn't update elapsed_steps"""

    def _raise_exception(*_):
        raise gym.error.Error()

    env = TimeLimit(TestingEnv(reset_fn=_raise_exception), 10)

    assert env.elapsed_steps is None
    with pytest.raises(gym.error.Error):
        env.reset()
    assert env.elapsed_steps is None

    env = TimeLimit(TestingEnv(step_fn=_raise_exception), 10)

    assert env.elapsed_steps is None
    env.reset()
    assert env.elapsed_steps == 0
    with pytest.raises(gym.error.Error):
        env.step(env.action_space.sample())
    assert env.elapsed_steps == 0
