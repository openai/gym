import pytest

import gym
from gym.spaces import Discrete
from gym.wrappers import StepCompatibility


class OldStepEnv(gym.Env):
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Discrete(2)

    def step(self, action):
        obs = self.observation_space.sample()
        rew = 0
        done = False
        info = {}
        return obs, rew, done, info


class NewStepEnv(gym.Env):
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Discrete(2)

    def step(self, action):
        obs = self.observation_space.sample()
        rew = 0
        terminated = False
        truncated = False
        info = {}
        return obs, rew, terminated, truncated, info


@pytest.mark.parametrize("env", [OldStepEnv, NewStepEnv])
def test_step_compatibility_to_new_api(env):
    env = StepCompatibility(env(), True)
    step_returns = env.step(0)
    _, _, terminated, truncated, _ = step_returns
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


@pytest.mark.parametrize("env", [OldStepEnv, NewStepEnv])
@pytest.mark.parametrize("return_two_dones", [None, False])
def test_step_compatibility_to_old_api(env, return_two_dones):
    if return_two_dones is None:
        env = StepCompatibility(env())  # default behavior is to retain old API
    else:
        env = StepCompatibility(env(), return_two_dones)
    step_returns = env.step(0)
    assert len(step_returns) == 4
    _, _, done, _ = step_returns
    assert isinstance(done, bool)


@pytest.mark.parametrize("return_two_dones", [None, True, False])
def test_step_compatibility_in_make(return_two_dones):
    if return_two_dones is None:
        env = gym.make("CartPole-v1")  # check default behavior
    else:
        env = gym.make("CartPole-v1", return_two_dones=return_two_dones)

    env.reset()
    step_returns = env.step(0)
    if return_two_dones == True:  # new api
        assert len(step_returns) == 5
        _, _, terminated, truncated, _ = step_returns
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    else:  # old api
        assert len(step_returns) == 4
        _, _, done, _ = step_returns
        assert isinstance(done, bool)
