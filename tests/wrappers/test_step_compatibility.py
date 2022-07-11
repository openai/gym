import pytest

import gym
from gym.spaces import Discrete
from gym.wrappers import StepAPICompatibility


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
    env = StepAPICompatibility(env(), True)
    step_returns = env.step(0)
    _, _, terminated, truncated, _ = step_returns
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


@pytest.mark.parametrize("env", [OldStepEnv, NewStepEnv])
@pytest.mark.parametrize("new_step_api", [None, False])
def test_step_compatibility_to_old_api(env, new_step_api):
    if new_step_api is None:
        env = StepAPICompatibility(env())  # default behavior is to retain old API
    else:
        env = StepAPICompatibility(env(), new_step_api)
    step_returns = env.step(0)
    assert len(step_returns) == 4
    _, _, done, _ = step_returns
    assert isinstance(done, bool)


@pytest.mark.parametrize("new_step_api", [None, True, False])
def test_step_compatibility_in_make(new_step_api):
    if new_step_api is None:
        with pytest.warns(
            DeprecationWarning, match="Initializing environment in old step API"
        ):
            env = gym.make("CartPole-v1")
    else:
        env = gym.make("CartPole-v1", new_step_api=new_step_api)

    env.reset()
    step_returns = env.step(0)
    if new_step_api:
        assert len(step_returns) == 5
        _, _, terminated, truncated, _ = step_returns
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    else:
        assert len(step_returns) == 4
        _, _, done, _ = step_returns
        assert isinstance(done, bool)
