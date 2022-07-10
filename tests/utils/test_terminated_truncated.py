import pytest

import gym
from gym.spaces import Discrete
from gym.vector import AsyncVectorEnv, SyncVectorEnv
from gym.wrappers import TimeLimit


# An environment where termination happens after 20 steps
class DummyEnv(gym.Env):
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Discrete(2)
        self.terminal_timestep = 20

        self.timestep = 0

    def step(self, action):
        self.timestep += 1
        terminated = True if self.timestep >= self.terminal_timestep else False
        truncated = False

        return 0, 0, terminated, truncated, {}

    def reset(self):
        self.timestep = 0
        return 0


@pytest.mark.parametrize("time_limit", [10, 20, 30])
def test_terminated_truncated(time_limit):
    test_env = TimeLimit(DummyEnv(), time_limit, new_step_api=True)

    terminated = False
    truncated = False
    test_env.reset()
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = test_env.step(0)

    if test_env.terminal_timestep < time_limit:
        assert terminated
        assert not truncated
    elif test_env.terminal_timestep == time_limit:
        assert (
            terminated
        ), "`terminated` should be True even when termination and truncation happen at the same step"
        assert (
            truncated
        ), "`truncated` should be True even when termination and truncation occur at same step "
    else:
        assert not terminated
        assert truncated


def test_terminated_truncated_vector():
    env0 = TimeLimit(DummyEnv(), 10, new_step_api=True)
    env1 = TimeLimit(DummyEnv(), 20, new_step_api=True)
    env2 = TimeLimit(DummyEnv(), 30, new_step_api=True)

    async_env = AsyncVectorEnv(
        [lambda: env0, lambda: env1, lambda: env2], new_step_api=True
    )
    async_env.reset()
    terminateds = [False, False, False]
    truncateds = [False, False, False]
    counter = 0
    while not all([x or y for x, y in zip(terminateds, truncateds)]):
        counter += 1
        _, _, terminateds, truncateds, _ = async_env.step(
            async_env.action_space.sample()
        )
    print(counter)
    assert counter == 20
    assert all(terminateds == [False, True, True])
    assert all(truncateds == [True, True, False])

    sync_env = SyncVectorEnv(
        [lambda: env0, lambda: env1, lambda: env2], new_step_api=True
    )
    sync_env.reset()
    terminateds = [False, False, False]
    truncateds = [False, False, False]
    counter = 0
    while not all([x or y for x, y in zip(terminateds, truncateds)]):
        counter += 1
        _, _, terminateds, truncateds, _ = sync_env.step(
            async_env.action_space.sample()
        )
    assert counter == 20
    assert all(terminateds == [False, True, True])
    assert all(truncateds == [True, True, False])
