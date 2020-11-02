from functools import partial
from multiprocessing import cpu_count
from typing import Optional

import gym
import numpy as np
import pytest
from gym import spaces
from gym.vector.batched_vector_env import BatchedVectorEnv
from gym.vector.vector_env import FINAL_STATE_KEY

N_CPUS = cpu_count()


@pytest.mark.parametrize("batch_size", [1, 5, N_CPUS, 11, 24])
@pytest.mark.parametrize("n_workers", [1, 3, N_CPUS])
def test_right_shapes(batch_size: int, n_workers: Optional[int]):
    env_fn = partial(gym.make, "CartPole-v0")
    env_fns = [env_fn for _ in range(batch_size)]
    env = BatchedVectorEnv(env_fns, n_workers=n_workers)
    env.seed(123)
    
    assert env.observation_space.shape == (batch_size, 4)
    assert isinstance(env.action_space, spaces.Tuple)
    assert len(env.action_space) == batch_size
    
    obs = env.reset()
    assert obs.shape == (batch_size, 4)

    for i in range(3):
        actions = env.action_space.sample()
        assert actions in env.action_space
        obs, rewards, done, info = env.step(actions)
        assert obs.shape == (batch_size, 4)
        assert len(rewards) == batch_size
        assert len(done) == batch_size
        assert len(info) == batch_size

    env.close()


class DummyEnvironment(gym.Env):
    """ Dummy environment for testing.
    
    The reward is how close to the target value the state (a counter) is. The
    actions are:
    0:  keep the counter the same.
    1:  Increment the counter.
    2:  Decrement the counter.
    """
    def __init__(self, start: int = 0, max_value: int = 10, target: int = 5):
        self.max_value = max_value
        self.i = start
        self.start = start
        self.reward_range = (0, max_value)
        self.action_space = gym.spaces.Discrete(n=3)  # type: ignore
        self.observation_space = gym.spaces.Discrete(n=max_value)  # type: ignore

        self.target = target
        self.reward_range = (0, max(target, max_value - target))

        self.done: bool = False
        self._reset: bool = False

    def step(self, action: int):
        # The action modifies the state, producing a new state, and you get the
        # reward associated with that transition.
        if not self._reset:
            raise RuntimeError("Need to reset before you can step.")
        if action == 1:
            self.i += 1
        elif action == 2:
            self.i -= 1
        self.i %= self.max_value
        done = (self.i == self.target)
        reward = abs(self.i - self.target)
        print(self.i, reward, done, action)
        return self.i, reward, done, {}

    def reset(self):
        self._reset = True
        self.i = self.start
        return self.i


@pytest.mark.parametrize("batch_size", [1, 2, 5, N_CPUS, 10, 24])
def test_ordering_of_env_fns_preserved(batch_size):
    """ Test that the order of the env_fns is also reproduced in the order of
    the observations, and that the actions are sent to the right environments.
    """
    target = 50
    env_fns = [
        partial(DummyEnvironment, start=i, target=target, max_value=100)
        for i in range(batch_size)
    ]
    env = BatchedVectorEnv(env_fns, n_workers=4)
    env.seed(123)
    obs = env.reset()
    assert obs.tolist() == list(range(batch_size))

    obs, reward, done, info = env.step(np.zeros(batch_size))
    assert obs.tolist() == list(range(batch_size))
    # Increment only the 'counters' at even indices.
    actions = [
        int(i % 2 == 0) for i in range(batch_size)
    ]
    obs, reward, done, info = env.step(actions)
    even = np.arange(batch_size) % 2 == 0
    odd = np.arange(batch_size) % 2 == 1
    assert obs[even].tolist() == (np.arange(batch_size) + 1)[even].tolist()
    assert obs[odd].tolist() == np.arange(batch_size)[odd].tolist(), (obs, obs[odd], actions)
    assert reward.tolist() == (np.ones(batch_size) * target - obs).tolist()

    env.close()

def test_done_reset_behaviour():
    """ Test that when one of the envs in the batch is reset, the final
    observation is stored in the "info" dict, at key FINAL_STATE_KEY
    ("final_state" atm.).
    """
    batch_size = 10
    n_workers = 4
    target = batch_size
    starting_values = np.arange(batch_size)
    env_fns = [
        partial(DummyEnvironment, start=start_i, target=target, max_value=target * 2)
        for start_i in starting_values
    ]
    env = BatchedVectorEnv(env_fns, n_workers=n_workers)
    env.seed(123)
    obs = env.reset()
    assert obs.tolist() == list(range(batch_size))

    # Increment all the counters.
    obs, reward, done, info = env.step(np.ones(batch_size))
    # Only the last env (at position batch_size-1) should have 'done=True',
    # since it reached the 'target' value of batch_size + 1 
    last_index = batch_size - 1
    is_last = np.arange(batch_size) == batch_size - 1
    
    assert done[last_index]
    assert all(done == is_last)
    # The observation at the last index should be the new 'starting'
    # observation.
    assert obs[~done].tolist() == (np.arange(batch_size) + 1)[~done].tolist()
    assert obs[done].tolist() == starting_values[done].tolist()

    # The 'info' dict should have the final state as an observation.
    assert info[last_index][FINAL_STATE_KEY] == target
    assert all(FINAL_STATE_KEY not in info_i for info_i in info[:last_index])
    env.close()
