from functools import partial
from multiprocessing import cpu_count
from typing import Optional

import gym
import numpy as np
import pytest
from gym import spaces
from gym.vector.batched_vector_env import BatchedVectorEnv
from gym.vector.utils import batch_space

N_CPUS = cpu_count()

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


class TupleObservationsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, second_space: gym.Space):
        super().__init__(env)
        self.observation_space: gym.Space = spaces.Tuple([
            env.observation_space,
            second_space,
        ])
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return (observation, self.observation_space[1].sample()), reward, done, info

    def reset(self):
        observation = self.env.reset()
        return (observation, self.observation_space[1].sample())


@pytest.mark.parametrize("batch_size", [1, 5, 11, 24])
@pytest.mark.parametrize("n_workers", [1, 3, None])
@pytest.mark.parametrize("base_env", ["CartPole-v0", "Pendulum-v0"])
def test_space_with_tuple_observations(batch_size: int, n_workers: Optional[int], base_env: str):

    def make_env():
        env = gym.make(base_env)
        env = TupleObservationsWrapper(env, spaces.Discrete(1))
        return env
    
    env_fn = make_env
    env_fns = [env_fn for _ in range(batch_size)]
    env = BatchedVectorEnv(env_fns, n_workers=n_workers)
    env.seed(123)
    
    single_env = make_env()
    assert env.single_observation_space == single_env.observation_space
    assert env.single_action_space == single_env.action_space
    assert env.observation_space == batch_space(single_env.observation_space, batch_size)
    # NOTE: VectorEnvs currently have a tuple of actions as the action space,
    # rather than using the result of the batch_space function.
    # assert env.action_space == batch_space(single_env.action_space, batch_size)
    
    single_env_obs = single_env.reset()
    obs = env.reset()
    assert isinstance(obs, tuple)
    assert len(obs) == len(single_env_obs)
    
    for i, (obs_item, single_obs_item) in enumerate(zip(obs, single_env_obs)):
        assert obs_item.shape == (batch_size, *np.asanyarray(single_obs_item).shape)

    assert obs in env.observation_space
    assert single_env_obs in env.single_observation_space
    
    single_action = single_env.action_space.sample()
    assert single_action in env.single_action_space
    actions = env.action_space.sample()
    step_obs, rewards, done, info = env.step(actions)
    assert step_obs in env.observation_space
    
    assert len(rewards) == batch_size
    assert len(done) == batch_size
    assert all([isinstance(v, bool) for v in done.tolist()]), [type(v) for v in done]
    assert len(info) == batch_size


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


import gym
import numpy as np

from collections import OrderedDict
from gym.spaces import Dict, Box, Discrete
from gym.vector import BatchedVectorEnv

class DictEnv(gym.Env):
    def __init__(self):
        super(DictEnv, self).__init__()
        self.observation_space = Dict(OrderedDict([
            ('position', Box(-2., 2., shape=(2,), dtype=np.float32)),
            ('velocity', Box(0., 1., shape=(2,), dtype=np.float32))
        ]))
        self.action_space = Discrete(2)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        observation = self.observation_space.sample()
        reward, done = 0., False
        return (observation, reward, done, {})


@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("n_workers", [1, 4])
def test_dict_observations(batch_size: int, n_workers: int):
    def make_env(seed):
        def _make_env():
            return DictEnv()
        return _make_env
    
    single_env = make_env(0)()
    
    env = BatchedVectorEnv([make_env(i) for i in range(batch_size)], n_workers=4)
    
    assert env.observation_space == batch_space(single_env.observation_space, batch_size)
    observations = env.reset()

    assert observations["position"].shape == (batch_size, 2)
    assert observations["velocity"].shape == (batch_size, 2)
