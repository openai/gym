""" Mix of AsyncVectorEnv and SyncVectorEnv, with support for 'chunking' and for
where we have a series of environments on each worker.
"""
import math
import multiprocessing as mp
import itertools
from functools import partial
from typing import Any, Callable, Iterable, List, Sequence, Tuple, TypeVar, Union, Optional

import gym
import numpy as np
from gym import spaces

from .sync_vector_env import SyncVectorEnv
from .async_vector_env import AsyncVectorEnv


T = TypeVar("T")


class BatchedVectorEnv(gym.Env):
    """ Batched vectorized environment.

    Adds the following features, compared to using the vectorized Async and Sync
    VectorEnvs:

    -   Chunking: Running more than one environment per worker. This is done by
        passing `SyncVectorEnv`s as the env_fns to the `AsyncVectorEnv`.

    -   Flexible batch size: Supports any number of environments, irrespective
        of the number of workers or of CPUs. The number of environments will be
        spread out as equally as possible between the workers.
      
        For example, if you want to have a batch_size of 17, and n_workers is 6,
        then the number of environments per worker will be: [3, 3, 3, 3, 3, 2].

        Internally, this works by creating up to two AsyncVectorEnvs, env_a and
        env_b. If the number of envs (batch_size) isn't a multiple of the number
        of workers, then we create the second AsyncVectorEnv (env_b).

        In the first environment (env_a), each env will contain
        ceil(n_envs / n_workers) each. If env_b is needed, then each of its envs
        will contain floor(n_envs / n_workers) environments.

    The observations/actions/rewards are reshaped to be (n_envs, *shape), i.e.
    they don't have an extra 'chunk' dimension.

    NOTE: In order to get this to work, I had to modify the `if done:` statement
    in the worker to be `if done if isinstance(done, bool) else all(done):`.
    """
    def __init__(self,
                 env_fns,
                 n_workers: int = None,
                 **kwargs):
        assert env_fns, "need at least one env_fn."
        self.batch_size: int = len(env_fns)
        
        if n_workers is None:
            n_workers = mp.cpu_count()
        self.n_workers: int = n_workers

        if self.n_workers > self.batch_size:
            self.n_workers = self.batch_size

        # Divide the env_fns as evenly as possible between the workers.
        groups = distribute(env_fns, self.n_workers)

        # Find the first index where the group has a different length.
        self.chunk_a = len(groups[0])
        self.chunk_b = 0
        # First, assume there is no need for another environment, as all the
        # groups have the same length.
        self.start_index_b = self.n_workers
        for i, group in enumerate(groups):
            if len(group) != self.chunk_a:
                self.start_index_b = i
                self.chunk_b = len(group)
                break
        # Total number of envs in each environment.
        self.n_a = sum(map(len, groups[:self.start_index_b]))
        self.n_b = sum(map(len, groups[self.start_index_b:]))

        # Create a SyncVectorEnv per group.
        chunk_env_fns: List[Callable[[], gym.Env]] = [
            partial(SyncVectorEnv, env_fns_group) for env_fns_group in groups
        ]
        env_a_fns = chunk_env_fns[:self.start_index_b]
        env_b_fns = chunk_env_fns[self.start_index_b:]
        
        # Create the AsyncVectorEnvs.
        self.env_a = AsyncVectorEnv(env_fns=env_a_fns, **kwargs)
        self.env_b: Optional[AsyncVectorEnv] = None
        if env_b_fns:
            self.env_b = AsyncVectorEnv(env_fns=env_b_fns, **kwargs)

        self.observation_space: gym.Space  # type: ignore
        self.action_space: gym.Space  # type: ignore

        # Unbatch & join the observations/actions spaces.
        flat_obs_a = remove_extra_batch_dim(self.env_a.observation_space)  # type: ignore
        flat_act_a = remove_extra_batch_dim(self.env_a.action_space)  # type: ignore
        if self.env_b:
            flat_obs_b = remove_extra_batch_dim(self.env_b.observation_space)  # type: ignore
            flat_act_b = remove_extra_batch_dim(self.env_b.action_space)  # type: ignore
            
            self.observation_space = concat_spaces(flat_obs_a, flat_obs_b)
            self.action_space = concat_spaces(flat_act_a, flat_act_b)
        else:
            self.observation_space = flat_obs_a
            self.action_space = flat_act_a

        self.reward_range = self.env_a.reward_range

    def reset(self):
        obs_a = self.env_a.reset()
        if self.env_b:
            obs_b = self.env_b.reset()
            return self.unchunk(obs_a, obs_b)
        return self.unchunk(obs_a)

    def step(self, action: Sequence):
        if self.env_b:
            flat_actions_a, flat_actions_b = action[:self.n_a], action[self.n_a:]
            actions_a = self.chunk(flat_actions_a, self.chunk_a)
            actions_b = self.chunk(flat_actions_b, self.chunk_b)

            obs_a, rew_a, done_a, info_a = self.env_a.step(actions_a)
            obs_b, rew_b, done_b, info_b = self.env_b.step(actions_b)

            observations = self.unchunk(obs_a, obs_b)
            rewards = self.unchunk(rew_a, rew_b)
            done = self.unchunk(done_a, done_b)
            info = self.unchunk(info_a, info_b)
        else:
            action = self.chunk(action, self.chunk_a)
            observations, rewards, done, info = self.env_a.step(action)

            observations = self.unchunk(observations)
            rewards = self.unchunk(rewards)
            done = self.unchunk(done)
            info = self.unchunk(info)
        return observations, rewards, done, info

    def seed(self, seeds: Union[Optional[int], Sequence[Optional[int]]] = None):
        if seeds is None:
            seeds = [None for _ in range(self.batch_size)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.batch_size)]
        assert len(seeds) == self.batch_size

        seeds_a = self.chunk(seeds[:self.n_a], self.chunk_a)
        seeds_b = self.chunk(seeds[self.n_a:], self.chunk_b)
        self.env_a.seed(seeds_a)
        if self.env_b:
            self.env_b.seed(seeds_b)       

    def close(self):
        self.env_a.close()
        if self.env_b:
            self.env_b.close()

    @staticmethod
    def unchunk(*values: Sequence[Sequence[T]]) -> Sequence[T]:
        """ Combine 'chunked' results coming from the envs into a single
        batch.
        """
        all_values: List[T] = []
        for sequence in values:
            all_values.extend(itertools.chain.from_iterable(sequence))
        if isinstance(values[0], np.ndarray):
            return np.array(all_values)
        return all_values

    @staticmethod
    def chunk(values: Sequence[T], chunk_length: int) -> Sequence[Sequence[T]]:
        """ Add the 'chunk'/second batch dimension to the list of items. """
        groups = list(n_consecutive(values, chunk_length))
        if isinstance(values, np.ndarray):
            groups = np.array(groups)
        return groups


def distribute(values: Sequence[T], n_groups: int) -> List[Sequence[T]]:
    """ Distribute the values 'values' as evenly as possible into n_groups.

    >>> distribute(list(range(14)), 5)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13]]
    >>> distribute(list(range(9)), 4)
    [[0, 1, 2], [3, 4], [5, 6], [7, 8]]
    >>> import numpy as np
    >>> distribute(np.arange(9), 4)
    [array([0, 1, 2]), array([3, 4]), array([5, 6]), array([7, 8])]
    """
    n_values = len(values)
    # Determine the final lengths of each group.
    min_values_per_group = math.floor(n_values / n_groups)
    max_values_per_group = math.ceil(n_values / n_groups)
    remainder = n_values % n_groups
    group_lengths = [
        max_values_per_group if i < remainder else min_values_per_group
        for i in range(n_groups)
    ]
    # Equivalent, but maybe a tiny bit slower:
    # group_lengths: List[int] = [0 for _ in range(n_groups)]
    # for i in range(len(values)):
    #     group_lengths[i % n_groups] += 1
    groups: List[Sequence[T]] = [[] for _ in range(n_groups)]

    start_index = 0
    for i, group_length in enumerate(group_lengths):
        end_index = start_index + group_length
        groups[i] = values[start_index:end_index]
        start_index += group_length
    return groups


def n_consecutive(items: Iterable[T], n: int=2, yield_last_batch=True) -> Iterable[Tuple[T, ...]]:
    """Collect data into chunks of up to `n` elements.
    Adapted from the itertools recipes.

    When `yield_last_batch` is True, the final chunk (which might have fewer
    than `n` items) will also be yielded.
    
    >>> list(n_consecutive("ABCDEFG", 3))
    [["A", "B", "C"], ["D", "E", "F"], ["G"]]
    """
    values: List[T] = []
    for item in items:
        values.append(item)
        if len(values) == n:
            yield tuple(values)
            values.clear()
    if values and yield_last_batch:
        yield tuple(values)


def concat_spaces(space_a: gym.Space, space_b: gym.Space) -> gym.Space:
    """ Concatenate two gym spaces of the same types. """
    if type(space_a) != type(space_b):
        raise RuntimeError(f"Can only concat spaces of the same type: {space_a} {space_b}")
    
    if isinstance(space_a, spaces.Box):
        assert isinstance(space_b, spaces.Box)

        assert space_a.shape[1:] == space_b.shape[1:]
        new_low = np.concatenate([space_a.low, space_b.low])
        new_high = np.concatenate([space_a.high, space_b.high])
        return spaces.Box(low=new_low, high=new_high, dtype=space_a.dtype)

    if isinstance(space_a, spaces.Tuple):
        assert isinstance(space_b, spaces.Tuple)

        new_spaces = space_a.spaces + space_b.spaces
        return spaces.Tuple(new_spaces)
    
    if isinstance(space_a, spaces.Dict):
        assert isinstance(space_b, spaces.Dict)
        new_spaces = {}
        for key in space_a.spaces:
            value_a = space_a[key]
            value_b = space_b[key]
            new_spaces[key] = concat_spaces(value_a, value_b)
        return spaces.Dict(new_spaces)
    
    raise NotImplementedError(f"Unsupported spaces {space_a} {space_b}")


def remove_extra_batch_dim(space: gym.Space):
    if isinstance(space, spaces.Box):
        dims = space.shape
        assert len(space.shape) >= 2, space
        new_shape = tuple([dims[0] * dims[1], *dims[2:]])
        new_low = space.low.reshape(new_shape)
        new_high = space.high.reshape(new_shape)
        return spaces.Box(low=new_low, high=new_high, dtype=space.dtype)

    if isinstance(space, spaces.Tuple):
        assert len(space.spaces) and isinstance(space.spaces[0], spaces.Tuple)
        all_spaces: List[gym.Space] = []
        for sub_space in space.spaces:
            all_spaces.extend(sub_space.spaces)
        return spaces.Tuple(all_spaces)

    if isinstance(space, spaces.Dict):
        return spaces.Dict({
            k: remove_extra_batch_dim(sub_space)
            for k, sub_space in space.spaces.items()
        })

    raise NotImplementedError(f"Unsupported space {space}")



if __name__ == "__main__":
    import doctest
    doctest.testmod()
