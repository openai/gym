""" Mix of AsyncVectorEnv and SyncVectorEnv, with support for 'chunking' and for
where we have a series of environments on each worker.
"""
import itertools
import math
import multiprocessing as mp
from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, TypeVar, Union)
import numpy as np


from gym import spaces, Env, Space
from gym.spaces.utils import flatten, unflatten
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.utils import batch_space, concatenate, create_empty_array
from gym.vector.vector_env import VectorEnv

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
M = TypeVar("M")


class BatchedVectorEnv(VectorEnv):
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

    -   When some environments have `done=True` while stepping, those
        environments are reset, as was done previously. Additionally, the final
        observation for those environments is placed in the info dict at key
        FINAL_STATE_KEY (currently 'final_state').
    """
    def __init__(self,
                 env_fns,
                 n_workers: int = None,
                 **kwargs):
        assert env_fns, "need at least one env_fn."
        self.batch_size: int = len(env_fns)

        # Use one of the env_fns to get the observation/action space.
        with env_fns[0]() as temp_env:
            single_observation_space = temp_env.observation_space
            single_action_space = temp_env.action_space
            self.reward_range = temp_env.reward_range
        del temp_env

        super().__init__(
            num_envs=self.batch_size,
            observation_space=single_observation_space,
            action_space=single_action_space,
        )

        if n_workers is None:
            n_workers = mp.cpu_count()
        self.n_workers: int = n_workers

        if self.n_workers > self.batch_size:
            self.n_workers = self.batch_size

        # Divide the env_fns as evenly as possible between the workers.
        groups = distribute(env_fns, self.n_workers)

        # Find the first index where the group has a different length.
        self.chunk_length_a = len(groups[0])
        self.chunk_length_b = 0

        # First, assume there is no need for another environment (all the
        # groups have the same length).
        self.start_index_b = self.n_workers
        for i, group in enumerate(groups):
            if len(group) != self.chunk_length_a:
                self.start_index_b = i
                self.chunk_length_b = len(group)
                break

        # Total number of envs in each environment.
        self.n_a = sum(map(len, groups[:self.start_index_b]))
        self.n_b = sum(map(len, groups[self.start_index_b:]))

        # Create a SyncVectorEnv per group.
        chunk_env_fns: List[Callable[[], Env]] = [
            partial(SyncVectorEnv, env_fns_group) for env_fns_group in groups
        ]
        env_a_fns = chunk_env_fns[:self.start_index_b]
        env_b_fns = chunk_env_fns[self.start_index_b:]
        # Create the AsyncVectorEnvs.
        self.env_a = AsyncVectorEnv(env_fns=env_a_fns, **kwargs)
        self.env_b: Optional[AsyncVectorEnv] = None
        if env_b_fns:
            self.env_b = AsyncVectorEnv(env_fns=env_b_fns, **kwargs)

        # Unbatch & join the observations/actions spaces.        

    def reset_async(self):
        self.env_a.reset_async()
        if self.env_b:
            self.env_b.reset_async()

    def reset_wait(self, timeout=None, **kwargs):
        obs_a = self.env_a.reset_wait(timeout=timeout)
        obs_a = unroll(obs_a, item_space=self.single_observation_space)
        obs_b = []
        if self.env_b:
            obs_b = self.env_b.reset_wait(timeout=timeout)
            obs_b = unroll(obs_b, item_space=self.single_observation_space)
        observations = fuse_and_batch(self.single_observation_space, obs_a, obs_b, n_items = self.n_a + self.n_b)
        return observations

    def step_async(self, action: Sequence) -> None:
        if self.env_b:
            flat_actions_a, flat_actions_b = action[:self.n_a], action[self.n_a:]
            actions_a = chunk(flat_actions_a, self.chunk_length_a)
            actions_b = chunk(flat_actions_b, self.chunk_length_b)
            self.env_a.step_async(actions_a)
            self.env_b.step_async(actions_b)

        else:
            action = chunk(action, self.chunk_length_a)
            self.env_a.step_async(action)

    def step_wait(self, timeout: Union[int, float] = None, **kwargs):
        obs_a, rew_a, done_a, info_a = self.env_a.step_wait(timeout)
        obs_a = unroll(obs_a, item_space=self.single_observation_space)
        rew_a = unroll(rew_a)
        done_a = unroll(done_a)
        info_a = unroll(info_a)
        obs_b = []
        rew_b = []
        done_b = []
        info_b = []
        if self.env_b:
            obs_b, rew_b, done_b, info_b = self.env_b.step_wait(timeout)
            obs_b = unroll(obs_b, item_space=self.single_observation_space)
            rew_b = unroll(rew_b)
            done_b = unroll(done_b)
            info_b = unroll(info_b)
        observations = fuse_and_batch(self.single_observation_space, obs_a, obs_b, n_items = self.n_a + self.n_b)
        rewards = np.array(rew_a + rew_b)
        done = np.array(done_a + done_b)
        # TODO: Should we batch the info dict? or just give back the list of
        # 'info' dicts for each env, like so?
        info = info_a + info_b
        return observations, rewards, done, info

    def seed(self, seeds: Union[int, Sequence[Optional[int]]] = None):
        if seeds is None:
            seeds = [None for _ in range(self.batch_size)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.batch_size)]
        assert len(seeds) == self.batch_size

        seeds_a = chunk(seeds[:self.n_a], self.chunk_length_a)
        seeds_b = chunk(seeds[self.n_a:], self.chunk_length_b)
        self.env_a.seed(seeds_a)
        if self.env_b:
            self.env_b.seed(seeds_b)       

    def close_extras(self, **kwargs):
        self.env_a.close_extras(**kwargs)
        if self.env_b:
            self.env_b.close_extras(**kwargs)

    def render(self, mode: str = "rgb_array"):
        chunked_images_a = self.env_a.render(mode="rgb_array")
        images_a: List[np.ndarray] = unroll(chunked_images_a)
        images_b: List[np.ndarray] = []
        
        if self.env_b:
            chunked_images_b = self.env_b.render(mode="rgb_array")
            images_b = unroll(chunked_images_b)
        
        image_batch = np.stack(images_a + images_b)
        
        if mode == "rgb_array":
            return image_batch
        
        raise NotImplementedError(f"Render mode {mode} isn't implemented yet.")

        # NOTE: This is only here for illustration purposes.
        if mode == "human":
            # See PR #1624 for the tile_images function.
            tiled_version = tile_images(image_batch)
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(tiled_version)
            return self.viewer.isopen


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



def chunk(values: Sequence[T], chunk_length: int) -> Sequence[Sequence[T]]:
    """ Add the 'chunk'/second batch dimension to the list of items.
    
    NOTE: I don't think this would work with tuples as inputs, but it hasn't
    been a problem yet because the action/reward spaces haven't been tuples yet.
    """
    groups = list(n_consecutive(values, chunk_length))
    if isinstance(values, np.ndarray):
        groups = np.array(groups)
    return groups


def unroll(chunks: Sequence[Sequence[T]], item_space: Space = None) -> List[T]:
    """ Unroll the given chunks, to get a list of individual items.

    This is the inverse operation of 'chunk' above.
    """
    if isinstance(item_space, spaces.Tuple):
        # 'flatten out' the chunks for each index. The returned value will be a
        # tuple of lists of samples. 
        chunked_items = list(zip(chunks))
        return tuple([
            unroll(chunk_items_i, item_space=item_space.spaces[i])
            for i, chunk_items_i in enumerate(chunked_items)  
        ])

    if isinstance(chunks, np.ndarray):
        # Remove the 'chunk' dimension.
        return list(chunks.reshape([-1, *chunks.shape[2:]]))
    
    values: List[T] = []
    for chunk in chunks:
        values.extend(chunk)
    return values

from functools import singledispatch
from gym.vector.utils.spaces import _BaseGymSpaces


@singledispatch
def fuse_and_batch(item_space: spaces.Space, *sequences: Sequence[Sequence[T]], n_items: int) -> Sequence[T]:
    """Concatenate two sequences of items, and then fuse them into a single
    batch.
    """
    assert all(isinstance(sequence, list) for sequence in sequences)
    out = create_empty_array(item_space, n=n_items)
    # # Concatenate the (two) batches into a single batch of samples.
    items_batch = np.concatenate([
        np.asarray(v).reshape([-1, *item_space.shape])
        for v in itertools.chain(*sequences)
    ])
    # # Split this batch of samples into a list of items from each space.
    items = [
        v.reshape(item_space.shape) for v in np.split(items_batch, n_items)
    ]
    # TODO: Need to add more tests to make sure this works with custom spaces and Dict spaces.
    return concatenate(items, out, item_space)



@fuse_and_batch.register
def fuse_and_batch_dicts(item_space: spaces.Dict, *sequences: Sequence[Dict[K, V]], n_items: int) -> Dict[K, Sequence[T]]:
    values: Dict[K, List[V]] = {
        k: [] for k in item_space.spaces.keys()
    }
    for sequence in sequences:
        for item in sequence:
            for k, v in item.items():
                values[k].append(v)
    return {
        k: fuse_and_batch(item_space.spaces[k], values[k], n_items=n_items)
        for k in values
    }


@fuse_and_batch.register
def fuse_and_batch_tuples(item_space: spaces.Tuple, *sequences: Sequence[Tuple[T, ...]], n_items: int) -> Tuple[Sequence[T], ...]:
    # Join the list of items for each subspace of the item_space Tuple.
    joined_sequences: Sequence[List[T]] = [
        sum(items, []) for items in itertools.zip_longest(*sequences, fillvalue=[])
    ]
    return tuple(
        fuse_and_batch(space, sequence, n_items=n_items)
        for space, sequence in zip(item_space.spaces, joined_sequences)
        # np.concatenate(sequence) for sequence in joined_sequences
    )


def n_consecutive(items: Iterable[T], n: int=2, yield_last_batch=True) -> Iterable[Tuple[T, ...]]:
    """Collect data into chunks of up to `n` elements.
    
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


def zip_dicts(*dicts: Dict[K, V],
               missing: M = None) -> Iterable[Tuple[K, Tuple[Union[M, V], ...]]]:
    """Iterator over the union of all keys, giving the value from each dict if
    present, else `missing`.
    """
    # If any attributes are common to both the Experiment and the State,
    # copy them over to the Experiment.
    keys = set(itertools.chain(*dicts))
    for key in keys:
        yield (key, tuple(d.get(key, missing) for d in dicts))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
