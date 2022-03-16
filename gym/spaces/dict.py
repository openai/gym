from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Dict as TypingDict
import numpy as np
from .space import Space


class Dict(Space[TypingDict[str, Space]], Mapping):
    """
    A dictionary of simpler spaces.

    Example usage:
    self.observation_space = spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})

    Example usage [nested]:
    self.nested_observation_space = spaces.Dict({
        'sensors':  spaces.Dict({
            'position': spaces.Box(low=-100, high=100, shape=(3,)),
            'velocity': spaces.Box(low=-1, high=1, shape=(3,)),
            'front_cam': spaces.Tuple((
                spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                spaces.Box(low=0, high=1, shape=(10, 10, 3))
            )),
            'rear_cam': spaces.Box(low=0, high=1, shape=(10, 10, 3)),
        }),
        'ext_controller': spaces.MultiDiscrete((5, 2, 2)),
        'inner_state':spaces.Dict({
            'charge': spaces.Discrete(100),
            'system_checks': spaces.MultiBinary(10),
            'job_status': spaces.Dict({
                'task': spaces.Discrete(5),
                'progress': spaces.Box(low=0, high=100, shape=()),
            })
        })
    })
    """

    def __init__(
        self,
        spaces: dict[str, Space] | None = None,
        seed: dict | int | None = None,
        **spaces_kwargs: Space,
    ):
        assert (spaces is None) or (
            not spaces_kwargs
        ), "Use either Dict(spaces=dict(...)) or Dict(foo=x, bar=z)"

        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, dict) and not isinstance(spaces, OrderedDict):
            try:
                spaces = OrderedDict(sorted(spaces.items()))
            except TypeError:  # raise when sort by different types of keys
                spaces = OrderedDict(spaces.items())
        if isinstance(spaces, Sequence):
            spaces = OrderedDict(spaces)

        assert isinstance(spaces, OrderedDict), "spaces must be a dictionary"

        self.spaces = spaces
        for space in spaces.values():
            assert isinstance(
                space, Space
            ), "Values of the dict should be instances of gym.Space"
        super().__init__(
            None, None, seed  # type: ignore
        )  # None for shape and dtype, since it'll require special handling

    def seed(self, seed: dict | int | None = None) -> list:
        seeds = []
        if isinstance(seed, dict):
            for key, seed_key in zip(self.spaces, seed):
                assert key == seed_key, print(
                    "Key value",
                    seed_key,
                    "in passed seed dict did not match key value",
                    key,
                    "in spaces Dict.",
                )
                seeds += self.spaces[key].seed(seed[seed_key])
        elif isinstance(seed, int):
            seeds = super().seed(seed)
            try:
                subseeds = self.np_random.choice(
                    np.iinfo(int).max,
                    size=len(self.spaces),
                    replace=False,  # unique subseed for each subspace
                )
            except ValueError:
                subseeds = self.np_random.choice(
                    np.iinfo(int).max,
                    size=len(self.spaces),
                    replace=True,  # we get more than INT_MAX subspaces
                )

            for subspace, subseed in zip(self.spaces.values(), subseeds):
                seeds.append(subspace.seed(int(subseed))[0])
        elif seed is None:
            for space in self.spaces.values():
                seeds += space.seed(seed)
        else:
            raise TypeError("Passed seed not of an expected type: dict or int or None")

        return seeds

    def sample(self) -> dict:
        return OrderedDict([(k, space.sample()) for k, space in self.spaces.items()])

    def contains(self, x) -> bool:
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def __getitem__(self, key):
        return self.spaces[key]

    def __setitem__(self, key, value):
        self.spaces[key] = value

    def __iter__(self):
        yield from self.spaces

    def __len__(self) -> int:
        return len(self.spaces)

    def __repr__(self) -> str:
        return (
            "Dict("
            + ", ".join([str(k) + ":" + str(s) for k, s in self.spaces.items()])
            + ")"
        )

    def to_jsonable(self, sample_n: list) -> dict:
        # serialize as dict-repr of vectors
        return {
            key: space.to_jsonable([sample[key] for sample in sample_n])
            for key, space in self.spaces.items()
        }

    def from_jsonable(self, sample_n: dict[str, list]) -> list:
        dict_of_list: dict[str, list] = {}
        for key, space in self.spaces.items():
            dict_of_list[key] = space.from_jsonable(sample_n[key])
        ret = []
        n_elements = len(next(iter(dict_of_list.values())))
        for i in range(n_elements):
            entry = {}
            for key, value in dict_of_list.items():
                entry[key] = value[i]
            ret.append(entry)
        return ret
