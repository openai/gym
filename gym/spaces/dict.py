from collections import OrderedDict
import numpy as np
from .space import Space


class Dict(Space):
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

    def __init__(self, spaces=None, seed=None, **spaces_kwargs):
        assert (spaces is None) or (
            not spaces_kwargs
        ), "Use either Dict(spaces=dict(...)) or Dict(foo=x, bar=z)"

        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, dict) and not isinstance(spaces, OrderedDict):
            spaces = OrderedDict(sorted(list(spaces.items())))
        if isinstance(spaces, list):
            spaces = OrderedDict(spaces)
        self.spaces = spaces
        for space in spaces.values():
            assert isinstance(
                space, Space
            ), "Values of the dict should be instances of gym.Space"
        super(Dict, self).__init__(
            None, None, seed
        )  # None for shape and dtype, since it'll require special handling

    def seed(self, seed=None):
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

    def sample(self):
        return OrderedDict([(k, space.sample()) for k, space in self.spaces.items()])

    def contains(self, x):
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
        for key in self.spaces:
            yield key

    def __len__(self):
        return len(self.spaces)

    def __contains__(self, item):
        return self.contains(item)

    def __repr__(self):
        return (
            "Dict("
            + ", ".join([str(k) + ":" + str(s) for k, s in self.spaces.items()])
            + ")"
        )

    def to_jsonable(self, sample_n):
        # serialize as dict-repr of vectors
        return {
            key: space.to_jsonable([sample[key] for sample in sample_n])
            for key, space in self.spaces.items()
        }

    def from_jsonable(self, sample_n):
        dict_of_list = {}
        for key, space in self.spaces.items():
            dict_of_list[key] = space.from_jsonable(sample_n[key])
        ret = []
        for i, _ in enumerate(dict_of_list[key]):
            entry = {}
            for key, value in dict_of_list.items():
                entry[key] = value[i]
            ret.append(entry)
        return ret

    def __eq__(self, other):
        return isinstance(other, Dict) and self.spaces == other.spaces

    def keys(self):
        return self.spaces.keys()

    def values(self):
        return self.spaces.values()

    def items(self):
        return self.spaces.items()
