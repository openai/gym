from typing import List

import numpy as np

from gym.spaces import (
    Box,
    Dict,
    Discrete,
    Graph,
    MultiBinary,
    MultiDiscrete,
    Space,
    Text,
    Tuple,
)

TESTING_FUNDAMENTAL_SPACES = [
    Discrete(3),
    Discrete(3, start=-1),
    Box(low=0.0, high=1.0),
    Box(low=0.0, high=np.inf, shape=(2, 2)),
    Box(low=np.array([-10.0, 0.0]), high=np.array([10.0, 10.0]), dtype=np.float64),
    Box(low=-np.inf, high=0.0, shape=(2, 1)),
    Box(low=0.0, high=np.inf, shape=(2, 1)),
    MultiDiscrete([2, 2]),
    MultiDiscrete([[2, 3], [3, 2]]),
    MultiBinary(8),
    MultiBinary([2, 3]),
    Text(10),
    Text(min_length=5, max_length=10),
    Text(10, charset="abcdef"),
]

TESTING_COMPOSITE_SPACES = [
    Tuple([Discrete(5), Discrete(10)]),
    Tuple(
        (
            Discrete(5),
            Box(
                low=np.array([0.0, 0.0]),
                high=np.array([1.0, 5.0]),
                dtype=np.float64,
            ),
        )
    ),
    Tuple((Discrete(5), Tuple((Box(low=0.0, high=1.0, shape=(3,)), Discrete(2))))),
    Tuple((Discrete(3), Dict(position=Box(low=0.0, high=1.0), velocity=Discrete(2)))),
    Dict(
        {
            "position": Discrete(5),
            "velocity": Box(
                low=np.array([0.0, 0.0]),
                high=np.array([1.0, 5.0]),
                dtype=np.float64,
            ),
        }
    ),
    Dict(
        {
            "a": Box(low=0, high=1, shape=(3, 3)),
            "b": Dict(
                {
                    "b_1": Box(low=-100, high=100, shape=(2,)),
                    "b_2": Box(low=-1, high=1, shape=(2,)),
                }
            ),
            "c": Discrete(4),
        }
    ),
    Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=Discrete(5)),
    Graph(node_space=Discrete(5), edge_space=Box(low=-100, high=100, shape=(3, 4))),
    Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=None),
]

TESTING_SPACES: List[Space] = TESTING_FUNDAMENTAL_SPACES + TESTING_COMPOSITE_SPACES
TESTING_SPACES_IDS = [f"{space}" for space in TESTING_SPACES]
