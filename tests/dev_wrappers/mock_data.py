"""Reusable mock data for easier testing."""
import numpy as np

from gym.spaces import Box, Dict, Discrete, Tuple

SEED = 1

NUM_STEPS = 5
NUM_ENVS = 3

# Discrete data
DISCRETE_SPACE = Discrete(1)
DISCRETE_ACTION = 0

# Box data
BOX_LOW, BOX_HIGH, BOX_DIM = -5, 5, 1
BOX_SPACE = Box(BOX_LOW, BOX_HIGH, (BOX_DIM,), dtype=np.float64)

NEW_BOX_LOW, NEW_BOX_HIGH = 0, 2
NEW_BOX_DIM = (1, 1)
NEW_BOX_DIM_IMPOSSIBLE = (3,)

# Dict data
DICT_SPACE = Dict(discrete=DISCRETE_SPACE, box=BOX_SPACE, box2=BOX_SPACE)
FLATTENEND_DICT_SIZE = 3  # 1 discrete + 1 Box + 1 Box

NESTED_DICT_SPACE = Dict(
    discrete=DISCRETE_SPACE,
    box=BOX_SPACE,
    nested=Dict(nested=BOX_SPACE),
)

DOUBLY_NESTED_DICT_SPACE = Dict(
    discrete=DISCRETE_SPACE,
    box=BOX_SPACE,
    nested=Dict(nested=Dict(nested=BOX_SPACE)),
)

# Tuple data
TUPLE_SPACE = Tuple([DISCRETE_SPACE, BOX_SPACE])
TWO_BOX_TUPLE_SPACE = Tuple([BOX_SPACE, BOX_SPACE])

NESTED_TUPLE_SPACE = Tuple([BOX_SPACE, Tuple([DISCRETE_SPACE, BOX_SPACE])])

DOUBLY_NESTED_TUPLE_SPACE = Tuple(
    [
        BOX_SPACE,
        Tuple(
            [
                DISCRETE_SPACE,
                Tuple([DISCRETE_SPACE, BOX_SPACE]),
            ]
        ),
    ]
)

# Mix Tuple/Dict data
TUPLE_WITHIN_DICT_SPACE = Dict(discrete=DISCRETE_SPACE, tuple=Tuple([BOX_SPACE]))
DICT_WITHIN_TUPLE_SPACE = Tuple([DISCRETE_SPACE, Dict(dict=BOX_SPACE)])
