from gym.spaces.box import Box
from gym.spaces.dict import Dict
from gym.spaces.discrete import Discrete
from gym.spaces.multi_binary import MultiBinary
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.space import Space
from gym.spaces.tuple import Tuple
from gym.spaces.utils import flatdim, flatten, flatten_space, unflatten

__all__ = [
    "Space",
    "Box",
    "Discrete",
    "MultiDiscrete",
    "MultiBinary",
    "Tuple",
    "Dict",
    "flatdim",
    "flatten_space",
    "flatten",
    "unflatten",
]
