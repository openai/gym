import numpy as np
import pytest

from gym.spaces import Box, Dict, Discrete, Tuple
from tests.dev_wrappers.mock_data import DISCRETE_ACTION
from tests.dev_wrappers.utils import TestingEnv

try:
    from gym.wrappers import filter_observations_v0
except ImportError:
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (
            TestingEnv(observation_space=Dict(obs=Box(-1, 1, (1,)), time=Discrete(3))),
            {"obs": True, "time": False},
        ),
    ],
)
def test_dict_filter_observation_v0(env, args):
    """Test correct filtering of `Dict` observation space."""
    wrapped_env = filter_observations_v0(env, args)

    assert wrapped_env.observation_space.get("obs", False)
    assert not wrapped_env.observation_space.get("time", False)

    obs, _, _, _ = wrapped_env.step(0)

    assert obs.get("obs", False)
    assert not obs.get("time", False)


@pytest.mark.parametrize(
    ("env", "args", "filtered_space_size"),
    [
        (
            TestingEnv(
                observation_space=Tuple([Box(-1, 1, ()), Box(-2, 2, ()), Discrete(3)])
            ),
            [True, False, True],
            2,
        ),
    ],
)
def test_tuple_filter_observation_v0(env, args, filtered_space_size):
    """Test correct filtering of `Tuple` observation space."""
    wrapped_env = filter_observations_v0(env, args)

    assert len(wrapped_env.observation_space) == filtered_space_size

    obs, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    assert len(obs) == filtered_space_size

    assert isinstance(obs[0], np.ndarray)
    assert isinstance(obs[1], int)


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (
            TestingEnv(
                observation_space=Dict(
                    x=Tuple([Discrete(2), Box(-1, 1, (1,))]),
                    y=Dict(box=Box(-1, 1, (1,)), box2=Box(1, 1, (1,))),
                )
            ),
            {"x": [True, False], "y": {"box": True, "box2": False}},
        ),
    ],
)
def test_nested_filter_observation_v0(env, args):
    """Test correct filtering of `Tuple` observation space."""
    wrapped_env = filter_observations_v0(env, args)
    obs, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    assert len(obs["x"]) == 1
    assert "box" in obs["y"]
    assert "box2" not in obs["y"]
