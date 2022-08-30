import pytest

import gym
from tests.dev_wrappers.mock_data import DISCRETE_ACTION, NUM_ENVS

try:
    from gym.wrappers import GrayscaleObservationsV0
except ImportError:
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize(
    ("env"),
    [gym.make("CarRacing-v2", continuous=False, disable_env_checker=True)],
)
def test_grayscale_observation_v0(env):
    """Test correct transformation of observation in grayscale."""
    wrapped_env = GrayscaleObservationsV0(env)
    wrapped_env.reset()

    obs, _, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    assert len(obs.shape) == 2  # height and width. No more color dim


@pytest.mark.parametrize(
    ("env"),
    [
        gym.vector.make(
            "CarRacing-v2",
            continuous=False,
            num_envs=NUM_ENVS,
            disable_env_checker=True,
        )
    ],
)
def test_grayscale_observation_v0_vectorenv(env):
    """Test correct transformation of observation in grayscale."""
    wrapped_env = GrayscaleObservationsV0(env)
    wrapped_env.reset()

    obs, _, _, _, _ = wrapped_env.step([DISCRETE_ACTION] * NUM_ENVS)

    assert len(obs.shape) == 3  # height and width. No more color dim
    assert obs.shape[0] == NUM_ENVS
