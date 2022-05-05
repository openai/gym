import numpy as np
import pytest

from gym import envs
from gym.error import InvalidAction
from tests.envs.spec_list import SKIP_MUJOCO_WARNING_MESSAGE, skip_mujoco

ENVIRONMENT_IDS = ("HalfCheetah-v2",)


@pytest.mark.skipif(skip_mujoco, reason=SKIP_MUJOCO_WARNING_MESSAGE)
@pytest.mark.parametrize("environment_id", ENVIRONMENT_IDS)
def test_serialize_deserialize(environment_id):
    env = envs.make(environment_id)
    env.reset()

    with pytest.raises(ValueError, match="Action dimension mismatch"):
        env.step([0.1])

    with pytest.raises(ValueError, match="Action dimension mismatch"):
        env.step(0.1)


@pytest.mark.parametrize(
    "environment_id",
    (
        "Acrobot-v1",
        "CartPole-v1",
        "MountainCar-v0",
        "CarRacingDiscrete-v1",
        "LunarLander-v2",
        "Blackjack-v1",
        "CliffWalking-v0",
        "FrozenLake-v1",
        "Taxi-v3",
    ),
)
def test_discrete_actions_out_of_bound(environment_id):
    env = envs.make(environment_id)
    env.reset()

    action_space = env.action_space
    upper_bound = action_space.start + action_space.n - 1

    with pytest.raises(InvalidAction):
        env.step(upper_bound + 1)


@pytest.mark.parametrize(
    "environment_id",
    (
        "MountainCarContinuous-v0",
        "LunarLanderContinuous-v2",
        "CarRacing-v1",
    ),
)
def test_box_actions_out_of_bound(environment_id):
    env = envs.make(environment_id)
    env.reset()

    action_space_shape = env.action_space.shape
    dtype = env.action_space.dtype
    action = np.zeros(action_space_shape, dtype)

    is_bounded = env.action_space.is_bounded()
    upper_bounds = env.action_space.high
    lower_bounds = env.action_space.low

    for i, (is_upper_bound, is_lower_bound) in enumerate(
        zip(env.action_space.bounded_above, env.action_space.bounded_below)
    ):
        if is_upper_bound:
            action[i] = upper_bounds[i] + np.cast[dtype](1)
        elif is_lower_bound:
            action[i] = lower_bounds[i] - np.cast[dtype](1)

    if is_bounded:
        with pytest.warns(UserWarning):
            env.step(action)
