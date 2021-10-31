import pickle

import pytest

from gym import envs
from tests.envs.spec_list import skip_mujoco, SKIP_MUJOCO_WARNING_MESSAGE


ENVIRONMENT_IDS = (
    "FetchPickAndPlace-v1",
    "FetchPush-v1",
    "FetchReach-v1",
    "FetchSlide-v1",
    "HandManipulatePen-v0",
    "HandManipulateBlock-v0",
    "HandManipulateEgg-v0",
    "HandReach-v0",
    "HandManipulateEggTouchSensors-v0",
    "HandManipulateEggTouchSensors-v1",
    "HandManipulatePenTouchSensors-v0",
    "HandManipulatePenTouchSensors-v1",
    "HandManipulateBlockTouchSensors-v0",
    "HandManipulateBlockTouchSensors-v1",
    "Hopper-v2",
    "Walker2d-v2",
    "Humanoid-v2",
    "HumanoidStandup-v2",
    "HalfCheetah-v2",
    "Swimmer-v2",
    "Ant-v2",
    "Hopper-v3",
    "Walker2d-v3",
    "Humanoid-v3",
    "HalfCheetah-v3",
    "Swimmer-v3",
    "Ant-v3",
    "Reacher-v2",
    "Pusher-v2",
    "Thrower-v2",
    "Striker-v2",
)


# @pytest.mark.skipif(skip_mujoco, reason=SKIP_MUJOCO_WARNING_MESSAGE)
@pytest.mark.parametrize("environment_id", ENVIRONMENT_IDS)
def test_serialize_deserialize(environment_id):
    env = envs.make(environment_id)
    print(
        "Testing {}".format(
            environment_id,
        )
    )
    if env.action_space.shape[-1] <= 1:
        return
    env.reset()
    obs = 0
    try:
        obs = env.step(0.1)
    except:
        pass
    try:
        obs = env.step([0.1])
    except:
        pass

    assert not obs, "env {}: scalar input dimension does not check".format(
        environment_id
    )
