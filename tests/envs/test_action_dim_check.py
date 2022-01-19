import pickle

import pytest

from gym import envs
from tests.envs.spec_list import skip_mujoco, SKIP_MUJOCO_WARNING_MESSAGE


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
