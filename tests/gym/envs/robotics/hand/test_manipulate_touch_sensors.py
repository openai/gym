import pickle

from gym import envs
import pytest


ENVIRONMENT_IDS = (
    'HandManipulateEggTouchSensors-v1',
    'HandManipulatePenTouchSensors-v0',
    'HandManipulateBlockTouchSensors-v0',
)


@pytest.mark.parametrize("environment_id", ENVIRONMENT_IDS)
def test_serialize_deserialize(environment_id):
    env1 = envs.make(environment_id, target_position='fixed')
    env1.reset()
    env2 = pickle.loads(pickle.dumps(env1))

    assert env1.target_position == env2.target_position, (
        env1.target_position, env2.target_position)
