import pickle

import pytest

from gym import envs
from gym.envs.tests.spec_list import SKIP_MUJOCO_WARNING_MESSAGE, skip_mujoco


@pytest.mark.skipif(skip_mujoco, reason=SKIP_MUJOCO_WARNING_MESSAGE)
def test_serialize_deserialize():
    env1 = envs.make('HandReach-v0', distance_threshold=1e-6)
    env1.reset()
    env2 = pickle.loads(pickle.dumps(env1))

    assert env1.distance_threshold == env2.distance_threshold, (
        env1.distance_threshold, env2.distance_threshold)
