import numpy as np
import pytest

from gym.vector.utils import BraxInfoProcessor


@pytest.mark.parametrize(
    ("num_envs", "info_key", "boolean_info_key"), ([3, "info", "_info"],)
)
def test_brax_vec_env_info(num_envs, info_key, boolean_info_key):
    infos = BraxInfoProcessor(num_envs)

    for i in range(num_envs - 1):
        info = {info_key: i}
        infos.add_info(info, i)

    expected_info = {
        info_key: np.array([0, 1, 0]),
        boolean_info_key: np.array([True, True, False]),
    }
    infos = infos.get_info()

    assert expected_info.keys() == infos.keys()
    assert all(expected_info[info_key] == infos[info_key])
