import pytest

from gym.vector.utils import BraxVecEnvInfoStrategy, ClassicVecEnvInfoStrategy


def test_classic_vec_env_info_strategy():
    NUM_ENVS = 3

    infos = ClassicVecEnvInfoStrategy(NUM_ENVS)
    for i in range(NUM_ENVS):
        info = {"example_info": i}
        infos.add_info(info, i)

    expected_info = [{"example_info": 0}, {"example_info": 1}, {"example_info": 2}]
    assert expected_info == infos.get_info()


def test_brax_vec_env_info_strategy():
    NUM_ENVS = 3

    infos = BraxVecEnvInfoStrategy(NUM_ENVS)
    for i in range(NUM_ENVS):
        info = {"example_info": i}
        infos.add_info(info, i)

    expected_info = {"example_info": [0, 1, 2]}
    assert expected_info == infos.get_info()


def test_brax_vec_env_info_strategy_with_nones():
    NUM_ENVS = 5

    infos = BraxVecEnvInfoStrategy(NUM_ENVS)
    for i in range(NUM_ENVS):
        if i % 2 == 0:
            info = {"example_info": i}
            infos.add_info(info, i)

    expected_info = {"example_info": [0, None, 2, None, 4]}
    assert expected_info == infos.get_info()
