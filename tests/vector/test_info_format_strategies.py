import pytest

from gym.error import InvalidInfoFormat
from gym.vector.utils import (
    BraxVecEnvInfoStrategy,
    ClassicVecEnvInfoStrategy,
    get_info_strategy,
)


@pytest.mark.parametrize(("num_envs"), [3])
def test_classic_vec_env_info_strategy(num_envs):
    infos = ClassicVecEnvInfoStrategy(num_envs)
    for i in range(num_envs):
        info = {"example_info": i}
        infos.add_info(info, i)

    expected_info = [{"example_info": 0}, {"example_info": 1}, {"example_info": 2}]
    assert expected_info == infos.get_info()


@pytest.mark.parametrize(("num_envs"), [3])
def test_brax_vec_env_info_strategy(num_envs):
    NUM_ENVS = 3

    infos = BraxVecEnvInfoStrategy(num_envs)
    for i in range(num_envs):
        info = {"example_info": i}
        infos.add_info(info, i)

    expected_info = {"example_info": [0, 1, 2]}
    assert expected_info == infos.get_info()


@pytest.mark.parametrize(("num_envs"), [5])
def test_brax_vec_env_info_strategy_with_nones(num_envs):
    infos = BraxVecEnvInfoStrategy(num_envs)
    for i in range(num_envs):
        if i % 2 == 0:
            info = {"example_info": i}
            infos.add_info(info, i)

    expected_info = {"example_info": [0, None, 2, None, 4]}
    assert expected_info == infos.get_info()


@pytest.mark.parametrize(("info_format"), [("classic"), ("brax"), ("non_existent")])
def test_get_info_strategy(info_format):
    if info_format == "classic":
        info_strategy = get_info_strategy(info_format)
        assert info_strategy == ClassicVecEnvInfoStrategy
    elif info_format == "brax":
        info_strategy = get_info_strategy(info_format)
        assert info_strategy == BraxVecEnvInfoStrategy
    else:
        with pytest.raises(InvalidInfoFormat):
            get_info_strategy(info_format)
