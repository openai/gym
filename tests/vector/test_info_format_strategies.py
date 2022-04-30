import numpy as np
import pytest

from gym.error import InvalidInfoFormat, InvalidInfoStrategy
from gym.vector.utils import (
    BraxVecEnvInfoStrategy,
    ClassicVecEnvInfoStrategy,
    InfoStrategyFactory,
    VecEnvInfoStrategy,
)


@pytest.mark.parametrize(("num_envs", "info_key"), ([3, "info"],))
def test_classic_vec_env_info_strategy(num_envs, info_key):
    infos = ClassicVecEnvInfoStrategy(num_envs)
    for i in range(num_envs):
        info = {info_key: i}
        infos.add_info(info, i)

    expected_info = [
        {info_key: 0},
        {info_key: 1},
        {info_key: 2},
    ]
    assert expected_info == infos.get_info()


@pytest.mark.parametrize(("num_envs", "info_key"), ([3, "info"],))
def test_brax_vec_env_info_strategy(num_envs, info_key):
    infos = BraxVecEnvInfoStrategy(num_envs)

    for i in range(num_envs):
        info = {info_key: i}
        infos.add_info(info, i)

    expected_info = {info_key: np.array([0, 1, 2])}
    infos = infos.get_info()

    assert expected_info.keys() == infos.keys()
    assert all(expected_info[info_key] == infos[info_key])


@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # bool dtype raise an informative warning to the user
@pytest.mark.parametrize(
    ("num_envs", "dtype", "info_key"),
    ([4, int, "info"], [4, float, "info"], [4, bool, "info"]),
)
def test_brax_vec_env_info_strategy_with_nones(num_envs, dtype, info_key):
    infos = BraxVecEnvInfoStrategy(num_envs)

    for i in range(num_envs):
        if i % 2 == 0:
            info = {info_key: dtype(1)}
            infos.add_info(info, i)

    expected_info = {info_key: np.array([1, 0, 1, 0], dtype=dtype)}
    infos = infos.get_info()

    for info in infos[info_key]:
        assert info.dtype == dtype
    assert all(expected_info[info_key] == infos[info_key])


@pytest.mark.parametrize(
    ("num_envs", "dtype", "info_key"),
    ([4, str, "info"], [4, list, "info"], [4, dict, "info"]),
)
def test_brax_vec_env_obj_info(num_envs, dtype, info_key):
    infos = BraxVecEnvInfoStrategy(num_envs)

    for i in range(num_envs):
        info = {info_key: dtype()}
        infos.add_info(info, i)

    infos = infos.get_info()
    assert infos[info_key].dtype == object


@pytest.mark.parametrize(("info_format"), ("classic", "brax", "non_existent"))
def test_get_info_strategy(info_format):
    if info_format == "classic":
        info_strategy = InfoStrategyFactory.get_info_strategy(info_format)
        assert info_strategy == ClassicVecEnvInfoStrategy
    elif info_format == "brax":
        info_strategy = InfoStrategyFactory.get_info_strategy(info_format)
        assert info_strategy == BraxVecEnvInfoStrategy
    else:
        with pytest.raises(InvalidInfoFormat):
            InfoStrategyFactory.get_info_strategy(info_format)


def test_add_valid_info_strategy():
    class Strategy(VecEnvInfoStrategy):
        ...

    InfoStrategyFactory.add_info_strategy("valid", Strategy)
    info_strategy = InfoStrategyFactory.get_info_strategy("valid")
    assert info_strategy == Strategy


def test_add_not_valid_info_strategy():
    class Strategy:
        ...

    with pytest.raises(InvalidInfoStrategy):
        InfoStrategyFactory.add_info_strategy("not_valid", Strategy)
