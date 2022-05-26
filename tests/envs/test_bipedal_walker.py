"""Test BipedalWalker environment."""
import pytest

from gym.envs.box2d import BipedalWalker


@pytest.mark.parametrize("seed", range(10))
def test_bipedal_walker_hardcore_creation(seed: int):
    """Test BipedalWalker hardcore creation.

    BipedalWalker with `hardcore=True` should have ladders
    stumps and pitfalls. A convenient way to identify if ladders,
    stumps and pitfall are created is checking whether the terrain
    has that particular terrain color.

    Args:
        seed (int): environment seed
    """
    HC_TERRAINS_COLOR1 = (255, 255, 255)
    HC_TERRAINS_COLOR2 = (153, 153, 153)

    env = BipedalWalker(hardcore=False)
    env.reset(seed=seed)

    hc_env = BipedalWalker(hardcore=True)
    hc_env.reset(seed=seed)

    for terrain in env.terrain:
        assert terrain.color1 != HC_TERRAINS_COLOR1
        assert terrain.color2 != HC_TERRAINS_COLOR2

    hc_terrains_color1_count = 0
    hc_terrains_color2_count = 0
    for terrain in hc_env.terrain:
        if terrain.color1 == HC_TERRAINS_COLOR1:
            hc_terrains_color1_count += 1
        if terrain.color2 == HC_TERRAINS_COLOR2:
            hc_terrains_color2_count += 1

    assert hc_terrains_color1_count > 0
    assert hc_terrains_color2_count > 0
