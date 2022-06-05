import pytest

import gym
from gym.envs.box2d import BipedalWalker
from gym.envs.box2d.lunar_lander import demo_heuristic_lander
from gym.envs.toy_text.frozen_lake import generate_random_map


def test_lunar_lander_heuristics():
    lunar_lander = gym.make("LunarLander-v2", disable_env_checker=True)
    total_reward = demo_heuristic_lander(lunar_lander, seed=1)
    assert total_reward > 100


@pytest.mark.parametrize(
    "env_name,kwargs",
    [
        ["LunarLanderContinuous-v2", {"continuous": True}],
        ["BipedalWalkerHardcore-v3", {"hardcore": True}],
        ["CarRacingDomainRandomize-v1", {"domain_randomize": True}],
        ["CarRacingDiscrete-v1", {"continuous": False}],
        [
            "CarRacingDomainRandomizeDiscrete-v1",
            {"domain_randomize": True, "continuous": False},
        ],
    ],
)
def test_env_kwargs(env_name, kwargs):
    # TODO
    # env = gym.make(env_name, disable_env_checker=True)
    pass


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

    env = gym.make("BipedalWalker-v3", disable_env_checker=True).unwrapped
    hc_env = gym.make("BipedalWalkerHardcore-v3", disable_env_checker=True).unwrapped
    assert isinstance(env, BipedalWalker) and isinstance(hc_env, BipedalWalker)
    assert env.hardcore is False and env.hardcore is True

    env.reset(seed=seed)
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


# Test that FrozenLake map generation creates valid maps of various sizes.
def test_frozenlake_dfs_map_generation():
    def frozenlake_dfs_path_exists(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == "G":
                        return True
                    if res[r_new][c_new] not in "#H":
                        frontier.append((r_new, c_new))
        return False

    map_sizes = [
        5,
        10,
        16,
    ]  # Currently, you can only create max 8x8 maps so there is no point testing beyond this
    for size in map_sizes:
        new_frozenlake = generate_random_map(size)
        assert len(new_frozenlake) == size
        assert len(new_frozenlake[0]) == size
        assert frozenlake_dfs_path_exists(new_frozenlake)
