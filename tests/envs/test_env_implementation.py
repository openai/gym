import pytest
from typing import Optional

import gym
from gym.envs.box2d import BipedalWalker
from gym.envs.box2d.lunar_lander import demo_heuristic_lander
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np


def test_lunar_lander_heuristics():
    lunar_lander = gym.make("LunarLander-v2", disable_env_checker=True)
    total_reward = demo_heuristic_lander(lunar_lander, seed=1)
    assert total_reward > 100


@pytest.mark.parametrize("seed", range(5))
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
    assert env.hardcore is False and hc_env.hardcore is True

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


@pytest.mark.parametrize("map_size", [5, 10, 16])
def test_frozenlake_dfs_map_generation(map_size: int):
    """Frozenlake has the ability to generate random maps.

    This function checks that the random maps will always be possible to solve for sizes 5, 10, 16,
    currently only 8x8 maps can be generated.
    """
    new_frozenlake = generate_random_map(map_size)
    assert len(new_frozenlake) == map_size
    assert len(new_frozenlake[0]) == map_size

    # Runs a depth first search through the map to find the path.
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        row, col = frontier.pop()
        if (row, col) not in discovered:
            discovered.add((row, col))

            for row_direction, col_direction in directions:
                new_row = row + row_direction
                new_col = col + col_direction
                if 0 <= new_row < map_size and 0 <= new_col < map_size:
                    if new_frozenlake[new_row][new_col] == "G":
                        return  # Successful, a route through the map was found
                    if new_frozenlake[new_row][new_col] not in "#H":
                        frontier.append((new_row, new_col))
    raise AssertionError("No path through the frozenlake was found.")


@pytest.mark.parametrize("low_high",
                         [None,
                          (-1.0, 1.0),
                          (np.array(-1.0), np.array(1.0)),
                          (-1., 2.),
                          (np.array(-1.0), np.array(2.0)),
                          (-2., 1.),
                          (np.array(-2.0), np.array(1.0))])
def test_customizable_resets(low_high: Optional[list]):
    envs = {
        'acrobot': gym.make('Acrobot-v1'),
        'cartpole': gym.make('CartPole-v1'),
        'mountaincar': gym.make('MountainCar-v0'),
        'mountaincar_continuous': gym.make('MountainCarContinuous-v0'),
        'pendulum': gym.make('Pendulum-v1'),
    }
    for env in envs:
        # First ensure we can do a reset and the values are within expected ranges.
        if low_high is None:
            envs[env].reset()
        else:
            low, high = low_high
            for i in range(15):
                if env == 'pendulum':
                    # Pendulum is initialized a little differently, where we specify the
                    # x and y values for the upper limit (and lower limit is just the
                    # negative of it).
                    envs[env].reset(options={'x': low, 'y': high})
                else:
                    envs[env].reset(options={'low': low, 'high': high})
                assert np.all((envs[env].state >= low) & (envs[env].state <= high))
                if env.endswith('continuous') or env == 'pendulum':
                    envs[env].step([0])
                else:
                    envs[env].step(0)

@pytest.mark.parametrize("low_high",
                         [('x', 'y'), ([-1.], [1.]),
                          (np.array([-1.]), np.array([1.]))])
def test_invalid_customizable_resets(low_high: list):
    envs = {
        'acrobot': gym.make('Acrobot-v1'),
        'cartpole': gym.make('CartPole-v1'),
        'mountaincar': gym.make('MountainCar-v0'),
        'mountaincar_continuous': gym.make('MountainCarContinuous-v0'),
        'pendulum': gym.make('Pendulum-v1'),
    }
    for env in envs:
        low, high = low_high
        if env == 'pendulum':
            with pytest.raises(AssertionError):
                envs[env].reset(options={'x': low, 'y': high})
        else:
            with pytest.raises(AssertionError):
                envs[env].reset(options={'low': low, 'high': high})
