from typing import Optional

import numpy as np
import pytest

import gym
from gym.envs.box2d import BipedalWalker
from gym.envs.box2d.lunar_lander import demo_heuristic_lander
from gym.envs.toy_text import TaxiEnv
from gym.envs.toy_text.frozen_lake import generate_random_map


def test_lunar_lander_heuristics():
    """Tests the LunarLander environment by checking if the heuristic lander works."""
    lunar_lander = gym.make("LunarLander-v2", disable_env_checker=True)
    total_reward = demo_heuristic_lander(lunar_lander, seed=1)
    assert total_reward > 100


def test_carracing_domain_randomize():
    """Tests the CarRacing Environment domain randomization.

    CarRacing DomainRandomize should have different colours at every reset.
    However, it should have same colours when `options={"randomize": False}` is given to reset.
    """
    env = gym.make("CarRacing-v2", domain_randomize=True)

    road_color = env.road_color
    bg_color = env.bg_color
    grass_color = env.grass_color

    env.reset(options={"randomize": False})

    assert (
        road_color == env.road_color
    ).all(), f"Have different road color after reset with randomize turned off. Before: {road_color}, after: {env.road_color}."
    assert (
        bg_color == env.bg_color
    ).all(), f"Have different bg color after reset with randomize turned off. Before: {bg_color}, after: {env.bg_color}."
    assert (
        grass_color == env.grass_color
    ).all(), f"Have different grass color after reset with randomize turned off. Before: {grass_color}, after: {env.grass_color}."

    env.reset()

    assert (
        road_color != env.road_color
    ).all(), f"Have same road color after reset. Before: {road_color}, after: {env.road_color}."
    assert (
        bg_color != env.bg_color
    ).all(), (
        f"Have same bg color after reset. Before: {bg_color}, after: {env.bg_color}."
    )
    assert (
        grass_color != env.grass_color
    ).all(), f"Have same grass color after reset. Before: {grass_color}, after: {env.grass_color}."


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


def test_taxi_action_mask():
    env = TaxiEnv()

    for state in env.P:
        mask = env.action_mask(state)
        for action, possible in enumerate(mask):
            _, next_state, _, _ = env.P[state][action][0]
            assert state != next_state if possible else state == next_state


def test_taxi_encode_decode():
    env = TaxiEnv()

    state, info = env.reset()
    for _ in range(100):
        assert (
            env.encode(*env.decode(state)) == state
        ), f"state={state}, encode(decode(state))={env.encode(*env.decode(state))}"
        state, _, _, _, _ = env.step(env.action_space.sample())


@pytest.mark.parametrize(
    "env_name",
    ["Acrobot-v1", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0"],
)
@pytest.mark.parametrize(
    "low_high", [None, (-0.4, 0.4), (np.array(-0.4), np.array(0.4))]
)
def test_customizable_resets(env_name: str, low_high: Optional[list]):
    env = gym.make(env_name)
    env.action_space.seed(0)
    # First ensure we can do a reset.
    if low_high is None:
        env.reset()
    else:
        low, high = low_high
        env.reset(options={"low": low, "high": high})
        assert np.all((env.state >= low) & (env.state <= high))
    # Make sure we can take a step.
    env.step(env.action_space.sample())


# We test Pendulum separately, as the parameters are handled differently.
@pytest.mark.parametrize(
    "low_high",
    [
        None,
        (1.2, 1.0),
        (np.array(1.2), np.array(1.0)),
    ],
)
def test_customizable_pendulum_resets(low_high: Optional[list]):
    env = gym.make("Pendulum-v1")
    env.action_space.seed(0)
    # First ensure we can do a reset and the values are within expected ranges.
    if low_high is None:
        env.reset()
    else:
        low, high = low_high
        # Pendulum is initialized a little differently than the other
        # environments, where we specify the x and y values for the upper
        # limit (and lower limit is just the negative of it).
        env.reset(options={"x_init": low, "y_init": high})
    # Make sure we can take a step.
    env.step(env.action_space.sample())


@pytest.mark.parametrize(
    "env_name",
    ["Acrobot-v1", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0"],
)
@pytest.mark.parametrize(
    "low_high",
    [
        ("x", "y"),
        (10.0, 8.0),
        ([-1.0, -1.0], [1.0, 1.0]),
        (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
    ],
)
def test_invalid_customizable_resets(env_name: str, low_high: list):
    env = gym.make(env_name)
    low, high = low_high
    with pytest.raises(ValueError):
        # match=re.escape(f"Lower bound ({low}) must be lower than higher bound ({high}).")
        # match=f"An option ({x}) could not be converted to a float."
        env.reset(options={"low": low, "high": high})
