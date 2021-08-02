"""
This file is originally from the Stable Baselines3 repository hostedon GitHub
(https://github.com/DLR-RM/stable-baselines3/)
Original Author: Antonin Raffin

This file is covered by the MIT License, as described here:
The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import warnings
from typing import Union

import gym
import numpy as np
from gym import spaces

def _is_numpy_array_space(space: spaces.Space) -> bool:
    """
    Returns False if provided space is not representable as a single numpy array
    (e.g. Dict and Tuple spaces return False)
    """
    return not isinstance(space, (spaces.Dict, spaces.Tuple))


def _check_image_input(observation_space: spaces.Box, key: str = "") -> None:
    """
    Check that the input adheres to general standards
    when the observation is apparently an image.
    """
    if observation_space.dtype != np.uint8:
        warnings.warn(
            f"It seems that your observation {key} is an image but the `dtype` "
            "of your observation_space is not `np.uint8`. "
            "If your observation is not an image, we recommend you to flatten the observation "
            "to have only a 1D vector"
        )

    if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
        warnings.warn(
            f"It seems that your observation space {key} is an image but the "
            "upper and lower bounds are not in [0, 255]. "
            "Generally, CNN policies assume observations are within that range, "
            "so you may encounter an issue if the observation values are not."
        )

def _check_unsupported_spaces(
    env: gym.Env, observation_space: spaces.Space, action_space: spaces.Space
) -> None:
    """Emit warnings when the observation space or action space used is not supported by Stable-Baselines."""

    if isinstance(observation_space, spaces.Dict):
        nested_dict = False
        for space in observation_space.spaces.values():
            if isinstance(space, spaces.Dict):
                nested_dict = True
        if nested_dict:
            warnings.warn(
                "Nested observation spaces are not supported by Stable Baselines3 "
                "(Dict spaces inside Dict space). "
                "You should flatten it to have only one level of keys."
                "For example, `dict(space1=dict(space2=Box(), space3=Box()), spaces4=Discrete())` "
                "is not supported but `dict(space2=Box(), spaces3=Box(), spaces4=Discrete())` is."
            )

    if isinstance(observation_space, spaces.Tuple):
        warnings.warn(
            "The observation space is a Tuple,"
            "this is currently not supported by Stable Baselines3. "
            "However, you can convert it to a Dict observation space "
            "(cf. https://github.com/openai/gym/blob/master/gym/spaces/dict.py). "
            "which is supported by SB3."
        )

    if not _is_numpy_array_space(action_space):
        warnings.warn(
            "The action space is not based off a numpy array. Typically this means it's either a Dict or Tuple space. "
            "This type of action space is currently not supported by Stable Baselines 3. You should try to flatten the "
            "action using a wrapper."
        )


def _check_nan(env: gym.Env) -> None:
    """Check for Inf and NaN."""
    for _ in range(10):
        action = np.array([env.action_space.sample()])
        _, _, _, _ = env.step(action)


def _check_obs(
    obs: Union[tuple, dict, np.ndarray, int],
    observation_space: spaces.Space,
    method_name: str,
) -> None:
    """
    Check that the observation returned by the environment
    correspond to the declared one.
    """
    if not isinstance(observation_space, spaces.Tuple):
        assert not isinstance(
            obs, tuple
        ), f"The observation returned by the `{method_name}()` method should be a single value, not a tuple"

    # The check for a GoalEnv is done by the base class
    if isinstance(observation_space, spaces.Discrete):
        assert isinstance(
            obs, int
        ), f"The observation returned by `{method_name}()` method must be an int"
    elif _is_numpy_array_space(observation_space):
        assert isinstance(
            obs, np.ndarray
        ), f"The observation returned by `{method_name}()` method must be a numpy array"

    assert observation_space.contains(
        obs
    ), f"The observation returned by the `{method_name}()` method does not match the given observation space"


def _check_box_obs(observation_space: spaces.Box, key: str = "") -> None:
    """
    Check that the observation space is correctly formatted
    when dealing with a ``Box()`` space. In particular, it checks:
    - that the dimensions are big enough when it is an image, and that the type matches
    - that the observation has an expected shape (warn the user if not)
    """
    # If image, check the low and high values, the type and the number of channels
    # and the shape (minimal value)
    if len(observation_space.shape) == 3:
        _check_image_input(observation_space)

    if len(observation_space.shape) not in [1, 3]:
        warnings.warn(
            f"Your observation {key} has an unconventional shape (neither an image, nor a 1D vector). "
            "We recommend you to flatten the observation "
            "to have only a 1D vector or use a custom policy to properly process the data."
        )


def _check_returned_values(
    env: gym.Env, observation_space: spaces.Space, action_space: spaces.Space
) -> None:
    """
    Check the returned values by the env when calling `.reset()` or `.step()` methods.
    """
    # because env inherits from gym.Env, we assume that `reset()` and `step()` methods exists
    obs = env.reset()

    if isinstance(observation_space, spaces.Dict):
        assert isinstance(
            obs, dict
        ), "The observation returned by `reset()` must be a dictionary"
        for key in observation_space.spaces.keys():
            try:
                _check_obs(obs[key], observation_space.spaces[key], "reset")
            except AssertionError as e:
                raise AssertionError(f"Error while checking key={key}: " + str(e))
    else:
        _check_obs(obs, observation_space, "reset")

    # Sample a random action
    action = action_space.sample()
    data = env.step(action)

    assert (
        len(data) == 4
    ), "The `step()` method must return four values: obs, reward, done, info"

    # Unpack
    obs, reward, done, info = data

    if isinstance(observation_space, spaces.Dict):
        assert isinstance(
            obs, dict
        ), "The observation returned by `step()` must be a dictionary"
        for key in observation_space.spaces.keys():
            try:
                _check_obs(obs[key], observation_space.spaces[key], "step")
            except AssertionError as e:
                raise AssertionError(f"Error while checking key={key}: " + str(e))

    else:
        _check_obs(obs, observation_space, "step")

    # We also allow int because the reward will be cast to float
    assert isinstance(
        reward, (float, int)
    ), "The reward returned by `step()` must be a float"
    assert isinstance(done, bool), "The `done` signal must be a boolean"
    assert isinstance(
        info, dict
    ), "The `info` returned by `step()` must be a python dictionary"

    if isinstance(env, gym.GoalEnv):
        # For a GoalEnv, the keys are checked at reset
        assert reward == env.compute_reward(
            obs["achieved_goal"], obs["desired_goal"], info
        )


def _check_spaces(env: gym.Env) -> None:
    """
    Check that the observation and action spaces are defined
    and inherit from gym.spaces.Space.
    """
    # Helper to link to the code, because gym has no proper documentation
    gym_spaces = " cf https://github.com/openai/gym/blob/master/gym/spaces/"

    assert hasattr(env, "observation_space"), (
        "You must specify an observation space (cf gym.spaces)" + gym_spaces
    )
    assert hasattr(env, "action_space"), (
        "You must specify an action space (cf gym.spaces)" + gym_spaces
    )

    assert isinstance(env.observation_space, spaces.Space), (
        "The observation space must inherit from gym.spaces" + gym_spaces
    )
    assert isinstance(env.action_space, spaces.Space), (
        "The action space must inherit from gym.spaces" + gym_spaces
    )


# Check render cannot be covered by CI
def _check_render(
    env: gym.Env, warn: bool = True, headless: bool = False
) -> None:  # pragma: no cover
    """
    Check the declared render modes and the `render()`/`close()`
    method of the environment.
    :param env: The environment to check
    :param warn: Whether to output additional warnings
    :param headless: Whether to disable render modes
        that require a graphical interface. False by default.
    """
    render_modes = env.metadata.get("render.modes")
    if render_modes is None:
        if warn:
            warnings.warn(
                "No render modes was declared in the environment "
                " (env.metadata['render.modes'] is None or not defined), "
                "you may have trouble when calling `.render()`"
            )

    else:
        # Don't check render mode that require a
        # graphical interface (useful for CI)
        if headless and "human" in render_modes:
            render_modes.remove("human")
        # Check all declared render modes
        for render_mode in render_modes:
            env.render(mode=render_mode)
        env.close()


def check_env(env: gym.Env, warn: bool = True, skip_render_check: bool = True) -> None:
    """
    Check that an environment follows Gym API.
    This is particularly useful when using a custom environment.
    Please take a look at https://github.com/openai/gym/blob/master/gym/core.py
    for more information about the API.
    It also optionally check that the environment is compatible with Stable-Baselines.
    :param env: The Gym environment that will be checked
    :param warn: Whether to output additional warnings
        mainly related to the interaction with Stable Baselines
    :param skip_render_check: Whether to skip the checks for the render method.
        True by default (useful for the CI)
    """
    assert isinstance(
        env, gym.Env
    ), "Your environment must inherit from the gym.Env class cf https://github.com/openai/gym/blob/master/gym/core.py"

    # ============= Check the spaces (observation and action) ================
    _check_spaces(env)

    # Define aliases for convenience
    observation_space = env.observation_space
    action_space = env.action_space

    # Warn the user if needed.
    # A warning means that the environment may run but not work properly with Stable Baselines algorithms
    if warn:
        _check_unsupported_spaces(env, observation_space, action_space)

        obs_spaces = (
            observation_space.spaces
            if isinstance(observation_space, spaces.Dict)
            else {"": observation_space}
        )
        for key, space in obs_spaces.items():
            if isinstance(space, spaces.Box):
                _check_box_obs(space, key)

        # Check for the action space, it may lead to hard-to-debug issues
        if isinstance(action_space, spaces.Box) and (
            np.any(np.abs(action_space.low) != np.abs(action_space.high))
            or np.any(np.abs(action_space.low) > 1)
            or np.any(np.abs(action_space.high) > 1)
        ):
            warnings.warn(
                "We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) "
                "cf https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html"
            )

    # ============ Check the returned values ===============
    _check_returned_values(env, observation_space, action_space)

    # ==== Check the render method and the declared render modes ====
    if not skip_render_check:
        _check_render(env, warn=warn)  # pragma: no cover

    # The check only works with numpy arrays
    if _is_numpy_array_space(observation_space) and _is_numpy_array_space(action_space):
        _check_nan(env)
