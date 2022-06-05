"""A set of functions for checking an environment details.

This file is originally from the Stable Baselines3 repository hosted on GitHub
(https://github.com/DLR-RM/stable-baselines3/)
Original Author: Antonin Raffin

It also uses some warnings/assertions from the PettingZoo repository hosted on GitHub
(https://github.com/PettingZoo-Team/PettingZoo)
Original Author: J K Terry

This was rewritten and split into "env_checker.py" and "passive_env_checker.py" for invasive and passive environment checking
Original Author: Mark Towers

These projects are covered by the MIT License.
"""

import inspect
from copy import deepcopy

import numpy as np

import gym
from gym import error, logger
from gym.utils.passive_env_checker import (
    check_action_space,
    check_observation_space,
    passive_env_reset_check,
    passive_env_step_check,
)


def data_equivalence(data_1, data_2) -> bool:
    """Assert equality between data 1 and 2, i.e observations, actions, info.

    Args:
        data_1: data structure 1
        data_2: data structure 2

    Returns:
        If observation 1 and 2 are equivalent
    """
    if type(data_1) == type(data_2):
        if isinstance(data_1, dict):
            return data_1.keys() == data_2.keys() and all(
                data_equivalence(data_1[k], data_2[k]) for k in data_1.keys()
            )
        elif isinstance(data_1, tuple):
            return len(data_1) == len(data_2) and all(
                data_equivalence(o_1, o_2) for o_1, o_2 in zip(data_1, data_2)
            )
        elif isinstance(data_1, np.ndarray):
            return np.all(data_1 == data_2)
        else:
            return data_1 == data_2
    else:
        return False


def check_reset_seed(env: gym.Env):
    """Check that the environment can be reset with a seed.

    Args:
        env: The environment to check

    Raises:
        AssertionError: The environment cannot be reset with a random seed,
            even though `seed` or `kwargs` appear in the signature.
    """
    signature = inspect.signature(env.reset)
    if "seed" in signature.parameters or "kwargs" in signature.parameters:
        try:
            obs_1 = env.reset(seed=123)
            assert obs_1 in env.observation_space
            obs_2 = env.reset(seed=123)
            assert obs_2 in env.observation_space
            assert data_equivalence(obs_1, obs_2)
            seed_123_rng = deepcopy(env.unwrapped.np_random)

            # Note: for some environment, they may initialise at the same state, therefore we cannot check the obs_1 != obs_3
            obs_4 = env.reset(seed=None)
            assert obs_4 in env.observation_space

            assert (
                env.unwrapped.np_random.bit_generator.state
                != seed_123_rng.bit_generator.state
            )
        except TypeError as e:
            raise AssertionError(
                "The environment cannot be reset with a random seed, even though `seed` or `kwargs` appear in the signature. "
                "This should never happen, please report this issue. "
                f"The error was: {e}"
            )

        if env.unwrapped.np_random is None:
            logger.warn(
                "Resetting the environment did not result in seeding its random number generator. "
                "This is likely due to not calling `super().reset(seed=seed)` in the `reset` method. "
                "If you do not use the python-level random number generator, this is not a problem."
            )

        seed_param = signature.parameters.get("seed")
        # Check the default value is None
        if seed_param is not None and seed_param.default is not None:
            logger.warn(
                "The default seed argument in reset should be `None`, "
                "otherwise the environment will by default always be deterministic"
            )
    else:
        raise error.Error(
            "The `reset` method does not provide the `return_info` keyword argument"
        )


def check_reset_info(env: gym.Env):
    """Checks that :meth:`reset` supports the ``return_info`` keyword.

    Args:
        env: The environment to check

    Raises:
        AssertionError: The environment cannot be reset with `return_info=True`,
            even though `return_info` or `kwargs` appear in the signature.
    """
    signature = inspect.signature(env.reset)
    if "return_info" in signature.parameters or "kwargs" in signature.parameters:
        try:
            result = env.reset(return_info=True)
            assert (
                len(result) == 2
            ), "Calling the reset method with `return_info=True` did not return a 2-tuple"
            obs, info = result
            assert isinstance(
                info, dict
            ), "The second element returned by `env.reset(return_info=True)` was not a dictionary"
        except TypeError as e:
            raise AssertionError(
                "The environment cannot be reset with `return_info=True`, even though `return_info` or `kwargs` "
                "appear in the signature. This should never happen, please report this issue. "
                f"The error was: {e}"
            )
    else:
        raise error.Error(
            "The `reset` method does not provide the `return_info` keyword argument"
        )


def check_reset_options(env: gym.Env):
    """Check that the environment can be reset with options.

    Args:
        env: The environment to check

    Raises:
        AssertionError: The environment cannot be reset with options,
            even though `options` or `kwargs` appear in the signature.
    """
    signature = inspect.signature(env.reset)
    if "options" in signature.parameters or "kwargs" in signature.parameters:
        try:
            env.reset(options={})
        except TypeError as e:
            raise AssertionError(
                "The environment cannot be reset with options, even though `options` or `kwargs` appear in the signature. "
                "This should never happen, please report this issue. "
                f"The error was: {e}"
            )
    else:
        raise error.Error(
            "The `reset` method does not provide the `options` keyword argument"
        )


# Check render cannot be covered by CI
def check_render(env: gym.Env, headless: bool = False):
    """Check the declared render modes/fps and the :meth:`render`/:meth:`close` method of the environment.

    Args:
        env: The environment to check
        headless: Whether to disable render modes that require a graphical interface. False by default.
    """
    render_modes = env.metadata.get("render_modes")
    if render_modes is None:
        logger.warn(
            "No render modes was declared in the environment  (env.metadata['render_modes'] is None or not defined), you may have trouble when calling `.render()`"
        )

    render_fps = env.metadata.get("render_fps")
    # We only require `render_fps` if rendering is actually implemented
    if render_fps is None:
        logger.warn(
            "No render fps was declared in the environment (env.metadata['render_fps'] is None or not defined), rendering may occur at inconsistent fps"
        )

    if render_modes is not None:
        # Don't check render mode that require a
        # graphical interface (useful for CI)
        if headless and "human" in render_modes:
            render_modes.remove("human")

        # Check all declared render modes
        for mode in render_modes:
            env.render(mode=mode)
        env.close()


def check_env(env: gym.Env, warn: bool = None, skip_render_check: bool = True):
    """Check that an environment follows Gym API.

    This is an invasive function that calls the environment's reset and step.

    This is particularly useful when using a custom environment.
    Please take a look at https://www.gymlibrary.ml/content/environment_creation/
    for more information about the API.

    Args:
        env: The Gym environment that will be checked
        warn: Ignored
        skip_render_check: Whether to skip the checks for the render method. True by default (useful for the CI)
    """
    if warn is not None:
        logger.warn("`check_env` warn parameter is now ignored.")

    assert isinstance(
        env, gym.Env
    ), "Your environment must inherit from the gym.Env class https://www.gymlibrary.ml/content/environment_creation/"

    # ============= Check the spaces (observation and action) ================
    assert hasattr(
        env, "action_space"
    ), "You must specify a action space. https://www.gymlibrary.ml/content/environment_creation/"
    check_observation_space(env.action_space)
    assert hasattr(
        env, "observation_space"
    ), "You must specify an observation space. https://www.gymlibrary.ml/content/environment_creation/"
    check_action_space(env.observation_space)

    # ==== Check the reset method ====
    check_reset_seed(env)
    check_reset_options(env)
    check_reset_info(env)

    # ============ Check the returned values ===============
    passive_env_reset_check(env)
    passive_env_step_check(env, env.action_space.sample())

    # ==== Check the render method and the declared render modes ====
    if not skip_render_check:
        check_render(env)
