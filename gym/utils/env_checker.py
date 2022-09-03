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
from gym import logger, spaces
from gym.utils.passive_env_checker import (
    check_action_space,
    check_observation_space,
    env_render_passive_checker,
    env_reset_passive_checker,
    env_step_passive_checker,
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
        elif isinstance(data_1, (tuple, list)):
            return len(data_1) == len(data_2) and all(
                data_equivalence(o_1, o_2) for o_1, o_2 in zip(data_1, data_2)
            )
        elif isinstance(data_1, np.ndarray):
            return data_1.shape == data_2.shape and np.allclose(
                data_1, data_2, atol=0.00001
            )
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
    if "seed" in signature.parameters or (
        "kwargs" in signature.parameters
        and signature.parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    ):
        try:
            obs_1, info = env.reset(seed=123)
            assert (
                obs_1 in env.observation_space
            ), "The observation returned by `env.reset(seed=123)` is not within the observation space."
            assert (
                env.unwrapped._np_random  # pyright: ignore [reportPrivateUsage]
                is not None
            ), "Expects the random number generator to have been generated given a seed was passed to reset. Mostly likely the environment reset function does not call `super().reset(seed=seed)`."
            seed_123_rng = deepcopy(
                env.unwrapped._np_random  # pyright: ignore [reportPrivateUsage]
            )

            obs_2, info = env.reset(seed=123)
            assert (
                obs_2 in env.observation_space
            ), "The observation returned by `env.reset(seed=123)` is not within the observation space."
            if env.spec is not None and env.spec.nondeterministic is False:
                assert data_equivalence(
                    obs_1, obs_2
                ), "Using `env.reset(seed=123)` is non-deterministic as the observations are not equivalent."
            assert (
                env.unwrapped._np_random.bit_generator.state  # pyright: ignore [reportPrivateUsage]
                == seed_123_rng.bit_generator.state
            ), "Mostly likely the environment reset function does not call `super().reset(seed=seed)` as the random generates are not same when the same seeds are passed to `env.reset`."

            obs_3, info = env.reset(seed=456)
            assert (
                obs_3 in env.observation_space
            ), "The observation returned by `env.reset(seed=456)` is not within the observation space."
            assert (
                env.unwrapped._np_random.bit_generator.state  # pyright: ignore [reportPrivateUsage]
                != seed_123_rng.bit_generator.state
            ), "Mostly likely the environment reset function does not call `super().reset(seed=seed)` as the random number generators are not different when different seeds are passed to `env.reset`."

        except TypeError as e:
            raise AssertionError(
                "The environment cannot be reset with a random seed, even though `seed` or `kwargs` appear in the signature. "
                f"This should never happen, please report this issue. The error was: {e}"
            )

        seed_param = signature.parameters.get("seed")
        # Check the default value is None
        if seed_param is not None and seed_param.default is not None:
            logger.warn(
                "The default seed argument in reset should be `None`, otherwise the environment will by default always be deterministic. "
                f"Actual default: {seed_param.default}"
            )
    else:
        raise gym.error.Error(
            "The `reset` method does not provide a `seed` or `**kwargs` keyword argument."
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
    if "options" in signature.parameters or (
        "kwargs" in signature.parameters
        and signature.parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    ):
        try:
            env.reset(options={})
        except TypeError as e:
            raise AssertionError(
                "The environment cannot be reset with options, even though `options` or `**kwargs` appear in the signature. "
                f"This should never happen, please report this issue. The error was: {e}"
            )
    else:
        raise gym.error.Error(
            "The `reset` method does not provide an `options` or `**kwargs` keyword argument."
        )


def check_reset_return_info_deprecation(env: gym.Env):
    """Makes sure support for deprecated `return_info` argument is dropped.

    Args:
        env: The environment to check
    Raises:
        UserWarning
    """
    signature = inspect.signature(env.reset)
    if "return_info" in signature.parameters:
        logger.warn(
            "`return_info` is deprecated as an optional argument to `reset`. `reset`"
            "should now always return `obs, info` where `obs` is an observation, and `info` is a dictionary"
            "containing additional information."
        )


def check_seed_deprecation(env: gym.Env):
    """Makes sure support for deprecated function `seed` is dropped.

    Args:
        env: The environment to check
    Raises:
        UserWarning
    """
    seed_fn = getattr(env, "seed", None)
    if callable(seed_fn):
        logger.warn(
            "Official support for the `seed` function is dropped. "
            "Standard practice is to reset gym environments using `env.reset(seed=<desired seed>)`"
        )


def check_reset_return_type(env: gym.Env):
    """Checks that :meth:`reset` correctly returns a tuple of the form `(obs , info)`.

    Args:
        env: The environment to check
    Raises:
        AssertionError depending on spec violation
    """
    result = env.reset()
    assert isinstance(
        result, tuple
    ), f"The result returned by `env.reset()` was not a tuple of the form `(obs, info)`, where `obs` is a observation and `info` is a dictionary containing additional information. Actual type: `{type(result)}`"
    assert (
        len(result) == 2
    ), f"Calling the reset method did not return a 2-tuple, actual length: {len(result)}"

    obs, info = result
    assert (
        obs in env.observation_space
    ), "The first element returned by `env.reset()` is not within the observation space."
    assert isinstance(
        info, dict
    ), f"The second element returned by `env.reset()` was not a dictionary, actual type: {type(info)}"


def check_space_limit(space, space_type: str):
    """Check the space limit for only the Box space as a test that only runs as part of `check_env`."""
    if isinstance(space, spaces.Box):
        if np.any(np.equal(space.low, -np.inf)):
            logger.warn(
                f"A Box {space_type} space minimum value is -infinity. This is probably too low."
            )
        if np.any(np.equal(space.high, np.inf)):
            logger.warn(
                f"A Box {space_type} space maximum value is -infinity. This is probably too high."
            )

        # Check that the Box space is normalized
        if space_type == "action":
            if len(space.shape) == 1:  # for vector boxes
                if (
                    np.any(
                        np.logical_and(
                            space.low != np.zeros_like(space.low),
                            np.abs(space.low) != np.abs(space.high),
                        )
                    )
                    or np.any(space.low < -1)
                    or np.any(space.high > 1)
                ):
                    # todo - Add to gymlibrary.ml?
                    logger.warn(
                        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). "
                        "See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information."
                    )
    elif isinstance(space, spaces.Tuple):
        for subspace in space.spaces:
            check_space_limit(subspace, space_type)
    elif isinstance(space, spaces.Dict):
        for subspace in space.values():
            check_space_limit(subspace, space_type)


def check_env(env: gym.Env, warn: bool = None, skip_render_check: bool = False):
    """Check that an environment follows Gym API.

    This is an invasive function that calls the environment's reset and step.

    This is particularly useful when using a custom environment.
    Please take a look at https://www.gymlibrary.dev/content/environment_creation/
    for more information about the API.

    Args:
        env: The Gym environment that will be checked
        warn: Ignored
        skip_render_check: Whether to skip the checks for the render method. True by default (useful for the CI)
    """
    if warn is not None:
        logger.warn("`check_env(warn=...)` parameter is now ignored.")

    assert isinstance(
        env, gym.Env
    ), "The environment must inherit from the gym.Env class. See https://www.gymlibrary.dev/content/environment_creation/ for more info."

    if env.unwrapped is not env:
        logger.warn(
            f"The environment ({env}) is different from the unwrapped version ({env.unwrapped}). This could effect the environment checker as the environment most likely has a wrapper applied to it. We recommend using the raw environment for `check_env` using `env.unwrapped`."
        )

    # ============= Check the spaces (observation and action) ================
    assert hasattr(
        env, "action_space"
    ), "The environment must specify an action space. See https://www.gymlibrary.dev/content/environment_creation/ for more info."
    check_action_space(env.action_space)
    check_space_limit(env.action_space, "action")

    assert hasattr(
        env, "observation_space"
    ), "The environment must specify an observation space. See https://www.gymlibrary.dev/content/environment_creation/ for more info."
    check_observation_space(env.observation_space)
    check_space_limit(env.observation_space, "observation")

    # ==== Check the reset method ====
    check_seed_deprecation(env)
    check_reset_return_info_deprecation(env)
    check_reset_return_type(env)
    check_reset_seed(env)
    check_reset_options(env)

    # ============ Check the returned values ===============
    env_reset_passive_checker(env)
    env_step_passive_checker(env, env.action_space.sample())

    # ==== Check the render method and the declared render modes ====
    if not skip_render_check:
        if env.render_mode is not None:
            env_render_passive_checker(env)

        # todo: recreate the environment with a different render_mode for check that each work
