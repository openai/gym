"""A set of functions for passively checking environment implementations."""
import inspect
from functools import partial
from typing import Callable

import numpy as np

from gym import Space, error, logger, spaces


def _check_box_observation_space(observation_space: spaces.Box):
    """Checks that a :class:`Box` observation space is defined in a sensible way.

    Args:
        observation_space: A box observation space
    """
    # Check if the box is an image
    if len(observation_space.shape) == 3:
        if observation_space.dtype != np.uint8:
            logger.warn(
                f"It seems a Box observation space is an image but the `dtype` is not `np.uint8`, actual type: {observation_space.dtype}. "
                "If the Box observation space is not an image, we recommend flattening the observation to have only a 1D vector."
            )
        if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
            logger.warn(
                "It seems a Box observation space is an image but the upper and lower bounds are not in [0, 255]. "
                "Generally, CNN policies assume observations are within that range, so you may encounter an issue if the observation values are not."
            )

    if len(observation_space.shape) not in [1, 3]:
        logger.warn(
            "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). "
            "We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. "
            f"Actual observation shape: {observation_space.shape}"
        )

    assert (
        observation_space.low.shape == observation_space.shape
    ), f"The Box observation space shape and low shape have different shapes, low shape: {observation_space.low.shape}, box shape: {observation_space.shape}"
    assert (
        observation_space.high.shape == observation_space.shape
    ), f"The Box observation space shape and high shape have have different shapes, high shape: {observation_space.high.shape}, box shape: {observation_space.shape}"

    if np.any(observation_space.low == observation_space.high):
        logger.warn("A Box observation space maximum and minimum values are equal.")
    elif np.any(observation_space.high < observation_space.low):
        logger.warn("A Box observation space low value is greater than a high value.")


def _check_box_action_space(action_space: spaces.Box):
    """Checks that a :class:`Box` action space is defined in a sensible way.

    Args:
        action_space: A box action space
    """
    assert (
        action_space.low.shape == action_space.shape
    ), f"The Box action space shape and low shape have have different shapes, low shape: {action_space.low.shape}, box shape: {action_space.shape}"
    assert (
        action_space.high.shape == action_space.shape
    ), f"The Box action space shape and high shape have different shapes, high shape: {action_space.high.shape}, box shape: {action_space.shape}"

    if np.any(action_space.low == action_space.high):
        logger.warn("A Box action space maximum and minimum values are equal.")
    elif np.any(action_space.high < action_space.low):
        logger.warn("A Box action space low value is greater than a high value.")


def check_space(
    space: Space, space_type: str, check_box_space_fn: Callable[[spaces.Box], None]
):
    """A passive check of the environment action space that should not affect the environment."""
    if not isinstance(space, spaces.Space):
        raise AssertionError(
            f"{space_type} space does not inherit from `gym.spaces.Space`, actual type: {type(space)}"
        )

    elif isinstance(space, spaces.Box):
        check_box_space_fn(space)
    elif isinstance(space, spaces.Discrete):
        assert (
            0 < space.n
        ), f"Discrete {space_type} space's number of elements must be positive, actual number of elements: {space.n}"
        assert (
            space.shape == ()
        ), f"Discrete {space_type} space's shape should be empty, actual shape: {space.shape}"
    elif isinstance(space, spaces.MultiDiscrete):
        assert (
            space.shape == space.nvec.shape
        ), f"Multi-discrete {space_type} space's shape must be equal to the nvec shape, space shape: {space.shape}, nvec shape: {space.nvec.shape}"
        assert np.all(
            0 < space.nvec
        ), f"Multi-discrete {space_type} space's all nvec elements must be greater than 0, actual nvec: {space.nvec}"
    elif isinstance(space, spaces.MultiBinary):
        assert np.all(
            0 < np.asarray(space.shape)
        ), f"Multi-binary {space_type} space's all shape elements must be greater than 0, actual shape: {space.shape}"
    elif isinstance(space, spaces.Tuple):
        assert 0 < len(
            space.spaces
        ), f"An empty Tuple {space_type} space is not allowed."
        for subspace in space.spaces:
            check_space(subspace, space_type, check_box_space_fn)
    elif isinstance(space, spaces.Dict):
        assert 0 < len(
            space.spaces.keys()
        ), f"An empty Dict {space_type} space is not allowed."
        for subspace in space.values():
            check_space(subspace, space_type, check_box_space_fn)


check_observation_space = partial(
    check_space,
    space_type="observation",
    check_box_space_fn=_check_box_observation_space,
)
check_action_space = partial(
    check_space, space_type="action", check_box_space_fn=_check_box_action_space
)


def check_obs(obs, observation_space: spaces.Space, method_name: str):
    """Check that the observation returned by the environment correspond to the declared one.

    Args:
        obs: The observation to check
        observation_space: The observation space of the observation
        method_name: The method name that generated the observation
    """
    pre = f"The obs returned by the `{method_name}()` method"
    if isinstance(observation_space, spaces.Discrete):
        if not isinstance(obs, (np.int64, int)):
            logger.warn(f"{pre} should be an int or np.int64, actual type: {type(obs)}")
    elif isinstance(observation_space, spaces.Box):
        if observation_space.shape != ():
            if not isinstance(obs, np.ndarray):
                logger.warn(
                    f"{pre} was expecting a numpy array, actual type: {type(obs)}"
                )
            elif obs.dtype != observation_space.dtype:
                logger.warn(
                    f"{pre} was expecting numpy array dtype to be {observation_space.dtype}, actual type: {obs.dtype}"
                )
    elif isinstance(observation_space, (spaces.MultiBinary, spaces.MultiDiscrete)):
        if not isinstance(obs, np.ndarray):
            logger.warn(f"{pre} was expecting a numpy array, actual type: {type(obs)}")
    elif isinstance(observation_space, spaces.Tuple):
        if not isinstance(obs, tuple):
            logger.warn(f"{pre} was expecting a tuple, actual type: {type(obs)}")
        assert len(obs) == len(
            observation_space.spaces
        ), f"{pre} length is not same as the observation space length, obs length: {len(obs)}, space length: {len(observation_space.spaces)}"
        for sub_obs, sub_space in zip(obs, observation_space.spaces):
            check_obs(sub_obs, sub_space, method_name)
    elif isinstance(observation_space, spaces.Dict):
        assert isinstance(obs, dict), f"{pre} must be a dict, actual type: {type(obs)}"
        assert (
            obs.keys() == observation_space.spaces.keys()
        ), f"{pre} observation keys is not same as the observation space keys, obs keys: {list(obs.keys())}, space keys: {list(observation_space.spaces.keys())}"
        for space_key in observation_space.spaces.keys():
            check_obs(obs[space_key], observation_space[space_key], method_name)

    try:
        if obs not in observation_space:
            logger.warn(f"{pre} is not within the observation space.")
    except Exception as e:
        logger.warn(f"{pre} is not within the observation space with exception: {e}")


def env_reset_passive_checker(env, **kwargs):
    """A passive check of the `Env.reset` function investigating the returning reset information and returning the data unchanged."""
    signature = inspect.signature(env.reset)
    if "seed" not in signature.parameters and "kwargs" not in signature.parameters:
        logger.warn(
            "Future gym versions will require that `Env.reset` can be passed a `seed` instead of using `Env.seed` for resetting the environment random number generator."
        )
    else:
        seed_param = signature.parameters.get("seed")
        # Check the default value is None
        if seed_param is not None and seed_param.default is not None:
            logger.warn(
                "The default seed argument in `Env.reset` should be `None`, otherwise the environment will by default always be deterministic. "
                f"Actual default: {seed_param}"
            )

    if "options" not in signature.parameters and "kwargs" not in signature.parameters:
        logger.warn(
            "Future gym versions will require that `Env.reset` can be passed `options` to allow the environment initialisation to be passed additional information."
        )

    # Checks the result of env.reset with kwargs
    result = env.reset(**kwargs)

    if not isinstance(result, tuple):
        logger.warn(
            f"The result returned by `env.reset()` was not a tuple of the form `(obs, info)`, where `obs` is a observation and `info` is a dictionary containing additional information. Actual type: `{type(result)}`"
        )
    elif len(result) != 2:
        logger.warn(
            "The result returned by `env.reset()` should be `(obs, info)` by default, , where `obs` is a observation and `info` is a dictionary containing additional information."
        )
    else:
        obs, info = result
        check_obs(obs, env.observation_space, "reset")
        assert isinstance(
            info, dict
        ), f"The second element returned by `env.reset()` was not a dictionary, actual type: {type(info)}"
    return result


def env_step_passive_checker(env, action):
    """A passive check for the environment step, investigating the returning data then returning the data unchanged."""
    # We don't check the action as for some environments then out-of-bounds values can be given
    result = env.step(action)
    assert isinstance(
        result, tuple
    ), f"Expects step result to be a tuple, actual type: {type(result)}"
    if len(result) == 4:
        logger.deprecation(
            "Core environment is written in old step API which returns one bool instead of two. "
            "It is recommended to rewrite the environment with new step API. "
        )
        obs, reward, done, info = result

        if not isinstance(done, (bool, np.bool8)):
            logger.warn(
                f"Expects `done` signal to be a boolean, actual type: {type(done)}"
            )
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result

        # np.bool is actual python bool not np boolean type, therefore bool_ or bool8
        if not isinstance(terminated, (bool, np.bool8)):
            logger.warn(
                f"Expects `terminated` signal to be a boolean, actual type: {type(terminated)}"
            )
        if not isinstance(truncated, (bool, np.bool8)):
            logger.warn(
                f"Expects `truncated` signal to be a boolean, actual type: {type(truncated)}"
            )
    else:
        raise error.Error(
            f"Expected `Env.step` to return a four or five element tuple, actual number of elements returned: {len(result)}."
        )

    check_obs(obs, env.observation_space, "step")

    if not (
        np.issubdtype(type(reward), np.integer)
        or np.issubdtype(type(reward), np.floating)
    ):
        logger.warn(
            f"The reward returned by `step()` must be a float, int, np.integer or np.floating, actual type: {type(reward)}"
        )
    else:
        if np.isnan(reward):
            logger.warn("The reward is a NaN value.")
        if np.isinf(reward):
            logger.warn("The reward is an inf value.")

    assert isinstance(
        info, dict
    ), f"The `info` returned by `step()` must be a python dictionary, actual type: {type(info)}"

    return result


def env_render_passive_checker(env, *args, **kwargs):
    """A passive check of the `Env.render` that the declared render modes/fps in the metadata of the environment is declared."""
    render_modes = env.metadata.get("render_modes")
    if render_modes is None:
        logger.warn(
            "No render modes was declared in the environment (env.metadata['render_modes'] is None or not defined), you may have trouble when calling `.render()`."
        )
    else:
        if not isinstance(render_modes, (list, tuple)):
            logger.warn(
                f"Expects the render_modes to be a sequence (i.e. list, tuple), actual type: {type(render_modes)}"
            )
        elif not all(isinstance(mode, str) for mode in render_modes):
            logger.warn(
                f"Expects all render modes to be strings, actual types: {[type(mode) for mode in render_modes]}"
            )

        render_fps = env.metadata.get("render_fps")
        # We only require `render_fps` if rendering is actually implemented
        if len(render_modes) > 0:
            if render_fps is None:
                logger.warn(
                    "No render fps was declared in the environment (env.metadata['render_fps'] is None or not defined), rendering may occur at inconsistent fps."
                )
            else:
                if not (
                    np.issubdtype(type(render_fps), np.integer)
                    or np.issubdtype(type(render_fps), np.floating)
                ):
                    logger.warn(
                        f"Expects the `env.metadata['render_fps']` to be an integer or a float, actual type: {type(render_fps)}"
                    )
                else:
                    assert (
                        render_fps > 0
                    ), f"Expects the `env.metadata['render_fps']` to be greater than zero, actual value: {render_fps}"

        # env.render is now an attribute with default None
        if len(render_modes) == 0:
            assert (
                env.render_mode is None
            ), f"With no render_modes, expects the Env.render_mode to be None, actual value: {env.render_mode}"
        else:
            assert env.render_mode is None or env.render_mode in render_modes, (
                "The environment was initialized successfully however with an unsupported render mode. "
                f"Render mode: {env.render_mode}, modes: {render_modes}"
            )

    result = env.render(*args, **kwargs)

    # TODO: Check that the result is correct

    return result
