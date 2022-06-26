"""A set of functions for passively checking environment implementations."""
import inspect

import numpy as np

from gym import error, logger, spaces


def _check_box_observation_space(observation_space: spaces.Box):
    """Checks that a :class:`Box` observation space is defined in a sensible way.

    Args:
        observation_space: A box observation space
    """
    # Check if the box is an image
    if len(observation_space.shape) == 3:
        if observation_space.dtype != np.uint8:
            logger.warn(
                f"It seems that your observation space is an image but the `dtype` of your observation_space is not `np.uint8`, actual type: {observation_space.dtype}. "
                "If your observation is not an image, we recommend you to flatten the observation to have only a 1D vector"
            )
        if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
            logger.warn(
                "It seems that your observation space is an image but the upper and lower bounds are not in [0, 255]. "
                "Generally, CNN policies assume observations are within that range, so you may encounter an issue if the observation values are not."
            )

    if len(observation_space.shape) not in [1, 3]:
        logger.warn(
            "Your observation space has an unconventional shape (neither an image, nor a 1D vector). "
            "We recommend you to flatten the observation to have only a 1D vector or use a custom policy to properly process the data. "
            f"Actual observation shape: {observation_space.shape}"
        )

    assert (
        observation_space.low.shape == observation_space.shape
    ), f"Agent's observation_space.low and observation_space have different shapes, low shape: {observation_space.low.shape}, box shape: {observation_space.shape}"
    assert (
        observation_space.high.shape == observation_space.shape
    ), f"Agent's observation_space.high and observation_space have different shapes, high shape: {observation_space.high.shape}, box shape: {observation_space.shape}"

    if np.any(np.equal(observation_space.low, observation_space.high)):
        logger.warn("Agent's maximum and minimum observation space values are equal")

    assert np.all(
        observation_space.low <= observation_space.high
    ), "An Agent's minimum observation value is greater than it's maximum"


def check_observation_space(observation_space):
    """A passive check of the environment observation space that should not affect the environment."""
    if not isinstance(observation_space, spaces.Space):
        raise AssertionError(
            f"Observation space does not inherit from `gym.spaces.Space`, actual type: {type(observation_space)}"
        )

    elif isinstance(observation_space, spaces.Box):
        # Check if the box is an image (shape is 3 elements and the last element is 1 or 3)
        _check_box_observation_space(observation_space)
    elif isinstance(observation_space, spaces.Discrete):
        assert (
            observation_space.shape == () and observation_space.n > 0
        ), f"Discrete observation space's number of dimensions must be positive, actual dimensions: {observation_space.n}"
    elif isinstance(observation_space, spaces.MultiDiscrete):
        assert (
            observation_space.shape == observation_space.nvec.shape
        ), f"Expect the MultiDiscrete shape is be equal to nvec.shape, space shape: {observation_space.shape}, nvec shape: {observation_space.nvec.shape}"
        assert np.all(
            observation_space.nvec > 0
        ), f"All dimensions of multi-discrete observation space must be greater than 0, actual shape: {observation_space.nvec}"
    elif isinstance(observation_space, spaces.MultiBinary):
        assert np.all(
            np.asarray(observation_space.shape) > 0
        ), f"All dimensions of multi-binary observation space must be greater than 0, actual shape: {observation_space.shape}"
    elif isinstance(observation_space, spaces.Tuple):
        assert (
            len(observation_space.spaces) > 0
        ), "An empty Tuple observation space is not allowed."
        for subspace in observation_space.spaces:
            check_observation_space(subspace)
    elif isinstance(observation_space, spaces.Dict):
        assert (
            len(observation_space.spaces.keys()) > 0
        ), "An empty Dict observation space is not allowed."
        for subspace in observation_space.values():
            check_observation_space(subspace)


def _check_box_action_space(action_space: spaces.Box):
    """Checks that a :class:`Box` action space is defined in a sensible way.

    Args:
        action_space: A box action space
    """
    assert (
        action_space.low.shape == action_space.shape
    ), f"Agent's action_space.low and action_space have different shapes, low shape: {action_space.low.shape}, box shape: {action_space.shape}"
    assert (
        action_space.high.shape == action_space.shape
    ), f"Agent's action_space.high and action_space have different shapes, high shape: {action_space.high.shape}, box shape: {action_space.shape}"

    if np.any(np.equal(action_space.low, action_space.high)):
        logger.warn("Agent's maximum and minimum action space values are equal")
    assert np.all(
        action_space.low <= action_space.high
    ), "Agent's minimum action value is greater than it's maximum"


def check_action_space(action_space):
    """A passive check of the environment action space that should not affect the environment."""
    if not isinstance(action_space, spaces.Space):
        raise AssertionError(
            f"Action space does not inherit from `gym.spaces.Space`, actual type: {type(action_space)}"
        )

    elif isinstance(action_space, spaces.Box):
        _check_box_action_space(action_space)
    elif isinstance(action_space, spaces.Discrete):
        assert (
            action_space.n > 0
        ), f"Discrete action space's number of dimensions must be positive, actual dimensions: {action_space.n}"
    elif isinstance(action_space, spaces.MultiDiscrete):
        assert np.all(
            np.asarray(action_space.shape) > 0
        ), f"All dimensions of multi-discrete action space must be greater than 0, actual shape: {action_space.shape}"
    elif isinstance(action_space, spaces.MultiBinary):
        assert np.all(
            np.asarray(action_space.shape) > 0
        ), f"All dimensions of multi-binary action space must be greater than 0, actual shape: {action_space.shape}"
    elif isinstance(action_space, spaces.Tuple):
        assert (
            len(action_space.spaces) > 0
        ), "An empty Tuple action space is not allowed."
        for subspace in action_space.spaces:
            check_action_space(subspace)
    elif isinstance(action_space, spaces.Dict):
        assert (
            len(action_space.spaces.keys()) > 0
        ), "An empty Dict action space is not allowed."
        for subspace in action_space.values():
            check_action_space(subspace)


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
            logger.warn(
                f"{pre} was expecting an int or np.int64, actually type: {type(obs)}"
            )
    elif isinstance(observation_space, spaces.Box):
        if observation_space.shape != ():
            if not isinstance(obs, np.ndarray):
                logger.warn(
                    f"{pre} was expecting a numpy array, actually type: {type(obs)}"
                )
    elif isinstance(observation_space, (spaces.MultiBinary, spaces.MultiDiscrete)):
        if not isinstance(obs, np.ndarray):
            logger.warn(
                f"{pre} was expecting a numpy array, actually type: {type(obs)}"
            )
    elif isinstance(observation_space, spaces.Tuple):
        if not isinstance(obs, tuple):
            logger.warn(f"{pre} was expecting a tuple, actually type: {type(obs)}")
        assert len(obs) == len(
            observation_space.spaces
        ), f"{pre} length is not same as the observation space length, obs length: {len(obs)}, space length: {len(observation_space.spaces)}"
        for sub_obs, sub_space in zip(obs, observation_space.spaces):
            check_obs(sub_obs, sub_space, method_name)
    elif isinstance(observation_space, spaces.Dict):
        assert isinstance(obs, dict), f"{pre} must be a dict, actually {type(obs)}"
        assert (
            obs.keys() == observation_space.spaces.keys()
        ), f"{pre} observation keys is not same as the observation space keys, obs keys: {list(obs.keys())}, space keys: {list(observation_space.spaces.keys())}"
        for space_key in observation_space.spaces.keys():
            check_obs(obs[space_key], observation_space[space_key], method_name)

    try:
        if obs not in observation_space:
            logger.warn(f"{pre} is not within the observation space")
    except Exception as e:
        logger.warn(f"{pre} is not within the observation space with exception: {e}")


def passive_env_reset_checker(env, **kwargs):
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

    if (
        "return_info" not in signature.parameters
        and "kwargs" not in signature.parameters
    ):
        logger.warn(
            "Future gym versions will require that `Env.reset` can be passed `return_info` to return information from the environment resetting."
        )

    if "options" not in signature.parameters and "kwargs" not in signature.parameters:
        logger.warn(
            "Future gym versions will require that `Env.reset` can be passed `options` to allow the environment initialisation to be passed additional information."
        )

    # Checks the result of env.reset with kwargs
    result = env.reset(**kwargs)
    if "return_info" in kwargs and kwargs["return_info"] is True:
        assert isinstance(
            result, tuple
        ), f"The result returned by `env.reset(return_info=True)` was not a tuple, actually type: {type(result)}"
        obs, info = result
        assert isinstance(
            info, dict
        ), f"The second element returned by `env.reset(return_info=True)` was not a dictionary, actually type: {type(info)}"
    else:
        obs = result

    check_obs(obs, env.observation_space, "reset")
    return result


def passive_env_step_checker(env, action):
    """A passive check for the environment step, investigating the returning data then returning the data unchanged."""
    # We don't check the action as for some environments then out-of-bounds values can be given
    result = env.step(action)
    assert isinstance(
        result, tuple
    ), f"Expects step result to be a tuple, actual type: {type(result)}"
    if len(result) == 4:
        obs, reward, done, info = result

        assert isinstance(
            done, (bool, np.bool_)
        ), f"The `done` signal must be a boolean, actual type: {type(done)}"
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result

        # np.bool is actual python bool not np boolean type, therefore bool_ or bool8
        if not isinstance(terminated, (bool, np.bool_)):
            logger.warn(
                f"The `terminated` signal must be a boolean, actual type: {type(terminated)}"
            )
        if not isinstance(truncated, (bool, np.bool_)):
            logger.warn(
                f"The `truncated` signal must be a boolean, actual type: {type(truncated)}"
            )
    else:
        raise error.Error(
            f"Expected `Env.step` to return a four or five element tuple, actually number of elements returned: {len(result)}."
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


def passive_env_render_checker(env, *args, **kwargs):
    """A passive check of the `Env.render` that the declared render modes/fps in the metadata of the environment is declared."""
    render_modes = env.metadata.get("render_modes")
    if render_modes is None:
        logger.warn(
            "No render modes was declared in the environment "
            "(env.metadata['render_modes'] is None or not defined), "
            "you may have trouble when calling `.render()`."
        )
    else:
        if not isinstance(render_modes, (list, tuple)):
            logger.warn(
                f"Expects the render_modes to be a sequence (i.e. list, tuple), actual type: {type(render_modes)}"
            )
        if not all(isinstance(mode, str) for mode in render_modes):
            logger.warn(
                f"Expects all render modes to be strings, actual types: {[type(mode) for mode in render_modes]}."
            )

        render_fps = env.metadata.get("render_fps")
        # We only require `render_fps` if rendering is actually implemented
        if len(render_modes) > 0:
            if render_fps is None:
                logger.warn(
                    "No render fps was declared in the environment "
                    "(env.metadata['render_fps'] is None or not defined), "
                    "rendering may occur at inconsistent fps."
                )
            else:
                if not isinstance(render_fps, int):
                    logger.warn(
                        f"Expects the `env.metadata['render_fps']` to be an integer, actual type: {type(render_fps)}."
                    )
                else:
                    assert (
                        render_fps > 0
                    ), f"Expects the `env.metadata['render_fps']` to be greater than zero, actual value: {render_fps}."

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

    return env.render(*args, **kwargs)
