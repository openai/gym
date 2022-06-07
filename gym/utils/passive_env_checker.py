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
                f"It seems that your observation space ({observation_space}) is an image but the `dtype` of your observation_space is not `np.uint8`. "
                "If your observation is not an image, we recommend you to flatten the observation to have only a 1D vector"
            )
        if np.any(observation_space.low != 0) or np.any(
            observation_space.high != 255
        ):  # todo np.all?
            logger.warn(
                "It seems that your observation space is an image but the upper and lower bounds are not in [0, 255]. "
                "Generally, CNN policies assume observations are within that range, so you may encounter an issue if the observation values are not."
            )

    if len(observation_space.shape) not in [1, 3]:
        logger.warn(
            "Your observation space has an unconventional shape (neither an image, nor a 1D vector). "
            "We recommend you to flatten the observation to have only a 1D vector or use a custom policy to properly process the data. "
            f"Observation space={observation_space}"
        )

    if np.any(np.equal(observation_space.low, -np.inf)):
        logger.warn(
            "Agent's minimum observation space value is -infinity. This is probably too low."
        )
    if np.any(np.equal(observation_space.high, np.inf)):
        logger.warn(
            "Agent's maximum observation space value is infinity. This is probably too high."
        )

    if np.any(np.equal(observation_space.low, observation_space.high)):
        logger.warn("Agent's maximum and minimum observation space values are equal")
    if np.any(np.greater(observation_space.low, observation_space.high)):
        raise AssertionError(
            "Agent's minimum observation value is greater than it's maximum"
        )
    if observation_space.low.shape != observation_space.shape:
        raise AssertionError(
            "Agent's observation_space.low and observation_space have different shapes"
        )
    if observation_space.high.shape != observation_space.shape:
        raise AssertionError(
            "Agent's observation_space.high and observation_space have different shapes"
        )


def _check_box_action_space(action_space: spaces.Box):
    """Checks that a :class:`Box` action space is defined in a sensible way.

    Args:
        action_space: A box action space
    """
    if np.any(np.equal(action_space.low, -np.inf)):
        logger.warn(
            "Agent's minimum action space value is -infinity. This is probably too low."
        )
    if np.any(np.equal(action_space.high, np.inf)):
        logger.warn(
            "Agent's maximum action space value is infinity. This is probably too high"
        )
    if np.any(np.equal(action_space.low, action_space.high)):
        logger.warn("Agent's maximum and minimum action space values are equal")
    if np.any(np.greater(action_space.low, action_space.high)):
        raise AssertionError(
            "Agent's minimum action value is greater than it's maximum"
        )
    if action_space.low.shape != action_space.shape:
        raise AssertionError(
            "Agent's action_space.low and action_space have different shapes"
        )
    if action_space.high.shape != action_space.shape:
        raise AssertionError(
            "Agent's action_space.high and action_space have different shapes"
        )

    # Check that the Box space is normalized
    if (
        np.any(np.abs(action_space.low) != np.abs(action_space.high))
        or np.any(np.abs(action_space.low) > 1)
        or np.any(np.abs(action_space.high) > 1)
    ):
        logger.warn(
            "We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) "
            "https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html"  # TODO Add to gymlibrary.ml?
        )


def _check_obs(obs, observation_space: spaces.Space, method_name: str):
    """Check that the observation returned by the environment correspond to the declared one.

    Args:
        obs: The observation to check
        observation_space: The observation space of the observation
        method_name: The method name that generated the observation
    """
    pre = f"The observation returned by the `{method_name}()` method"

    assert observation_space.contains(
        obs
    ), f"{pre} is not contained with the observation space ({observation_space})"

    if isinstance(observation_space, spaces.Discrete):
        assert isinstance(
            obs, int
        ), f"The observation returned by `{method_name}()` method must be an int, actually {type(obs)}"
    elif isinstance(
        observation_space, (spaces.Box, spaces.MultiBinary, spaces.MultiDiscrete)
    ):
        assert isinstance(
            obs, np.ndarray
        ), f"The observation returned by `{method_name}()` method must be a numpy array, actually {type(obs)}"
    elif isinstance(observation_space, spaces.Tuple):
        assert isinstance(
            obs, tuple
        ), f"The observation returned by the `{method_name}()` method must be a tuple, actually {type(obs)}"
        for sub_obs, sub_space in zip(obs, observation_space.spaces):
            _check_obs(sub_obs, sub_space, method_name)
    elif isinstance(observation_space, spaces.Dict):
        assert isinstance(
            obs, dict
        ), f"The observation returned by the `{method_name}()` method must be a dict, actually {type(obs)}"
        for space_key in observation_space.keys():
            _check_obs(obs[space_key], observation_space[space_key], method_name)


def check_observation_space(observation_space):
    """A passive check of the environment observation space that should not affect the environment."""
    if not isinstance(observation_space, spaces.Space):
        raise AssertionError(
            f"Observation space ({observation_space}) does not inherit from gym.spaces.Space"
        )

    elif isinstance(observation_space, spaces.Box):
        # Check if the box is an image (shape is 3 elements and the last element is 1 or 3)
        _check_box_observation_space(observation_space)
    elif isinstance(observation_space, spaces.Discrete):
        assert (
            observation_space.n > 0
        ), f"There are no available discrete observations, n={observation_space.n}"
    elif isinstance(observation_space, spaces.MultiDiscrete):
        assert np.all(
            observation_space.nvec > 0
        ), f"All dimensions of multi-discrete must be greater than 0, {observation_space.nvec}"
    elif isinstance(observation_space, spaces.MultiBinary):
        assert np.all(
            np.asarray(observation_space.shape) > 0
        ), f"All dimensions of multi-binary must be greater than 0, {observation_space.shape}"
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


def check_action_space(action_space):
    """A passive check of the environment action space that should not affect the environment."""
    if not isinstance(action_space, spaces.Space):
        raise AssertionError(
            f"Action space ({action_space}) does not inherit from gym.spaces.Space"
        )

    elif isinstance(action_space, spaces.Box):
        _check_box_action_space(action_space)
    elif isinstance(action_space, spaces.Discrete):
        assert (
            action_space.n > 0
        ), f"There are no available discrete actions, n={action_space.n}"
    elif isinstance(action_space, spaces.MultiDiscrete):
        assert np.all(
            action_space.nvec > 0
        ), f"All dimensions of multi-discrete must be greater than 0, {action_space.nvec}"
    elif isinstance(action_space, spaces.MultiBinary):
        assert np.all(
            np.asarray(action_space.shape) > 0
        ), f"All dimensions of multi-binary must be greater than 0, {action_space.shape}"
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


def passive_env_reset_check(env, **kwargs):
    """A passive check of the `Env.reset` function investigating the returning reset information and returning the data unchanged."""
    signature = inspect.signature(env.reset)
    if "seed" not in signature.parameters or "kwargs" in signature.parameters:
        logger.warn(
            "Future gym versions will require that `Env.reset` can be passed a `seed` instead of using `Env.seed` for resetting the environment random number generator. "
        )
    else:
        seed_param = signature.parameters.get("seed")
        # Check the default value is None
        if seed_param is not None and seed_param.default is not None:
            logger.warn(
                "The default seed argument in `Env.reset` should be `None`, otherwise the environment will by default always be deterministic"
            )

    if "return_info" not in signature.parameters or "kwargs" in signature.parameters:
        logger.warn(
            "Future gym versions will require that `Env.reset` can be passed `return_info` to return information from the environment resetting."
        )

    if "options" not in signature.parameters or "kwargs" in signature.parameters:
        logger.warn(
            "Future gym versions will require that `Env.reset` can be passed `options` to allow the environment initialisation to be passed additional information."
        )

    # Checks the result of env.reset with kwargs
    result = env.reset(**kwargs)
    if "return_info" in kwargs and kwargs["return_info"] is True:
        obs, info = result
        _check_obs(obs, env.observation_space, "reset")
        assert isinstance(
            info, dict
        ), f"The second element returned by `env.reset(return_info=True)` was not a dictionary, actually {type(info)}"
    else:
        obs = result
        _check_obs(obs, env.observation_space, "reset")

    return result


def passive_env_step_check(env, action):
    """A passive check for the environment step, investigating the returning data then returning the data unchanged."""
    result = env.step(action)
    if len(result) == 4:
        obs, reward, done, info = result

        assert isinstance(done, bool), "The `done` signal must be a boolean"
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result

        assert isinstance(terminated, bool), "The `terminated` signal must be a boolean"
        assert isinstance(truncated, bool), "The `truncated` signal must be a boolean"
        assert (
            terminated is False or truncated is False
        ), "Only `terminated` or `truncated` can be true, not both."
    else:
        raise error.Error(
            f"Expected `Env.step` to return a four or five elements, actually {len(result)} elements returned."
        )

    _check_obs(obs, env.observation_space, "step")
    if np.any(np.isnan(obs)):
        logger.warn("Encountered NaN value in observations.")
    if np.any(np.isinf(obs)):
        logger.warn("Encountered inf value in observations.")

    assert isinstance(
        reward, (float, int, np.float32)
    ), "The reward returned by `step()` must be a float"
    if np.any(np.isnan(reward)):
        logger.warn("Encountered NaN value in rewards.")
    if np.any(np.isinf(reward)):
        logger.warn("Encountered inf value in rewards.")

    assert isinstance(
        info, dict
    ), "The `info` returned by `step()` must be a python dictionary"

    return result


def passive_env_render_check(env, **kwargs):
    """A passive check of the `Env.render` that the declared render modes/fps in the metadata of the environment is decleared."""
    render_modes = env.metadata.get("render_modes")
    if render_modes is None:
        logger.warn(
            "No render modes was declared in the environment (env.metadata['render_modes'] is None or not defined), "
            "you may have trouble when calling `.render()`"
        )

    render_fps = env.metadata.get("render_fps")
    # We only require `render_fps` if rendering is actually implemented
    if render_fps is None:
        logger.warn(
            "No render fps was declared in the environment (env.metadata['render_fps'] is None or not defined), "
            "rendering may occur at inconsistent fps"
        )

    return env.render(**kwargs)
