"""Contains methods for step compatibility, from old-to-new and new-to-old API, to be removed in 1.0."""
from typing import Tuple, Union

import numpy as np

from gym.core import ObsType

OldStepType = Tuple[
    Union[ObsType, np.ndarray],
    Union[float, np.ndarray],
    Union[bool, np.ndarray],
    Union[dict, list],
]

NewStepType = Tuple[
    Union[ObsType, np.ndarray],
    Union[float, np.ndarray],
    Union[bool, np.ndarray],
    Union[bool, np.ndarray],
    Union[dict, list],
]


def step_to_new_api(
    step_returns: Union[OldStepType, NewStepType], is_vector_env=False
) -> NewStepType:
    """Function to transform step returns to new step API irrespective of input API.

    Args:
        step_returns (tuple): Items returned by step(). Can be (obs, rew, done, info) or (obs, rew, terminated, truncated, info)
        is_vector_env (bool): Whether the step_returns are from a vector environment
    """
    if len(step_returns) == 5:
        return step_returns
    else:
        assert len(step_returns) == 4
        observations, rewards, dones, infos = step_returns

        # Cases to handle - info single env /  info vector env (list) / info vector env (dict)
        if is_vector_env is False:
            truncated = infos.pop("TimeLimit.truncated", False)
            return (
                observations,
                rewards,
                dones and not truncated,
                dones and truncated,
                infos,
            )
        elif isinstance(infos, list):
            truncated = np.array(
                [info.pop("TimeLimit.truncated", False) for info in infos]
            )
            return (
                observations,
                rewards,
                np.logical_and(dones, np.logical_not(truncated)),
                np.logical_and(dones, truncated),
                infos,
            )
        elif isinstance(infos, dict):
            num_envs = len(dones)
            truncated = infos.pop("TimeLimit.truncated", np.zeros(num_envs, dtype=bool))
            return (
                observations,
                rewards,
                np.logical_and(dones, np.logical_not(truncated)),
                np.logical_and(dones, truncated),
                infos,
            )
        else:
            raise TypeError(
                f"Unexpected value of infos, as is_vector_envs=False, expects `info` to be a list or dict, actual type: {type(infos)}"
            )


def step_to_old_api(
    step_returns: Union[NewStepType, OldStepType], is_vector_env: bool = False
) -> OldStepType:
    """Function to transform step returns to old step API irrespective of input API.

    Args:
        step_returns (tuple): Items returned by step(). Can be (obs, rew, done, info) or (obs, rew, terminated, truncated, info)
        is_vector_env (bool): Whether the step_returns are from a vector environment
    """
    if len(step_returns) == 4:
        return step_returns
    else:
        assert len(step_returns) == 5
        observations, rewards, terminated, truncated, infos = step_returns

        # Cases to handle - info single env /  info vector env (list) / info vector env (dict)
        if is_vector_env is False:
            if truncated or terminated:
                infos["TimeLimit.truncated"] = truncated and not terminated
            return (
                observations,
                rewards,
                terminated or truncated,
                infos,
            )
        elif isinstance(infos, list):
            for info, env_truncated, env_terminated in zip(
                infos, truncated, terminated
            ):
                if env_truncated or env_terminated:
                    info["TimeLimit.truncated"] = env_truncated and not env_terminated
            return (
                observations,
                rewards,
                np.logical_or(terminated, truncated),
                infos,
            )
        elif isinstance(infos, dict):
            if np.logical_or(np.any(truncated), np.any(terminated)):
                infos["TimeLimit.truncated"] = np.logical_and(
                    truncated, np.logical_not(terminated)
                )
            return (
                observations,
                rewards,
                np.logical_or(terminated, truncated),
                infos,
            )
        else:
            raise TypeError(
                f"Unexpected value of infos, as is_vector_envs=False, expects `info` to be a list or dict, actual type: {type(infos)}"
            )


def step_api_compatibility(
    step_returns: Union[NewStepType, OldStepType],
    new_step_api: bool = False,
    is_vector_env: bool = False,
) -> Union[NewStepType, OldStepType]:
    """Function to transform step returns to the API specified by `new_step_api` bool.

    Old step API refers to step() method returning (observation, reward, done, info)
    New step API refers to step() method returning (observation, reward, terminated, truncated, info)
    (Refer to docs for details on the API change)

    Args:
        step_returns (tuple): Items returned by step(). Can be (obs, rew, done, info) or (obs, rew, terminated, truncated, info)
        new_step_api (bool): Whether the output should be in new step API or old (False by default)
        is_vector_env (bool): Whether the step_returns are from a vector environment

    Returns:
        step_returns (tuple): Depending on `new_step_api` bool, it can return (obs, rew, done, info) or (obs, rew, terminated, truncated, info)

    Examples:
        This function can be used to ensure compatibility in step interfaces with conflicting API. Eg. if env is written in old API,
         wrapper is written in new API, and the final step output is desired to be in old API.

        >>> obs, rew, done, info = step_api_compatibility(env.step(action))
        >>> obs, rew, terminated, truncated, info = step_api_compatibility(env.step(action), new_step_api=True)
        >>> observations, rewards, dones, infos = step_api_compatibility(vec_env.step(action), is_vector_env=True)
    """
    if new_step_api:
        return step_to_new_api(step_returns, is_vector_env)
    else:
        return step_to_old_api(step_returns, is_vector_env)
