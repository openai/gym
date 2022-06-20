"""Contains methods for step compatibility, from old-to-new and new-to-old API, to be removed in 1.0."""
from typing import Tuple, Union

import numpy as np

from gym.core import ObsType
from gym.logger import deprecation

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
        deprecation(
            "Transforming code with old step API into new. "
            "It is recommended to upgrade the core env to the new step API. This can also be done by setting `new_step_api=True` at make. "
            "If 'TimeLimit.truncated' is set at truncation, terminated and truncated values will be accurate. "
            "Otherwise, `terminated=done` and `truncated=False`"
        )

        observations, rewards, dones, infos = step_returns

        terminateds = []
        truncateds = []
        if not is_vector_env:
            dones = [dones]

        for i in range(len(dones)):
            # For every condition, handling - info single env /  info vector env (list) / info vector env (dict)

            # TimeLimit.truncated attribute not present - implies either terminated or episode still ongoing based on `done`
            if (not is_vector_env and "TimeLimit.truncated" not in infos) or (
                is_vector_env
                and (
                    (
                        isinstance(infos, list)
                        and "TimeLimit.truncated" not in infos[i]
                    )  # vector env, list info api
                    or (
                        "TimeLimit.truncated" in infos
                        and not infos["_TimeLimit.truncated"][i]
                    )  # vector env, dict info api, if mask is False, it's the same as TimeLimit.truncated attribute not being present for env 'i'
                )
            ):

                terminateds.append(dones[i])
                truncateds.append(False)

            # This means info["TimeLimit.truncated"] exists and is True, which means the truncation has occurred but termination has not.
            elif (
                infos["TimeLimit.truncated"]
                if not is_vector_env
                else (
                    infos["TimeLimit.truncated"][i]
                    if isinstance(infos, dict)
                    else infos[i]["TimeLimit.truncated"]
                )
            ):
                assert dones[i] is True
                terminateds.append(False)
                truncateds.append(True)
            else:
                # This means info["TimeLimit.truncated"] exists but is False, which means the core environment had already terminated,
                # but it also exceeded maximum timesteps at the same step.
                assert dones[i] is True
                terminateds.append(True)
                truncateds.append(True)

            # removing "TimeLimit.truncated" from info
            if isinstance(infos, list):
                infos[i].pop(["TimeLimit.truncated"], None)

        # if info dict vector, can only pop after all envs are processed (also for single env)
        if isinstance(infos, dict):
            infos.pop("TimeLimit.truncated", None)
            infos.pop("TimeLimit.truncated_", None)

        return (
            observations,
            rewards,
            np.array(terminateds, dtype=np.bool_) if is_vector_env else terminateds[0],
            np.array(truncateds, dtype=np.bool_) if is_vector_env else truncateds[0],
            infos,
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
        deprecation(
            "Using old step API which returns one boolean (done). Please upgrade to new API to return two booleans - terminated, truncated"
        )

        return step_returns
    else:
        assert len(step_returns) == 5
        deprecation(
            "Transforming code in new step API (which returns two booleans terminated, truncated) into old (returns one boolean done). "
            "It is recommended to upgrade accompanying code to be compatible with the new API, and use the new API by setting `new_step_api=True`. "
        )

        observations, rewards, terminateds, truncateds, infos = step_returns
        dones = []
        if not is_vector_env:
            terminateds = [terminateds]
            truncateds = [truncateds]

        n_envs = len(terminateds)

        for i in range(n_envs):
            dones.append(terminateds[i] or truncateds[i])
            if truncateds[i]:
                if is_vector_env:
                    # handle vector infos for dict and list
                    if isinstance(infos, dict):
                        if "TimeLimit.truncated" not in infos:
                            # TODO: This should ideally not be done manually and should use vector_env's _add_info()
                            infos["TimeLimit.truncated"] = np.zeros(n_envs, dtype=bool)
                            infos["_TimeLimit.truncated"] = np.zeros(n_envs, dtype=bool)

                        infos["TimeLimit.truncated"][i] = not terminateds[i]
                        infos["_TimeLimit.truncated"][i] = True
                    else:
                        # if vector info is a list
                        infos[i]["TimeLimit.truncated"] = not terminateds[i]
                else:
                    infos["TimeLimit.truncated"] = not terminateds[i]
        return (
            observations,
            rewards,
            np.array(dones, dtype=np.bool_) if is_vector_env else dones[0],
            infos,
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
