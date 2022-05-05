import numpy as np

from gym.logger import deprecation


def step_to_new_api(step_returns, is_vector_env=False):
    """Function to transform step returns to new step API irrespective of input API

    Args:
        step_returns (tuple): Items returned by step(). Can be (obs, rew, done, info) or (obs, rew, terminated, truncated, info)
        is_vector_env (bool): Whether the step_returns are from a vector environment
    """

    if len(step_returns) == 5:
        deprecation(
            "Using an environment with new step API that returns two bools terminated, truncated instead of one bool done. "
            "Take care to supporting code to be compatible with this API"
        )
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
            infos = [infos]
        for i in range(len(dones)):
            if "TimeLimit.truncated" not in infos[i]:
                terminateds.append(dones[i])
                truncateds.append(False)
            elif infos[i]["TimeLimit.truncated"]:
                terminateds.append(False)
                truncateds.append(True)
            else:
                # This means info["TimeLimit.truncated"] exists but is False, which means the core environment had already terminated,
                # but it also exceeded maximum timesteps at the same step.

                terminateds.append(True)
                truncateds.append(True)

        return (
            observations,
            rewards,
            np.array(terminateds, dtype=np.bool_) if is_vector_env else terminateds[0],
            np.array(truncateds, dtype=np.bool_) if is_vector_env else truncateds[0],
            infos if is_vector_env else infos[0],
        )


def step_to_old_api(step_returns, is_vector_env=False):
    """Function to transform step returns to old step API irrespective of input API

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
            infos = [infos]

        for i in range(len(terminateds)):
            dones.append(terminateds[i] or truncateds[i])
            # to be consistent with old API
            if truncateds[i]:
                infos[i]["TimeLimit.truncated"] = not terminateds[i]
        return (
            observations,
            rewards,
            np.array(dones, dtype=np.bool_) if is_vector_env else dones[0],
            infos if is_vector_env else infos[0],
        )


def step_api_compatibility(
    step_returns, new_step_api: bool = False, is_vector_env: bool = False
):
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
