import numpy as np

from gym.logger import deprecation


def step_to_new_api(step_returns, is_vector_env=False):
    # Method to transform step returns to new step API

    if len(step_returns) == 5:
        deprecation(
            "Using an environment with new step API that returns two bools terminated, truncated instead of one bool done. "
            "Take care to supporting code to be compatible with this API"
        )
        return step_returns
    else:
        assert len(step_returns) == 4
        deprecation(
            "Using a wrapper to transform env with old step API into new. This wrapper will be removed in v1.0. "
            "It is recommended to upgrade the core env to the new step API."
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
    # Method to transform step returns to old step API

    if len(step_returns) == 4:
        deprecation(
            "Core environment uses old step API which returns one boolean (done). Please upgrade to new API to return two booleans - terminated, truncated"
        )

        return step_returns
    else:
        assert len(step_returns) == 5
        deprecation(
            "Using a wrapper to transform new step API (which returns two booleans terminated, truncated) into old (returns one boolean done). "
            "This wrapper will be removed in v1.0 "
            "It is recommended to upgrade your accompanying code instead to be compatible with the new API, and use the new API. "
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
    if new_step_api:
        return step_to_new_api(step_returns, is_vector_env)
    else:
        return step_to_old_api(step_returns, is_vector_env)
