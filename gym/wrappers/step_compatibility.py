import gym
from gym import logger


class StepCompatibility(gym.Wrapper):
    r"""A wrapper which can transform an environment from new step API to old and vice-versa.

    Old step API refers to step() method returning (observation, reward, done, info)
    New step API refers to step() method returning (observation, reward, terminated, truncated, info)
    (Refer to docs for details on the API change)

    This wrapper is to be used to ease transition to new API and for backward compatibility. It will be removed in v1.0


    Parameters
    ----------
        env (gym.Env): the env to wrap. Can be in old or new API
        new_step_api (bool): True to use env with new step API, False to use env with old step API. (False by default)

    """

    def __init__(self, env: gym.Env, new_step_api=False):
        super().__init__(env)
        self.new_step_api = new_step_api
        if not self.new_step_api:
            logger.deprecation(
                "Initializing environment in old step API which returns one bool instead of two. "
                "Note that vector API and most wrappers would not work as these have been upgraded to the new API. "
                "To use these features, please set `new_step_api=True` in make to use new API (see docs for more details)."
            )

    def step(self, action):
        step_returns = self.env.step(action)
        if self.new_step_api:
            return step_to_new_api(step_returns)
        else:
            return step_to_old_api(step_returns)


def step_to_new_api(step_returns, is_vector_env=False):
    # Method to transform step returns to new step API

    if len(step_returns) == 5:
        logger.deprecation(
            "Using an environment with new step API that returns two bools terminated, truncated instead of one bool done. "
            "Take care to supporting code to be compatible with this API"
        )
        return step_returns
    else:
        assert len(step_returns) == 4
        logger.deprecation(
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
            terminateds if is_vector_env else terminateds[0],
            truncateds if is_vector_env else truncateds[0],
            infos if is_vector_env else infos[0],
        )


def step_to_old_api(step_returns, is_vector_env=False):
    # Method to transform step returns to old step API

    if len(step_returns) == 4:
        logger.deprecation(
            "Core environment uses old step API which returns one boolean (done). Please upgrade to new API to return two booleans - terminated, truncated"
        )

        return step_returns
    else:
        assert len(step_returns) == 5
        logger.deprecation(
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
            dones if is_vector_env else dones[0],
            infos if is_vector_env else infos[0],
        )


def step_api_compatibility(WrapperClass):
    """
    A step API compatibility wrapper function to transform wrappers in new step API to old
    """

    class StepCompatibilityWrapper(StepCompatibility):
        def __init__(self, env: gym.Wrapper, output_new_step_api: bool = False):
            super().__init__(WrapperClass(env), output_new_step_api)
            if hasattr(WrapperClass, "new_step_api"):
                self.has_new_step_api = WrapperClass.new_step_api
            else:
                self.has_new_step_api = False
            self.wrap = WrapperClass(env)

        def _get_env_step_returns(self, action):
            return (
                step_to_new_api(self.wrap.step(action))
                if self.has_new_step_api
                else step_to_old_api(self.wrap.step(action))
            )

    return StepCompatibilityWrapper


# def check_is_new_api(env: Union[gym.Env, gym.Wrapper]):
#     env_copy = deepcopy(env)
#     env_copy.reset()
#     step_returns = env_copy.step(env_copy.action_space.sample())
#     del env_copy
#     return len(step_returns) == 5
