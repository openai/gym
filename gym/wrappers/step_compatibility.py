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
        return_two_dones (bool): True to use env with new step API, False to use env with old step API. (False by default)

    """

    def __init__(self, env, return_two_dones=False):
        super().__init__(env)
        self._return_two_dones = return_two_dones
        if not self._return_two_dones:
            logger.deprecation(
                "Initializing environment in old step API which returns one bool instead of two. "
                "Note that vector API and most wrappers would not work as these have been upgraded to the new API. "
                "To use these features, please set `return_two_dones=True` in make to use new API (see docs for more details)."
            )

    def step(self, action):
        step_returns = self.env.step(action)
        if self._return_two_dones:
            if len(step_returns) == 5:
                logger.deprecation(
                    "Using an environment with new step API that returns two bools terminated, truncated instead of one bool done. "
                    "Take care to supporting code to be compatible with this API"
                )
                return step_returns
            else:
                return self._step_returns_old_to_new(step_returns)
        else:
            if len(step_returns) == 4:
                logger.deprecation(
                    "Core environment uses old step API which returns one boolean (done). Please upgrade to new API to return two booleans - terminated, truncated"
                )

                return step_returns
            elif len(step_returns) == 5:
                return self._step_returns_new_to_old(step_returns)

    def _step_returns_old_to_new(self, step_returns):
        # Method to transform old step API to new

        assert len(step_returns) == 4
        logger.deprecation(
            "Using a wrapper to transform env with old step API into new. This wrapper will be removed in v1.0. "
            "It is recommended to upgrade the core env to the new step API."
            "If 'TimeLimit.truncated' is set at truncation, terminated and truncated values will be accurate. "
            "Otherwise, `terminated=done` and `truncated=False`"
        )

        obs, rew, done, info = step_returns
        if "TimeLimit.truncated" not in info:
            terminated = done
            truncated = False
        elif info["TimeLimit.truncated"]:
            terminated = False
            truncated = True
        else:
            # This means info["TimeLimit.truncated"] exists but is False, which means the core environment had already terminated,
            # but it also exceeded maximum timesteps at the same step.

            terminated = True
            truncated = True

        return obs, rew, terminated, truncated, info

    def _step_returns_new_to_old(self, step_returns):
        # Method to transform new step API to old

        assert len(step_returns) == 5
        logger.deprecation(
            "Using a wrapper to transform new step API (which returns two booleans terminated, truncated) into old (returns one boolean done). "
            "This wrapper will be removed in v1.0 "
            "It is recommended to upgrade your accompanying code instead to be compatible with the new API, and use the new API. "
        )

        obs, reward, terminated, truncated, info = step_returns
        done = terminated or truncated
        if truncated:
            info[
                "TimeLimit.truncated"
            ] = not terminated  # to be consistent with old API
        return obs, reward, done, info
