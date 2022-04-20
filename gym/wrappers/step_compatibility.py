import gym
from gym import logger


class StepCompatibility(gym.Wrapper):
    def __init__(self, env, return_two_dones=False):
        super().__init__(env)
        self._return_two_dones = return_two_dones
        if not self._return_two_dones:
            logger.warn(
                "Initializing environment in old step API which returns one bool instead of two. "
                "Note that vector API and most wrappers would not work as these have been upgraded to the new API. "
                "To use these features, please set `return_two_dones=True` in make to use new API (see docs for more details)."
            )

    def step(self, action):
        step_returns = self.env.step(action)
        if self._return_two_dones:
            if len(step_returns) == 5:
                logger.warn(
                    "Using an environment with new step API that returns two bools terminated, truncated instead of one bool done. "
                    "Take care to update supporting code to be compatible with this API"
                )
                return step_returns
            else:
                return self._step_returns_old_to_new(step_returns)
        else:
            if len(step_returns) == 4:
                logger.warn(
                    "Core environment uses old step API which returns one boolean (done). Please upgrade to new API to return two booleans - terminated, truncated"
                )

                return step_returns
            elif len(step_returns) == 5:
                return self._step_returns_new_to_old(step_returns)

    def _step_returns_old_to_new(self, step_returns):
        assert len(step_returns) == 4
        logger.warn(
            "Using a wrapper to transform env with old step API into new. This wrapper will be removed in the future. "
            "It is recommended to upgrade the core env to the new step API."
            "If 'TimeLimit.truncated' is set at truncation, terminated and truncated values will be accurate"
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
        assert len(step_returns) == 5
        logger.warn(
            "Using a wrapper to transform new step API (which returns two booleans terminated, truncated) into old (returns one boolean done). "
            "This wrapper will be removed in the future. "
            "It is recommended to upgrade your accompanying code instead to be compatible with the new API, and use the new API. "
        )

        obs, reward, terminated, truncated, info = step_returns
        done = terminated or truncated
        if truncated:
            info[
                "TimeLimit.truncated"
            ] = not terminated  # to be consistent with old API
        return obs, reward, done, info
