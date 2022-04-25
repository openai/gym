import numpy as np

import gym
from gym import logger
from gym.vector.vector_env import VectorEnvWrapper


class StepCompatibilityVector(VectorEnvWrapper):
    r"""A wrapper which can transform a vector environment to a new or old step API.

    Old step API refers to step() method returning (observation, reward, done, info)
    New step API refers to step() method returning (observation, reward, terminated, truncated, info)
    (Refer to docs for details on the API change)

    This wrapper is to be used to ease transition to new API. It will be removed in v1.0

    Parameters
    ----------
        env (gym.vector.VectorEnv): the vector env to wrap. Has to be in new step API
        return_two_dones (bool): True to use vector env with new step API, False to use vector env with old step API. (True by default)

    """

    def __init__(self, env, return_two_dones=True):
        super().__init__(env)
        self._return_two_dones = return_two_dones

    def step_wait(self):
        step_returns = self.env.step_wait()
        if self._return_two_dones:
            return step_returns
        else:
            return self._step_returns_new_to_old(step_returns)

    def _step_returns_new_to_old(self, step_returns):
        assert len(step_returns) == 5
        observations, rewards, terminateds, truncateds, infos = step_returns
        logger.deprecation(
            "Using a vector wrapper to transform new step API (which returns two bool vectors terminateds, truncateds) into old (returns one bool vector dones). "
            "This wrapper will be removed in v1.0. "
            "It is recommended to upgrade your accompanying code instead to be compatible with the new API, and use the new API. "
        )
        dones = []
        for i in range(len(terminateds)):
            dones.append(terminateds[i] or truncateds[i])
            if truncateds[i]:
                infos[i]["TimeLimit.truncated"] = not terminateds[i]
        return observations, rewards, np.array(dones, dtype=np.bool_), infos

    def __del__(self):
        self.env.__del__()
