import numpy as np

import gym
from gym import logger
from gym.vector.vector_env import VectorEnvWrapper


class StepCompatibilityVector(VectorEnvWrapper):
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
        logger.warn(
            "Using a vector wrapper to transform new step API (which returns two bool vectors terminateds, truncateds) into old (returns one bool vector dones). "
            "This wrapper will be removed in the future. "
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
