from __future__ import annotations

from typing import Optional

from gym import Wrapper, Env
from gym.core import ObsType


class CompatibilityWrapper(Wrapper):
    """
    This wrapper is used to make the environment compatible with the new Gym interface.
    """

    def __init__(self, env: Env):
        super().__init__(env)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        return_info: bool = False,
    ) -> ObsType | tuple[ObsType, dict]:
        # Seed the environment if a seed is provided.
        if seed is not None:
            self.env.seed(seed)

        # Reset the environment. Options are ignored.
        obs = self.env.reset()

        if return_info:
            return obs, {}
        else:
            return obs
