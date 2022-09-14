"""A compatibility wrapper converting an old-style environment into a valid environment."""
import sys
from typing import Any, Dict, Optional, Tuple

import gym
from gym.core import ObsType
from gym.utils.step_api_compatibility import convert_to_terminated_truncated_step_api

if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
elif sys.version_info >= (3, 7):
    from typing_extensions import Protocol, runtime_checkable
else:
    Protocol = object
    runtime_checkable = lambda x: x  # noqa: E731


@runtime_checkable
class LegacyEnv(Protocol):
    """A protocol for environments using the old step API."""

    observation_space: gym.Space
    action_space: gym.Space

    def reset(self) -> Any:
        """Reset the environment and return the initial observation."""
        ...

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Run one timestep of the environment's dynamics."""
        ...

    def render(self, mode: Optional[str] = "human") -> Any:
        """Render the environment."""
        ...

    def close(self):
        """Close the environment."""
        ...

    def seed(self, seed: Optional[int] = None):
        """Set the seed for this env's random number generator(s)."""
        ...


class EnvCompatibility(gym.Env):
    r"""A wrapper which can transform an environment from the old API to the new API.

    Old step API refers to step() method returning (observation, reward, done, info), and reset() only retuning the observation.
    New step API refers to step() method returning (observation, reward, terminated, truncated, info) and reset() returning (observation, info).
    (Refer to docs for details on the API change)

    Known limitations:
    - Environments that use `self.np_random` might not work as expected.
    """

    def __init__(self, old_env: LegacyEnv, render_mode: Optional[str] = None):
        """A wrapper which converts old-style envs to valid modern envs.

        Some information may be lost in the conversion, so we recommend updating your environment.

        Args:
            old_env (LegacyEnv): the env to wrap, implemented with the old API
            render_mode (str): the render mode to use when rendering the environment, passed automatically to env.render
        """
        self.metadata = getattr(old_env, "metadata", {"render_modes": []})
        self.render_mode = render_mode
        self.reward_range = getattr(old_env, "reward_range", None)
        self.spec = getattr(old_env, "spec", None)
        self.env = old_env

        self.observation_space = old_env.observation_space
        self.action_space = old_env.action_space

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        """Resets the environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            (observation, info)
        """
        if seed is not None:
            self.env.seed(seed)
        # Options are ignored

        if self.render_mode == "human":
            self.render()

        return self.env.reset(), {}

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Steps through the environment.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        obs, reward, done, info = self.env.step(action)

        if self.render_mode == "human":
            self.render()

        return convert_to_terminated_truncated_step_api((obs, reward, done, info))

    def render(self) -> Any:
        """Renders the environment.

        Returns:
            The rendering of the environment, depending on the render mode
        """
        return self.env.render(mode=self.render_mode)

    def close(self):
        """Closes the environment."""
        self.env.close()

    def __str__(self):
        """Returns the wrapper name and the unwrapped environment string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)
