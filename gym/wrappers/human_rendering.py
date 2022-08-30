"""A wrapper that adds human-renering functionality to an environment."""
import numpy as np

import gym
from gym.error import DependencyNotInstalled


class HumanRendering(gym.Wrapper):
    """Performs human rendering for an environment that only supports "rgb_array"rendering.

    This wrapper is particularly useful when you have implemented an environment that can produce
    RGB images but haven't implemented any code to render the images to the screen.
    If you want to use this wrapper with your environments, remember to specify ``"render_fps"``
    in the metadata of your environment.

    The ``render_mode`` of the wrapped environment must be either ``'rgb_array'`` or ``'rgb_array_list'``.

    Example:
        >>> env = gym.make("LunarLander-v2", render_mode="rgb_array")
        >>> wrapped = HumanRendering(env)
        >>> wrapped.reset()     # This will start rendering to the screen

    The wrapper can also be applied directly when the environment is instantiated, simply by passing
    ``render_mode="human"`` to ``make``. The wrapper will only be applied if the environment does not
    implement human-rendering natively (i.e. ``render_mode`` does not contain ``"human"``).

    Example:
        >>> env = gym.make("NoNativeRendering-v2", render_mode="human")      # NoNativeRendering-v0 doesn't implement human-rendering natively
        >>> env.reset()     # This will start rendering to the screen

    Warning: If the base environment uses ``render_mode="rgb_array_list"``, its (i.e. the *base environment's*) render method
        will always return an empty list:

            >>> env = gym.make("LunarLander-v2", render_mode="rgb_array_list")
            >>> wrapped = HumanRendering(env)
            >>> wrapped.reset()
            >>> env.render()
            []          # env.render() will always return an empty list!

    """

    def __init__(self, env):
        """Initialize a :class:`HumanRendering` instance.

        Args:
            env: The environment that is being wrapped
        """
        super().__init__(env)
        assert env.render_mode in [
            "rgb_array",
            "rgb_array_list",
        ], f"Expected env.render_mode to be one of 'rgb_array' or 'rgb_array_list' but got '{env.render_mode}'"
        assert (
            "render_fps" in env.metadata
        ), "The base environment must specify 'render_fps' to be used with the HumanRendering wrapper"

        self.screen_size = None
        self.window = None
        self.clock = None

    @property
    def render_mode(self):
        """Always returns ``'human'``."""
        return "human"

    def step(self, *args, **kwargs):
        """Perform a step in the base environment and render a frame to the screen."""
        result = self.env.step(*args, **kwargs)
        self._render_frame()
        return result

    def reset(self, *args, **kwargs):
        """Reset the base environment and render a frame to the screen."""
        result = self.env.reset(*args, **kwargs)
        self._render_frame()
        return result

    def render(self):
        """This method doesn't do much, actual rendering is performed in :meth:`step` and :meth:`reset`."""
        return None

    def _render_frame(self):
        """Fetch the last frame from the base environment and render it to the screen."""
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[box2d]`"
            )
        if self.env.render_mode == "rgb_array_list":
            last_rgb_array = self.env.render()
            assert isinstance(last_rgb_array, list)
            last_rgb_array = last_rgb_array[-1]
        elif self.env.render_mode == "rgb_array":
            last_rgb_array = self.env.render()
        else:
            raise Exception(
                f"Wrapped environment must have mode 'rgb_array' or 'rgb_array_list', actual render mode: {self.env.render_mode}"
            )
        assert isinstance(last_rgb_array, np.ndarray)

        rgb_array = np.transpose(last_rgb_array, axes=(1, 0, 2))

        if self.screen_size is None:
            self.screen_size = rgb_array.shape[:2]

        assert (
            self.screen_size == rgb_array.shape[:2]
        ), f"The shape of the rgb array has changed from {self.screen_size} to {rgb_array.shape[:2]}"

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.screen_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(rgb_array)
        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def close(self):
        """Close the rendering window."""
        super().close()
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
