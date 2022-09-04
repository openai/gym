"""A wrapper that adds render collection mode to an environment."""
import numpy as np

import gym
from gym.error import DependencyNotInstalled


class RenderCollection(gym.Wrapper):
    """Save collection of render frames."""

    def __init__(self, env: gym.Env, pop_frames: bool = True, reset_clean: bool = True):
        super().__init__(env)
        assert env.render_mode is not None
        assert not env.render_mode.endswith("_list")
        self.frame_list = []
        self.reset_clean = reset_clean
        self.pop_frames = pop_frames

    @property
    def render_mode(self):
        return f"{self.env.render_mode}_list"

    def step(self, *args, **kwargs):
        output = self.env.step(*args, **kwargs)
        self.frame_list.append(self.env.render())
        return output

    def reset(self, *args, **kwargs):
        """Reset the base environment and render a frame to the screen."""
        result = self.env.reset(*args, **kwargs)

        if self.reset_clean:
            self.frame_list = []
        self.frame_list.append(self.env.render())

        return result

    def render(self):
        frames = self.frame_list
        if self.pop_frames:
            self.frame_list = []

        return frames
