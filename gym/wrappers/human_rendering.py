import gym
import pygame
import numpy as np


class HumanRendering(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert (
            "render_modes" in env.metadata
            and "rgb_array" in env.metadata["render_modes"]
        ), "Base environment must support rgb_array rendering"
        assert (
            "human" not in env.metadata["render_modes"]
        ), "Base environment already provides human-mode"
        assert (
            "render_fps" in env.metadata
        ), "Base environment does not specify framerate"

        self.screen_size = None
        self.window = None
        self.clock = None

        metadata = env.metadata
        metadata["render_modes"].append("human")
        self.metadata = metadata

    def render(self, mode="human", **kwargs):
        if mode == "human":
            rgb_array = np.transpose(
                self.env.render(mode="rgb_array", **kwargs), axes=(1, 0, 2)
            )

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
        else:
            return self.env.render(mode=mode, **kwargs)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
        self.env.close()
