'''
This environment uses gives the raw image
as output instead of the low level state
values.
'''

from gym.envs.classic_control import cartpole
import numpy as np


class CartPoleRawImgEnv(cartpole.CartPoleEnv):

    def __init__(self):
        super().__init__()
        self.drawer = DrawImage()
        self.raw_img = None

    def step(self, action):
        obs, rw, done, inf = super().step(action)
        cart_position = obs[0]
        pole_angle = obs[2]

        self.raw_img = self.drawer.draw(cart_position, pole_angle)
        return self.raw_img, rw, done, inf

    def render(self, mode='human'):

        if mode == 'rgb_array':
            return self.raw_img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self.raw_img)
            return self.viewer.isopen

    def reset(self):
        obs = super().reset()
        cart_position = obs[0]
        pole_angle = obs[2]

        self.raw_img = self.drawer.draw(cart_position, pole_angle)
        return self.raw_img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def __del__(self):
        self.close()


class DrawImage:

    def __init__(self):
        self.height = 210  # Atari-like image size
        self.width = 160
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8) + 255

    def draw(self, cart_pos, pol_angl):

        return self.canvas

