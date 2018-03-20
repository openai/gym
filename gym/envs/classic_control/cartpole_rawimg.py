'''
This environment uses gives the raw image
as output instead of the low level state
values.
'''

from gym.envs.classic_control import cartpole
import numpy as np
import math


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

colors = {
        'black': np.array([0, 0, 0]),
        'blue': np.array([33, 100, 209]),
        'red': np.array([214, 25, 28])
    }


class DrawImage:

    def __init__(self):
        self.height = 210  # Atari-like image size
        self.width = 160
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8) + 255

    def draw(self, cart_pos, pol_angl):
        self.__clear()
        self.__draw_base_line()
        self.__draw_box(int((1.0 + cart_pos/2.4) * (self.width // 2)))
        self.__draw_rod(int((1.0 + cart_pos/2.4) * (self.width // 2)), pol_angl)

        return self.canvas

    # Draw the elements of the image

    def __clear(self):

        self.canvas[:, :, :] = 255

    def __draw_base_line(self):

        start_x = 5
        end_x = self.width - 5

        start_end_y = self.height // 2 # the line is horizontal

        self.canvas[start_end_y, start_x:end_x] = colors['black']

    def __draw_box(self, x):
        y = self.height // 2
        left = x - 20
        right = x + 20
        top = y - 10
        bottom = y + 10

        self.canvas[top:bottom, left:right] = colors['blue']

    def __draw_rod(self, pos, angle):

        def draw_square(x, y):
            self.canvas[y-2:y+2, x-2:x+2] = colors['red']

        rod_length = 40
        x0 = pos
        y0 = self.height // 2 - 10

        x1 = int(x0 + rod_length / 2 * math.sin(angle))
        y1 = int(y0 - rod_length / 2 * math.cos(angle))

        x2 = int(x0 + rod_length * math.sin(angle))
        y2 = int(y0 - rod_length * math.cos(angle))

        draw_square(x0, y0)
        draw_square(x1, y1)
        draw_square(x2, y2)
