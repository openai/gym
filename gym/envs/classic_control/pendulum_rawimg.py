from gym.envs.classic_control import pendulum
import numpy as np
import math


class PendulumRawImgEnv(pendulum.PendulumEnv):

    def __init__(self):
        super().__init__()
        self.drawer = DrawImage()
        self.raw_img = None

    def step(self, action):
        obs, rw, done, inf = super().step(action)
        cos_theta = obs[0]
        sin_theta = obs[1]

        self.raw_img = self.drawer.draw(cos_theta, sin_theta)
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
        cos_theta = obs[0]
        sin_theta = obs[1]

        self.raw_img = self.drawer.draw(cos_theta, sin_theta)
        return self.raw_img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def __del__(self):
        self.close()

colors = {
        'black': np.array([0, 0, 0]),
        'red': np.array([214, 25, 28])
    }


class DrawImage:

    def __init__(self):
        self.height = 210  # Atari-like image size
        self.width = 160
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8) + 255

    def draw(self, cos_theta, sin_theta):
        self.__clear()
        self.__draw_rod(cos_theta, sin_theta)

        return self.canvas

    # Draw the elements of the image

    def __clear(self):

        self.canvas[:, :, :] = 255

    def __draw_rod(self, cos_theta, sin_theta):

        def draw_square(x, y):
            self.canvas[y-2:y+2, x-2:x+2] = colors['red']

        rod_length = 60
        x0 = self.width // 2
        y0 = self.height // 2

        x1 = int(x0 + rod_length / 3 * sin_theta)
        y1 = int(y0 - rod_length / 3 * cos_theta)

        x2 = int(x0 + 2* rod_length / 3 * sin_theta)
        y2 = int(y0 - 2* rod_length / 3 * cos_theta)

        x3 = int(x0 + rod_length * sin_theta)
        y3 = int(y0 - rod_length * cos_theta)

        draw_square(x0, y0)
        draw_square(x1, y1)
        draw_square(x2, y2)
        draw_square(x3, y3)
