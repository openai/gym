from gym.envs.classic_control import acrobot
import numpy as np


class AcrobotRawImgEnv(acrobot.AcrobotEnv):

    def __init__(self):
        super().__init__()
        self.drawer = DrawImage()
        self.raw_img = None

    def step(self, action):
        obs, rw, done, inf = super().step(action)
        cos_alpha = obs[0]  # the first angle is for the upper arm
        sin_alpha = obs[1]
        cos_beta = obs[2]
        sin_beta = obs[3]

        self.raw_img = self.drawer.draw(cos_alpha, sin_alpha, cos_beta, sin_beta)
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
        cos_alpha = obs[0]
        sin_alpha = obs[1]
        cos_beta = obs[2]
        sin_beta = obs[3]

        self.raw_img = self.drawer.draw(cos_alpha, sin_alpha, cos_beta, sin_beta)
        return self.raw_img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def __del__(self):
        self.close()

colors = {
        'black': np.array([0, 0, 0]),
        'brown': np.array([173, 97, 14]),
        'red': np.array([214, 25, 28]),
    }


class DrawImage:

    def __init__(self):
        self.height = 210  # Atari-like image size
        self.width = 160
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8) + 255

        self.rod_length = 40

    def draw(self, cos_alpha, sin_alpha, cos_beta, sin_beta):
        self.__clear()
        self.__draw_base_line()
        self.__draw_rod(cos_alpha, sin_alpha, cos_beta, sin_beta)

        return self.canvas

    # Draw the elements of the image

    def __clear(self):

        self.canvas[:, :, :] = 255

    def __draw_base_line(self):

        start_x = 5
        end_x = self.width - 5

        start_end_y = self.height // 2 - self.rod_length  # the line is horizontal

        self.canvas[start_end_y, start_x:end_x] = colors['black']

    def __draw_rod(self, cos_alpha, sin_alpha, cos_beta, sin_beta):

        def draw_square(x, y, color):
            self.canvas[y-2:y+2, x-2:x+2] = color

        x0 = self.width // 2
        y0 = self.height // 2

        x1 = int(x0 + self.rod_length / 2 * sin_alpha)
        y1 = int(y0 + self.rod_length / 2 * cos_alpha)

        x2 = int(x0 + self.rod_length * sin_alpha)
        y2 = int(y0 + self.rod_length * cos_alpha)

        sin_alpha_plus_beta = sin_alpha * cos_beta + cos_alpha * sin_beta
        cos_alpha_plus_beta = cos_alpha * cos_beta - sin_alpha * sin_beta

        x3 = int(x2 + self.rod_length / 2 * sin_alpha_plus_beta)
        y3 = int(y2 + self.rod_length / 2 * cos_alpha_plus_beta)

        x4 = int(x2 + self.rod_length * sin_alpha_plus_beta)
        y4 = int(y2 + self.rod_length * cos_alpha_plus_beta)

        draw_square(x0, y0, colors['brown'])
        draw_square(x1, y1, colors['red'])
        draw_square(x2, y2, colors['brown'])
        draw_square(x3, y3, colors['red'])
        draw_square(x4, y4, colors['red'])
