"""
Temperature sensitive Robot moving through a corridor, taken from Uther and Veloso 1998.
source: http://www.cs.cmu.edu/~mmv/papers/will-aaai98.pdf
"""

import logging

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

logger = logging.getLogger(__name__)


class RobotTemperature(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        # move back/fwd, temp heat/none/cool
        self.action_space = spaces.MultiDiscrete([[0, 1], [0, 2]])
        # position, temp
        self.corridor_len = 100
        self.observation_space = spaces.MultiDiscrete([[0, 10], [0, self.corridor_len]])
        self._seed()
        self.state = [0, 0]
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        move = action[0]
        temp_control = action[1]

        if self.state[0] < 33:
            corridor_temp = np.random.randint(5, 35)
        elif self.state[0] < 66:
            corridor_temp = np.random.randint(25, 75)
        else:
            corridor_temp = np.random.randint(65, 95)

        if temp_control == 0:  # cool
            self.state[1] = corridor_temp + np.random.randint(-45, -25)
        elif temp_control == 1:
            self.state[1] = corridor_temp
        elif temp_control == 2:
            self.state[1] = corridor_temp - np.random.randint(25, 45)

        # compute how likely it is we move
        if 30 < self.state[1] < 70:
            chance_of_move = 0.9
        else:
            chance_of_move = 0.1

        # move (or don't)
        r = np.random.rand()
        if r <= chance_of_move:
            if move == 1:
                self.state[0] += 1
            elif move == 0 and self.state[0] > 0:
                self.state[0] -= 1

        # check if we're done and handle reward
        if self.state[0] == self.corridor_len:
            done = True
            reward = 1
        else:
            done = False
            reward = -1

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state[0] = np.random.randint(0, self.corridor_len)
        self.state[1] = 50
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 200
        screen_height = 400
        robot_s = 40

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -robot_s / 2, robot_s / 2, robot_s / 2, -robot_s / 2
            self.robot = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.robot_trans = rendering.Transform()
            self.robot_color = self.robot.attrs[0]
            self.robot.add_attr(self.robot_trans)
            self.viewer.add_geom(self.robot)

            self.track = rendering.Line((screen_width / 2, 0), (screen_width / 2, screen_height))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        roboty = screen_height + robot_s / 2 - (self.state[0] * screen_height / self.corridor_len)
        self.robot_trans.set_translation(screen_width / 2, roboty)

        if self.state[1] > 50:
            r = (self.state[1] - 50) / 50
            b = 0
        else:
            r = 0
            b = self.state[1] / 50

        self.robot_color.vec4 = (r, 0, b, 1.0)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))
