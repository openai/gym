"""
    # Rubik layout:

            2:U

        0:L   1:F   3:R

            5:D

            4:B

    # Actions

        TODO: Explain how actions are given

        This functions might give some light

        RubikActionSpace.contains
        RubikActionSpace.sample


    # Reference:

        https://ruwix.com/the-rubiks-cube/notation/
        https://ruwix.com/puzzle-scramble-generator/?type=rubiks-cube
"""
import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding

import copy
import io
import re

try:
    import xtermcolor
    def get_color(*args, **kwargs):
        return xtermcolor.colorize(*args, **kwargs)
except ModuleNotFoundError:
    logger.warn("xtermcolor not found. Install it for better rendering.")
    def get_color(text, **kwargs):
        return text

notation_re = r"^([LFURBD])(')?([1-3])?(-([0-9]*))?$"

LEFT = 0
FRONT = 1
UP = 2
RIGHT = 3
BACK = 4
DOWN = 5

COLORS = (
    (0x00ff00, 'G'),
    (0xff0000, 'R'),
    (0xffffff, 'W'),
    (0x0000ff, 'B'),
    (0xff6e00, 'O'),
    (0xffff00, 'Y'),
)

class RubikEnv(gym.Env):
    notation_automata = re.compile(notation_re)

    metadata = {
        'render.modes': ['ansi'],
    }

    reward_range = (-1, 0)

    def __init__(self, size=3):
        self.size = size
        self.cube = np.arange(6).repeat(size**2).reshape(6, size, size)

        self.action_space = RubikActionSpace(size)
        self.observation_space = RubikPermutationSpace(size)

        self.seed()


    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        return [seed]


    def get_action(self, notation):
        # TODO: Check that builded action is correct

        parsed = RubikEnv.notation_automata.search(notation)
        opposite = {
            'D' : 'U',
            'R' : 'L',
            'B' : 'F'
        }

        if parsed is None:
            raise ValueError("Invalid notation")

        axis = parsed[0]
        reverse_rotation = parsed[1] == "'"
        rotation = 1 if parsed[2] is None else int(parsed[2])
        layer = 0 if parsed[4] is None else int(parsed[4])

        if axis in opposite:
            axis = opposite[axis]
            reverse_rotation = not reverse_rotation
            layer = self.size - layer - 1

        if reverse_rotation:
            rotation = -rotation

        return axis, layer, rotation


    def step(self, action):
        if isinstance(action, str):
            action = self.get_action(action)

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        axis, layer, rotation = action

        if rotation == -1:
            rotation = 3

        border = self.size - 1

        if axis == 'L':
            if layer == 0:
                self.rotate(LEFT, -rotation)
            elif layer == self.size - 1:
                self.rotate(RIGHT, rotation)

            for _ in range(rotation):
                self.shift([
                    (UP, 0, layer, +1, 0),
                    (FRONT, 0, layer, +1, 0),
                    (DOWN, 0, layer, +1, 0),
                    (BACK, 0, layer, +1, 0)
                ])

        elif axis == 'U':
            if layer == 0:
                self.rotate(UP, -rotation)
            elif layer == self.size - 1:
                self.rotate(DOWN, rotation)

            for _ in range(rotation):
                self.shift([
                    (LEFT, layer, 0, 0, +1),
                    (BACK, border - layer, 0, 0, +1),
                    (RIGHT, layer, border, 0, -1),
                    (FRONT, layer, border, 0, -1)
                ])

        elif axis == 'F':
            if layer == 0:
                self.rotate(FRONT, -rotation)
            elif layer == self.size - 1:
                self.rotate(BACK, rotation)

            for _ in range(rotation):
                self.shift([
                    (UP, border - layer, 0, 0, +1),
                    (RIGHT, 0, layer, +1, 0),
                    (DOWN, layer, border, 0, -1),
                    (LEFT, border, border - layer, -1, 0),
                ])

        done = self.done()
        reward = 0 if done else -1
        return np.array(self.cube), reward, done, {}


    def done(self):
        return np.alltrue(np.arange(6).repeat(self.size**2).reshape(6, self.size, self.size)\
                == self.cube)


    def shift(self, lines):
        f, sx, sy, dx, dy = lines[-1]

        tmp = [0] * self.size
        for i in range(self.size):
            tmp[i] = self.cube[f, sx + i * dx, sy + i * dy]

        for f, sx, sy, dx, dy in lines:
            for i in range(self.size):
                v = self.cube[f, sx + i * dx, sy + i * dy]
                self.cube[f, sx + i * dx, sy + i * dy] = tmp[i]
                tmp[i] = v


    def rotate(self, face, times):
        self.cube[face] = np.rot90(self.cube[face], k=times % 4)


    def reset(self, num_steps=0):
        self.cube = np.arange(6).repeat(self.size**2).reshape(6, self.size, self.size)

        if num_steps != 0:
            if num_steps == -1:
                num_steps = self.size**2 * 2 * 3

            for _ in range(num_steps):
                action = self.action_space.sample()
                self.step(action)


    def render(self, mode='ansi'):
        value = [[" "] * (self.size * 3 + 2) for i in range(self.size * 4 + 3)]

        width = self.size + 1

        start_pos = [(width, 0), (width, width), (0, width),
                        (width, 2 * width), (width * 3, width),
                        (width * 2, width)
                        ]

        for face, (x, y) in enumerate(start_pos):
            for i in range(self.size):
                for j in range(self.size):
                    idx = self.cube[face, i, j]
                    rgb, ch = COLORS[idx]
                    value[x + i][y + j] = get_color(ch, rgb=0, bg=rgb)

        print('\n\n' + '\n'.join(''.join(row) for row in value))


class RubikActionSpace(gym.Space):
    def __init__(self, size):
        self.size = size

    def sample(self):
        return (gym.spaces.np_random.choice(list("FLU")),
                gym.spaces.np_random.randint(self.size),
                gym.spaces.np_random.choice((1, 3))
        )

    def contains(self, action):
        axis, layer, rotation = action
        return axis in "LFU" and 0 <= layer < self.size and 1 <= rotation <= 3

    def __repr__(self):
        return "Rubik(%d)" % self.size

    def __eq__(self, other):
        return self.size == other.size


class RubikPermutationSpace(gym.Space):
    def __init__(self, size):
        super().__init__()