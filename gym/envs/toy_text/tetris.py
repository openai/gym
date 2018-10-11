from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from six import StringIO

import gym
from gym import spaces, utils

import numpy as np

SHAPES = {
    'T': [(0, 0), (-1, 0), (1, 0), (0, -1)],
    'J': [(0, 0), (-1, 0), (0, -1), (0, -2)],
    'L': [(0, 0), (1, 0), (0, -1), (0, -2)],
    'Z': [(0, 0), (-1, 0), (0, -1), (1, -1)],
    'S': [(0, 0), (-1, -1), (0, -1), (1, 0)],
    'I': [(0, 0), (0, -1), (0, -2), (0, -3)],
    'O': [(0, 0), (0, -1), (-1, 0), (-1, -1)],
}


def is_valid_index(x, y, board):
    return (
        y >= 0 and
        y < board.shape[1] and
        x >= 0 and
        x < board.shape[0]
    )


def tetris_action(f):
    def _f(self, board):
        shape, anchor = f(self, board)
        if self.is_valid_piece_for(board):
            self.shape = shape
            self.board = board
    return _f


class Tetrimino(object):
    def __init__(self, shape_id, anchor):
        assert len(anchor) == 2
        self.shape = SHAPES[shape_id]
        self.anchor = anchor

    @tetris_action
    def rotate_left(self, _):
        shape = [(-j, i) for i, j in self.shape]
        return (shape, self.anchor)

    @tetris_action
    def rotate_right(self, _):
        shape = [(j, -i) for i, j in self.shape]
        return (shape, self.anchor)

    @tetris_action
    def left(self, board):
        anchor = (self.anchor[0] - 1, self.anchor[1])
        return (self.shape, anchor)

    @tetris_action
    def right(self, board):
        anchor = (self.anchor[0] + 1, self.anchor[1])
        return (self.shape, anchor)

    @tetris_action
    def soft_drop(self, board):
        anchor = (self.anchor[0], self.anchor[1] + 1)
        return (self.shape, anchor)

    # Since this function updates its own anchor and shape, it doesn't need the
    # tetris_action wrapper to check for validity. Note that it is assumed
    # that the piece is in a valid location when this function is called.
    def hard_drop(self, board):
        while self.is_valid_piece_for(board):
            self.anchor = (self.anchor[0], self.anchor[1] + 1)
        self.anchor = (self.anchor[0], self.anchor[1] - 1)

    def take_action(self, board, action_id):
        assert 0 <= action_id <= 5
        if action_id == 0:
            return self.left(board)
        if action_id == 1:
            return self.right(board)
        if action_id == 2:
            return self.rotate_left(board)
        if action_id == 3:
            return self.rotate_right(board)
        if action_id == 4:
            return self.soft_drop(board)
        if action_id == 5:
            return self.hard_drop(board)

    @property
    def indices(self):
        return [
            (self.anchor[0] + i, self.anchor[1] + j)
            for i, j in self.shape
        ]

    def is_valid_piece_for(self, board):
        return all(
            is_valid_index(x, y, board) and not board[x, y]
            for x, y in self.indices
        )

    def is_dropped_for(self, board):
        return all(
            not is_valid_index(x, y + 1, board) or board[x, y + 1]
            for x, y in self.indices
        )

    @property
    def bounding_box(self):
        min_x = min(x for x, _ in self.shape)
        max_x = max(x for x, _ in self.shape)
        min_y = min(y for _, y in self.shape)
        max_y = max(y for _, y in self.shape)
        return ((min_x, max_x), (min_y, max_y))

    def __repr__(self):
        x_range, y_range = self.bounding_box
        return '\n'.join(
            ''.join(
                'X' if (x, y) in self.shape else ' '
                for x in range(x_range[0], x_range[1] + 1)
            )
            for y in range(y_range[0], y_range[1] + 1)
        )


class TetrisEnv(gym.Env):
    """Simple block dropping game."""

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, width=10, height=20):
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(width),
            spaces.Discrete(height)
        ))
        self.width, self.height = width, height
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    def set_piece(self, board):
        for x, y in self.tetrimino.indices:
            if is_valid_index(x, y, board):
                board[x, y] = True
        return board

    def choose_random_shape(self):
        shape = self.np_random.choice(list(SHAPES.keys()))
        anchor = (self.width // 2, 0)
        self.tetrimino = Tetrimino(shape, anchor)

    def clear_lines(self):
        cleared = np.all(self.board, axis=0)
        num_cleared = np.sum(cleared)
        if not num_cleared:
            return
        keep_lines, = np.where(~cleared)
        board = np.concatenate([
            np.zeros(shape=(self.width, num_cleared), dtype=np.bool),
            self.board[:, keep_lines],
        ], axis=1)
        return board, num_cleared

    def get_state(self):
        board = self.set_piece(np.copy(self.board))
        return board

    def step(self, action):
        assert self.action_space.contains(action)
        self.tetrimino.take_action(self.board, action)
        if self.tetrimino.is_dropped_for(self.board):
            if np.any(self.board[:, 0]):  # Indicates the player is dead.
                return self.get_state(), 0, True, {}
            self.toggle_piece(True)
            self.choose_random_shape()
            self.clear_lines()
        return self.get_state(), 0, False, {}

    def reset(self):
        self.board = np.zeros(shape=(self.width, self.height), dtype=np.bool)
        self.choose_random_shape()

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        s = ['o' + '-' * self.board.shape[0] + 'o']
        s += [
            '|' + ''.join('X' if j else ' ' for j in i) + '|'
            for i in self.board.T
        ]
        s += ['o' + '-' * self.board.shape[0] + 'o']
        s = '\n'.join(s) + '\n'

        outfile.write(s)
        if mode != 'human':
            return outfile
