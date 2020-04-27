# -*- coding: utf-8 -*-
import sys
from contextlib import closing
from io import StringIO

import numpy

from gym import Env
from gym.spaces import Discrete
from gym.spaces import Box


def _check_rows(board):
    for row in board:
        if len(set(row)) == 1:
            return row[0]
    return None


def _check_diagonals(board):
    if len(set([board[i][i] for i in range(len(board))])) == 1:
        return board[0][0]
    if len(set([board[i][len(board) - i - 1] for i in range(len(board))])) == 1:
        return board[0][len(board) - 1]
    return None


def _check_winner(board):
    for newBoard in [board, board.T]:
        result = _check_rows(newBoard)
        if result is not None:
            return result
    return _check_diagonals(board)


class TicTacToe(Env):
    """
    Tic Tac Toe.

    The environment automatically assign each action alternating the players,
    so that you can use this with a single agent playing with itself.

    Rewards:
        -2 if out the grid or a filled position
        -1 loose
         0 not a final move
         1 tie
         2 win
    """
    metadata = {'render.modes': ['human', 'ansi', ]}

    def __init__(self, dim=3):
        # the following needs to be filled for super `Env`
        self.observation_space = Box(low=-1.0, high=1.0, shape=(dim, dim), dtype=int)  # 9 possible position
        self.action_space = Discrete(9)
        self.reward_range = (-1, 1)
        self.grid = numpy.zeros(shape=(dim, dim), dtype=int) - 1
        self.moves = 0
        self.last_action = None

    def step(self, action):
        """ Here is the main logic for each step

        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        x = int(action / self.grid.shape[0])
        y = action % self.grid.shape[0]
        player = self.moves % 2

        self.last_action = action
        self.moves += 1

        # check if x and y are off the grid
        if not (0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]):
            return self.grid, -2, False, {}
        # check if chosen position can be filled
        if self.grid[x, y] != -1:
            return self.grid, -2, False, {}

        self.grid[x, y] = player

        r = _check_winner(self.grid)

        if r == -1:  # not finished yet
            return self.grid, 0, False, {}

        if player == r:  # this player won
            return self.grid, 2, True, {}

        if numpy.argwhere(self.grid == -1).shape[0] == 0:  # tie
            return self.grid, 1, True, {}

        if r is None: # not finished yet
            return self.grid, 0, False, {}

        return self.grid, -1, True, {}

    def reset(self):
        self.grid = numpy.zeros(shape=self.grid.shape, dtype=int) - 1
        self.moves = 0
        self.last_action = None
        return self.grid

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        player = self.moves % 2
        text = '\n'
        for i in range(self.grid.shape[0]):
            text += '|'
            for j in range(self.grid.shape[1]):
                if self.grid[i][j] != -1:
                    text += ' {} |'.format(self.grid[i][j])
                else:
                    text += '   |'
            text += '\n'
        if self.last_action is not None:
            x = int(self.last_action / self.grid.shape[0])
            y = self.last_action % self.grid.shape[0]
            text += '\nmoves {} (last action: ({}, {}) player: {})\n'.format(self.moves, x, y, player)
        else:
            text += '\nmoves {} (last action: None player: None)\n'.format(self.moves)
        outfile.write(text)

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
