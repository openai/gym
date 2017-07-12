import gym
import datetime
import time
from board import Board
from gym import error, spaces, utils
from gym.utils import seeding
import random
import pygame, sys
from lib import DisplayBoard


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.numb = 0
        # Board is a class consisting of functions like if game is over or not
        self.board = Board()

    def _step(self, action):
        # if it is a foul move end the game give a reward of -2
        if action not in self.board.getvalidmoves():
            return self.board, -2, True, None
        else:
            # do the action 
            self.board.move(action, 1)
            if self.board.rowcolumn():
                return self.board.state, 1, True,None
            elif self.board.full_posit():
                return self.board.state, 0, True,None
            else:
                # opponent's move was placed on the boaed
                self.opponentmove()
                if self.board.rowcolumn():
                    return self.board.state, -1, True,None
                elif self.board.full_posit():
                    return self.board.state, 0, True,None
                else:
                    return self.board.state, 0, False,None

    def opponentmove(self):
        # opponent was ensemble of random and safe players
        '''
        Random Player: plays randomly
        Safe Player : Will block or keep a win move.
        '''
        random.seed(datetime.datetime.now())
        r = random.uniform(0, 2)
        if r < 1:
            random.seed(datetime.datetime.now())
            random_choice = random.choice(self.board.getvalidmoves())
            self.board.move(random_choice, 2)
        else:
            vm = self.board.getvalidmoves()
            final_moves = []
            block_moves = []
            for i in range(len(vm)):
                self.board.state[vm[i]] = 2
                if self.board.rowcolumn():
                    final_moves.append(vm[i])
                self.board.state[vm[i]] = 0
                self.board.state[vm[i]] = 1
                if self.board.rowcolumn():
                    block_moves.append(vm[i])
                self.board.state[vm[i]] = 0
            if len(final_moves) == 0:
                if len(block_moves) == 0:
                    random.seed(datetime.datetime.now())
                    self.board.move(random.choice(vm), 2)
                else:
                    random.seed(datetime.datetime.now())
                    self.board.move(random.choice(block_moves), 2)
            else:
                random.seed(datetime.datetime.now())
                self.board.move(random.choice(final_moves), 2)

    def _reset(self):
        # when environment is reset it can be opponent's turn or players turn
        self.numb = 0
        self.board.reset()
        random.seed(datetime.datetime.now())
        turn = random.choice([1, 2])
        if turn == 2:
            self.opponentmove()
        return self.board.state

    def _render(self, mode='human', close=False):
        # To render pygame module was used
        if self.numb == 0:
            pygame.init()
            self.displayboard = DisplayBoard(grid_size=3, box_size=100, border=50, line_width=10)
            self.numb = 1
        else:
            for position in range(9):
                if position == 0:
                    self.displayboard.process_click(93, 101, self.board.state[position])
                elif position == 1:
                    self.displayboard.process_click(210, 104, self.board.state[position])
                elif position == 2:
                    self.displayboard.process_click(311, 99, self.board.state[position])
                elif position == 3:
                    self.displayboard.process_click(95, 212, self.board.state[position])
                elif position == 4:
                    self.displayboard.process_click(208, 213, self.board.state[position])
                elif position == 5:
                    self.displayboard.process_click(323, 211, self.board.state[position])
                elif position == 6:
                    self.displayboard.process_click(89, 323, self.board.state[position])
                elif position == 7:
                    self.displayboard.process_click(208, 320, self.board.state[position])
                elif position == 8:
                    self.displayboard.process_click(316, 317, self.board.state[position])
        pygame.display.update()
        time.sleep(0.59)
