from gym import error
try:
    import pachi_py
except ImportError as e:
    # The dependency group [pachi] should match the name is setup.py.
    raise error.DependencyNotInstalled('{}. (HINT: you may need to install the Go dependencies via "pip install gym[pachi]".)'.format(e))

import numpy as np
import gym
from gym import spaces
from six import StringIO
import sys
import six


# The coordinate representation of Pachi (and pachi_py) is defined on a board
# with extra rows and columns on the margin of the board, so positions on the board
# are not numbers in [0, board_size**2) as one would expect. For this Go env, we instead
# use an action representation that does fall in this more natural range.

def _pass_action(board_size):
    return board_size**2

def _resign_action(board_size):
    return board_size**2 + 1

def _coord_to_action(board, c):
    '''Converts Pachi coordinates to actions'''
    if c == pachi_py.PASS_COORD: return _pass_action(board.size)
    if c == pachi_py.RESIGN_COORD: return _resign_action(board.size)
    i, j = board.coord_to_ij(c)
    return i*board.size + j

def _action_to_coord(board, a):
    '''Converts actions to Pachi coordinates'''
    if a == _pass_action(board.size): return pachi_py.PASS_COORD
    if a == _resign_action(board.size): return pachi_py.RESIGN_COORD
    return board.ij_to_coord(a // board.size, a % board.size)

def str_to_action(board, s):
    return _coord_to_action(board, board.str_to_coord(s.encode()))

class GoState(object):
    '''
    Go game state. Consists of a current player and a board.
    Actions are exposed as integers in [0, num_actions), which is different
    from Pachi's internal "coord_t" encoding.
    '''
    def __init__(self, board, color):
        '''
        Args:
            board: current board
            color: color of current player
        '''
        assert color in [pachi_py.BLACK, pachi_py.WHITE], 'Invalid player color'
        self.board, self.color = board, color

    def act(self, action):
        '''
        Executes an action for the current player

        Returns:
            a new GoState with the new board and the player switched
        '''
        return GoState(
            self.board.play(_action_to_coord(self.board, action), self.color),
            pachi_py.stone_other(self.color))

    def __repr__(self):
        return 'To play: {}\n{}'.format(six.u(pachi_py.color_to_str(self.color)), self.board.__repr__().decode())


### Adversary policies ###
def random_policy(curr_state, prev_state, prev_action):
    b = curr_state.board
    legal_coords = b.get_legal_coords(curr_state.color)
    return _coord_to_action(b, np.random.choice(legal_coords))

def make_pachi_policy(board, engine_type='uct', threads=1, pachi_timestr=''):
    engine = pachi_py.PyPachiEngine(board, engine_type, six.b('threads=%d' % threads))

    def pachi_policy(curr_state, prev_state, prev_action):
        if prev_state is not None:
            assert engine.curr_board == prev_state.board, 'Engine internal board is inconsistent with provided board. The Pachi engine must be called consistently as the game progresses.'
            prev_coord = _action_to_coord(prev_state.board, prev_action)
            engine.notify(prev_coord, prev_state.color)
            engine.curr_board.play_inplace(prev_coord, prev_state.color)
        out_coord = engine.genmove(curr_state.color, pachi_timestr)
        out_action = _coord_to_action(curr_state.board, out_coord)
        engine.curr_board.play_inplace(out_coord, curr_state.color)
        return out_action

    return pachi_policy


def _play(black_policy_fn, white_policy_fn, board_size=19):
    '''
    Samples a trajectory for two player policies.
    Args:
        black_policy_fn, white_policy_fn: functions that maps a GoState to a move coord (int)
    '''
    moves = []

    prev_state, prev_action = None, None
    curr_state = GoState(pachi_py.CreateBoard(board_size), BLACK)

    while not curr_state.board.is_terminal:
        a = (black_policy_fn if curr_state.color == BLACK else white_policy_fn)(curr_state, prev_state, prev_action)
        next_state = curr_state.act(a)
        moves.append((curr_state, a, next_state))

        prev_state, prev_action = curr_state, a
        curr_state = next_state

    return moves


class GoEnv(gym.Env):
    '''
    Go environment. Play against a fixed opponent.
    '''
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, player_color, opponent, observation_type, illegal_move_mode, board_size):
        '''
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: An opponent policy
            observation_type: State encoding
            illegal_move_mode: What to do when the agent makes an illegal move. Choices: 'raise' or 'lose'
        '''
        assert isinstance(board_size, int) and board_size >= 1, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size

        colormap = {
            'black': pachi_py.BLACK,
            'white': pachi_py.WHITE,
        }
        try:
            self.player_color = colormap[player_color]
        except KeyError:
            raise error.Error("player_color must be 'black' or 'white', not {}".format(player_color))

        self.opponent_policy = None
        self.opponent = opponent

        assert observation_type in ['image3c']
        self.observation_type = observation_type

        assert illegal_move_mode in ['lose', 'raise']
        self.illegal_move_mode = illegal_move_mode

        # One action for each board position, pass, and resign
        self.action_space = spaces.Discrete(self.board_size**2 + 2)

        if self.observation_type == 'image3c':
            shape = pachi_py.CreateBoard(self.board_size).encode().shape
            self.observation_space = spaces.Box(np.zeros(shape), np.ones(shape))
        else:
            raise error.Error('Unsupported observation type: {}'.format(self.observation_type))

        self.reset()

    def _reset(self):
        self.state = GoState(pachi_py.CreateBoard(self.board_size), pachi_py.BLACK)

        # (re-initialize) the opponent
        # necessary because a pachi engine is attached to a game via internal data in a board
        # so with a fresh game, we need a fresh engine
        self._reset_opponent(self.state.board)

        # Let the opponent play if it's not the agent's turn
        opponent_resigned = False
        if self.state.color != self.player_color:
            self.state, opponent_resigned = self._exec_opponent_play(self.state, None, None)

        # We should be back to the agent color
        assert self.state.color == self.player_color

        self.done = self.state.board.is_terminal or opponent_resigned
        return self.state.board.encode()

    def _render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(repr(self.state) + '\n')
        return outfile

    def _step(self, action):
        assert self.state.color == self.player_color

        # If already terminal, then don't do anything
        if self.done:
            return self.state.board.encode(), 0., True, {'state': self.state}

        # If resigned, then we're done
        if action == _resign_action(self.board_size):
            self.done = True
            return self.state.board.encode(), -1., True, {'state': self.state}

        # Play
        prev_state = self.state
        try:
            self.state = self.state.act(action)
        except pachi_py.IllegalMove:
            if self.illegal_move_mode == 'raise':
                six.reraise(*sys.exc_info())
            elif self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                self.done = True
                return self.state.board.encode(), -1., True, {'state': self.state}
            else:
                raise error.Error('Unsupported illegal move action: {}'.format(self.illegal_move_mode))

        # Opponent play
        if not self.state.board.is_terminal:
            self.state, opponent_resigned = self._exec_opponent_play(self.state, prev_state, action)
            # After opponent play, we should be back to the original color
            assert self.state.color == self.player_color

            # If the opponent resigns, then the agent wins
            if opponent_resigned:
                self.done = True
                return self.state.board.encode(), 1., True, {'state': self.state}

        # Reward: if nonterminal, then the reward is 0
        if not self.state.board.is_terminal:
            self.done = False
            return self.state.board.encode(), 0., False, {'state': self.state}

        # We're in a terminal state. Reward is 1 if won, -1 if lost
        assert self.state.board.is_terminal
        self.done = True
        white_wins = self.state.board.official_score > 0
        player_wins = (white_wins and self.player_color == pachi_py.WHITE) or (not white_wins and self.player_color == pachi_py.BLACK)
        reward = 1. if player_wins else -1.
        return self.state.board.encode(), reward, True, {'state': self.state}

    def _exec_opponent_play(self, curr_state, prev_state, prev_action):
        assert curr_state.color != self.player_color
        opponent_action = self.opponent_policy(curr_state, prev_state, prev_action)
        opponent_resigned = opponent_action == _resign_action(self.board_size)
        return curr_state.act(opponent_action), opponent_resigned

    @property
    def _state(self):
        return self.state

    def _reset_opponent(self, board):
        if self.opponent == 'random':
            self.opponent_policy = random_policy
        elif self.opponent == 'pachi:uct:_2400':
            self.opponent_policy = make_pachi_policy(board=board, engine_type=six.b('uct'), pachi_timestr=six.b('_2400')) # TODO: strength as argument
        else:
            raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))
