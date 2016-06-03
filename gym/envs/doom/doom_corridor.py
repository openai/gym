import logging
import os

import numpy as np

from doom_py import DoomGame, Mode, Button, GameVariable, ScreenFormat, ScreenResolution, Loader
from gym import spaces
from gym.envs.doom import doom_env

logger = logging.getLogger(__name__)

class DoomCorridorEnv(doom_env.DoomEnv):
    """
    ------------ Training Mission 2 - Corridor ------------
    This map is designed to improve your navigation. There is a vest
    at the end of the corridor, with 6 enemies (3 groups of 2). Your goal
    is to get to the vest as soon as possible, without being killed.

    Allowed actions:
        [0]  - ATTACK                           - Shoot weapon - Values 0 or 1
        [10] - MOVE_RIGHT                       - Move to the right - Values 0 or 1
        [11] - MOVE_LEFT                        - Move to the left - Values 0 or 1
        [13] - MOVE_FORWARD                     - Move forward - Values 0 or 1
        [14] - TURN_RIGHT                       - Turn right - Values 0 or 1
        [15] - TURN_LEFT                        - Turn left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        + dX    - For getting closer to the vest
        - dX    - For getting further from the vest
        -100    - Penalty for being killed

    Goal: 1,270 points
     Reach the vest (try also killing guards, rather than just running)

    Ends when:
        - Player touches vest
        - Player is dead
        - Timeout (1 minutes - 2,100 frames)

    Actions:
    Either of
        1) action = [0, 1, 0, 0, 0, 0]
            where parameter #1 is ATTACK (0 or 1)
            where parameter #2 is MOVE_RIGHT (0 or 1)
            where parameter #3 is MOVE_LEFT (0 or 1)
            where parameter #4 is MOVE_FORWARD (0 or 1)
            where parameter #5 is TURN_RIGHT (0 or 1)
            where parameter #6 is TURN_LEFT (0 or 1)
        or
        2) actions = [0] * 41
           actions[0] = 0       # ATTACK
           actions[10] = 1      # MOVE_RIGHT
           actions[11] = 0      # MOVE_LEFT
           actions[13] = 0      # MOVE_FORWARD
           actions[14] = 0      # TURN_RIGHT
           actions[15] = 0      # TURN_LEFT
    -----------------------------------------------------
    """
    def __init__(self):
        super(DoomCorridorEnv, self).__init__()
        package_directory = os.path.dirname(os.path.abspath(__file__))
        self.loader = Loader()
        self.game = DoomGame()
        self.game.load_config(os.path.join(package_directory, 'assets/deadly_corridor.cfg'))
        self.game.set_vizdoom_path(self.loader.get_vizdoom_path())
        self.game.set_doom_game_path(self.loader.get_freedoom_path())
        self.game.set_doom_scenario_path(self.loader.get_scenario_path('deadly_corridor.wad'))
        self.screen_height = 480                    # Must match .cfg file
        self.screen_width = 640                     # Must match .cfg file
        self.game.set_window_visible(False)
        self.viewer = None
        self.game.init()
        self.game.new_episode()

        # action indexes are [0, 10, 11, 13, 14, 15]
        self.action_space = spaces.HighLow(np.matrix([[0, 1, 0]] * 36 + [[-10, 10, 0]] * 2 + [[0, 100, 0]] * 3))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.action_space.allowed_actions = [0, 10, 11, 13, 14, 15]

        self._seed()
