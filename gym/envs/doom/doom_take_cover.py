import logging
import os

import numpy as np

from doom_py import DoomGame, Mode, Button, GameVariable, ScreenFormat, ScreenResolution, Loader
from gym import spaces
from gym.envs.doom import doom_env

logger = logging.getLogger(__name__)

class DoomTakeCoverEnv(doom_env.DoomEnv):
    """
    ------------ Training Mission 8 - Take Cover ------------
    This map is to train you on the damage of incoming missiles.
    It is a rectangular map with monsters firing missiles and fireballs
    at you. You need to survive as long as possible.

    Allowed actions:
        [10] - MOVE_RIGHT                       - Move to the right - Values 0 or 1
        [11] - MOVE_LEFT                        - Move to the left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        +  1    - Several times per second - Survive as long as possible

    Goal: 750 points
        Survive for ~ 20 seconds

    Mode:
        - env.mode can be 'fast' or 'normal' (e.g. env.mode = 'fast')
        - 'fast' (default) will run as fast as possible (~75 fps) (best for simulation, harder for human to watch)
        - 'normal' will run at roughly 30 fps (easier for human to watch)

    Ends when:
        - Player is dead (one or two fireballs should be enough to kill you)
        - Timeout (60 seconds - 2,100 frames)

    Actions:
    Either of
        1) action = [0, 1]          # Recommended
            where parameter #1 is MOVE_RIGHT (0 or 1)
            where parameter #2 is MOVE_LEFT (0 or 1)
        or
        2) actions = [0] * 43       # To train for the Deathmatch level
           actions[10] = 0      # MOVE_RIGHT
           actions[11] = 1      # MOVE_LEFT
    -----------------------------------------------------
    """
    def __init__(self):
        super(DoomTakeCoverEnv, self).__init__()
        package_directory = os.path.dirname(os.path.abspath(__file__))
        self.loader = Loader()
        self.game = DoomGame()
        self.game.load_config(os.path.join(package_directory, 'assets/take_cover.cfg'))
        self.game.set_vizdoom_path(self.loader.get_vizdoom_path())
        self.game.set_doom_game_path(self.loader.get_freedoom_path())
        self.game.set_doom_scenario_path(self.loader.get_scenario_path('take_cover.wad'))
        self.game.set_doom_map('map01')
        self.screen_height = 480                    # Must match .cfg file
        self.screen_width = 640                     # Must match .cfg file
        self.game.set_window_visible(False)
        self.viewer = None
        self.game.init()
        self.game.new_episode()

        # 2 allowed actions [10, 11] (must match .cfg file)
        self.action_space = spaces.HighLow(np.matrix([[0, 1, 0]] * 38 + [[-10, 10, 0]] * 2 + [[-100, 100, 0]] * 3))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.action_space.allowed_actions = [10, 11]

        self._seed()
