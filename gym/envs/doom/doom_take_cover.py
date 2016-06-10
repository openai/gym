import logging
import os

import numpy as np

from doom_py import DoomGame, Mode, Button, GameVariable, ScreenFormat, ScreenResolution, Loader
from gym import error, spaces
from gym.envs.doom import doom_env
from gym.utils import seeding

logger = logging.getLogger(__name__)

class DoomTakeCoverEnv(doom_env.DoomEnv):
    """
    ------------ Training Mission 8 - Take Cover ------------
    This map is to train you on the damage of incoming missiles.
    It is a rectangular map with monsters firing missiles and fireballs
    at you. You need to survive as long as possible.

    Allowed actions:
        [9]  - MOVE_RIGHT                       - Move to the right - Values 0 or 1
        [10] - MOVE_LEFT                        - Move to the left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        +  1    - Several times per second - Survive as long as possible

    Goal: 750 points
        Survive for ~ 20 seconds

    Ends when:
        - Player is dead (one or two fireballs should be enough to kill you)
        - Timeout (60 seconds - 2,100 frames)
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

        # 2 allowed actions [9, 10] (must match .cfg file)
        self.action_space = spaces.HighLow(np.matrix([[0, 1, 0]] * 2))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))

        self._seed()

    def _seed(self, seed=None):
        seed = seeding.hash_seed(seed) % 2**32
        self.game.set_seed(seed)
        return [seed]
