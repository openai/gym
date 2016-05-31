import logging
import os

import numpy as np

from doom_py import DoomGame, Mode, Button, GameVariable, ScreenFormat, ScreenResolution, Loader
from gym import error, spaces
from gym.envs.doom import doom_env
from gym.utils import seeding

logger = logging.getLogger(__name__)

class DoomHealthGatheringEnv(doom_env.DoomEnv):
    """
    ------------ Training Mission 5 - Health Gathering ------------
    This map is a guide on how to survive by collecting health packs.
    It is a rectangle with green, acidic floor which hurts the player
    periodically. There are also medkits spread around the map, and
    additional kits will spawn at interval.

    Allowed actions:
        [12] - MOVE_FORWARD                     - Move forward - Values 0 or 1
        [13] - TURN_RIGHT                       - Turn right - Values 0 or 1
        [14] - TURN_LEFT                        - Turn left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        +  1    - Several times per second - Survive as long as possible
        -100    - Death penalty

    Goal: 1000 points
        Stay alive long enough to reach 1,000 points (~ 30 secs)

    Ends when:
        - Player is dead
        - Timeout (60 seconds - 2,100 frames)
    -----------------------------------------------------
    """
    def __init__(self):
        super(DoomHealthGatheringEnv, self).__init__()
        package_directory = os.path.dirname(os.path.abspath(__file__))
        self.loader = Loader()
        self.game = DoomGame()
        self.game.load_config(os.path.join(package_directory, 'assets/health_gathering.cfg'))
        self.game.set_vizdoom_path(self.loader.get_vizdoom_path())
        self.game.set_doom_game_path(self.loader.get_freedoom_path())
        self.game.set_doom_scenario_path(self.loader.get_scenario_path('health_gathering.wad'))
        self.game.set_doom_map('map01')
        self.screen_height = 480                    # Must match .cfg file
        self.screen_width = 640                     # Must match .cfg file
        self.game.set_window_visible(False)
        self.viewer = None
        self.game.init()
        self.game.new_episode()

        # 3 allowed actions [12, 13, 14] (must match .cfg file)
        self.action_space = spaces.HighLow(np.matrix([[0, 1, 0]] * 3))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))

        self._seed()

    def _seed(self, seed=None):
        seed = seeding.hash_seed(seed) % 2**32
        self.game.set_seed(seed)
        return [seed]
