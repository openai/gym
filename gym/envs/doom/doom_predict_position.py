import logging
import os

import numpy as np

from doom_py import DoomGame, Mode, Button, GameVariable, ScreenFormat, ScreenResolution, Loader
from gym import error, spaces
from gym.envs.doom import doom_env
from gym.utils import seeding

logger = logging.getLogger(__name__)

class DoomPredictPositionEnv(doom_env.DoomEnv):
    """
    ------------ Training Mission 7 - Predict Position ------------
    This map is designed to train you on using a rocket launcher.
    It is a rectangular map with a monster on the opposite side. You need
    to use your rocket launcher to kill it. The rocket adds a delay between
    the moment it is fired and the moment it reaches the other side of the room.
    You need to predict the position of the monster to kill it.

    Allowed actions:
        [0]  - ATTACK                           - Shoot weapon - Values 0 or 1
        [13] - TURN_RIGHT                       - Turn right - Values 0 or 1
        [14] - TURN_LEFT                        - Turn left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        +  1    - Killing the monster
        -0.0001 - Several times per second - Kill the monster faster!

    Goal: 0.5 point
        Kill the monster

    Hint: Missile launcher takes longer to load. You must wait a good second after the game starts
        before trying to fire it.

    Ends when:
        - Monster is dead
        - Out of missile (you only have one)
        - Timeout (20 seconds - 700 frames)
    -----------------------------------------------------
    """
    def __init__(self):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        self.loader = Loader()
        self.game = DoomGame()
        self.game.load_config(os.path.join(package_directory, 'assets/predict_position.cfg'))
        self.game.set_vizdoom_path(self.loader.get_vizdoom_path())
        self.game.set_doom_game_path(self.loader.get_freedoom_path())
        self.game.set_doom_scenario_path(self.loader.get_scenario_path('predict_position.wad'))
        self.game.set_doom_map('map01')
        self.screen_height = 480                    # Must match .cfg file
        self.screen_width = 640                     # Must match .cfg file
        self.game.set_window_visible(False)
        self.viewer = None
        self.game.init()
        self.game.new_episode()

        # 3 allowed actions [0, 13, 14] (must match .cfg file)
        self.action_space = spaces.HighLow(np.matrix([[0, 1, 0]] * 3))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))

        self._seed()

    def _seed(self, seed=None):
        # Derive a random seed.
        seed = seeding.hash_seed(seed) % 2**32
        self.game.set_seed(seed)
        return [seed]
