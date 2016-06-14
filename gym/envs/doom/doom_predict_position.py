import logging
from gym.envs.doom import doom_env

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
        [14] - TURN_RIGHT                       - Turn right - Values 0 or 1
        [15] - TURN_LEFT                        - Turn left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        +  1    - Killing the monster
        -0.0001 - 35 times per second - Kill the monster faster!

    Goal: 0.5 point
        Kill the monster

    Hint: Missile launcher takes longer to load. You must wait a good second after the game starts
        before trying to fire it.

    Mode:
        - env.mode can be 'fast', 'normal' or 'human' (e.g. env.mode = 'fast')
        - 'fast' (default) will run as fast as possible (~75 fps) (best for simulation)
        - 'normal' will run at roughly 35 fps (easier for human to watch)
        - 'human' will let you play the game (keyboard only: Arrow Keys, '<', '>' and Ctrl)

    Ends when:
        - Monster is dead
        - Out of missile (you only have one)
        - Timeout (20 seconds - 700 frames)

    Actions:
        actions = [0] * 43
        actions[0] = 0       # ATTACK
        actions[14] = 1      # TURN_RIGHT
        actions[15] = 0      # TURN_LEFT
    -----------------------------------------------------
    """
    def __init__(self):
        super(DoomPredictPositionEnv, self).__init__(6)
