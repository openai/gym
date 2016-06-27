import logging
from gym.envs.doom import doom_env

logger = logging.getLogger(__name__)

class DoomDeathmatchEnv(doom_env.DoomEnv):
    """
    ------------ Final Mission - Deathmatch ------------
    Kill as many monsters as possible without being killed.

    Allowed actions:
        ALL
    Note: see controls.md for details

    Rewards:
        +1      - Killing a monster

    Goal: 20 points
        Kill 20 monsters

    Mode:
        - env.mode can be 'fast', 'normal' or 'human' (e.g. env.mode = 'fast')
        - 'fast' (default) will run as fast as possible (~75 fps) (best for simulation)
        - 'normal' will run at roughly 35 fps (easier for human to watch)
        - 'human' will let you play the game (mouse and full keyboard)

    Ends when:
        - Player is dead
        - Timeout (3 minutes - 6,300 frames)

    Actions:
        actions = [0] * 43
        actions[0] = 0       # ATTACK
        actions[1] = 0       # USE
        [...]
        actions[42] = 0      # MOVE_UP_DOWN_DELTA
        A full list of possible actions is available in controls.md
    -----------------------------------------------------
    """
    def __init__(self):
        super(DoomDeathmatchEnv, self).__init__(8)
