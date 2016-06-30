import logging
from gym.envs.doom import doom_env

logger = logging.getLogger(__name__)

class DoomBasicEnv(doom_env.DoomEnv):
    """
    ------------ Training Mission 1 - Basic ------------
    This map is rectangular with gray walls, ceiling and floor.
    You are spawned in the center of the longer wall, and a red
    circular monster is spawned randomly on the opposite wall.
    You need to kill the monster (one bullet is enough).

    Allowed actions:
        [0]  - ATTACK                           - Shoot weapon - Values 0 or 1
        [10] - MOVE_RIGHT                       - Move to the right - Values 0 or 1
        [11] - MOVE_LEFT                        - Move to the left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        +101    - Killing the monster
        -  5    - Missing a shot
        -  1    - 35 times per second - Kill the monster faster!

    Goal: 10 points
        Kill the monster in 3 secs with 1 shot

    Mode:
        - env.mode can be 'fast', 'normal' or 'human' (e.g. env.mode = 'fast')
        - 'fast' (default) will run as fast as possible (~75 fps) (best for simulation)
        - 'normal' will run at roughly 35 fps (easier for human to watch)
        - 'human' will let you play the game (keyboard only: Arrow Keys, '<', '>' and Ctrl)

    Ends when:
        - Monster is dead
        - Player is dead
        - Timeout (10 seconds - 350 frames)

    Actions:
        actions = [0] * 43
        actions[0] = 0       # ATTACK
        actions[10] = 1      # MOVE_RIGHT
        actions[11] = 0      # MOVE_LEFT
    -----------------------------------------------------
    """
    def __init__(self):
        super(DoomBasicEnv, self).__init__(0)
