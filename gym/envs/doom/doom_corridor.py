import logging
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

    Goal: 1,000 points
        Reach the vest (or at least get past the guards in the 3rd group)

    Mode:
        - env.mode can be 'fast', 'normal' or 'human' (e.g. env.mode = 'fast')
        - 'fast' (default) will run as fast as possible (~75 fps) (best for simulation)
        - 'normal' will run at roughly 35 fps (easier for human to watch)
        - 'human' will let you play the game (keyboard only: Arrow Keys, '<', '>' and Ctrl)

    Ends when:
        - Player touches vest
        - Player is dead
        - Timeout (1 minutes - 2,100 frames)

    Actions:
        actions = [0] * 43
        actions[0] = 0       # ATTACK
        actions[10] = 1      # MOVE_RIGHT
        actions[11] = 0      # MOVE_LEFT
        actions[13] = 0      # MOVE_FORWARD
        actions[14] = 0      # TURN_RIGHT
        actions[15] = 0      # TURN_LEFT
    -----------------------------------------------------
    """
    def __init__(self):
        super(DoomCorridorEnv, self).__init__(1)
