import logging
from gym.envs.doom import doom_env

logger = logging.getLogger(__name__)

class DoomHealthGatheringEnv(doom_env.DoomEnv):
    """
    ------------ Training Mission 5 - Health Gathering ------------
    This map is a guide on how to survive by collecting health packs.
    It is a rectangle with green, acidic floor which hurts the player
    periodically. There are also medkits spread around the map, and
    additional kits will spawn at interval.

    Allowed actions:
        [13] - MOVE_FORWARD                     - Move forward - Values 0 or 1
        [14] - TURN_RIGHT                       - Turn right - Values 0 or 1
        [15] - TURN_LEFT                        - Turn left - Values 0 or 1
    Note: see controls.md for details

    Rewards:
        +  1    - 35 times per second - Survive as long as possible
        -100    - Death penalty

    Goal: 1000 points
        Stay alive long enough to reach 1,000 points (~ 30 secs)

    Mode:
        - env.mode can be 'fast', 'normal' or 'human' (e.g. env.mode = 'fast')
        - 'fast' (default) will run as fast as possible (~75 fps) (best for simulation)
        - 'normal' will run at roughly 35 fps (easier for human to watch)
        - 'human' will let you play the game (keyboard only: Arrow Keys, '<', '>' and Ctrl)

    Ends when:
        - Player is dead
        - Timeout (60 seconds - 2,100 frames)

    Actions:
        actions = [0] * 43
        actions[13] = 0      # MOVE_FORWARD
        actions[14] = 1      # TURN_RIGHT
        actions[15] = 0      # TURN_LEFT
    -----------------------------------------------------
    """
    def __init__(self):
        super(DoomHealthGatheringEnv, self).__init__(4)
