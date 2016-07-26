import logging
from gym.envs.soccer import soccer_empty_goal

logger = logging.getLogger(__name__)

class SoccerAgainstKeeperEnv(soccer_empty_goal.SoccerEmptyGoalEnv):
    """
    SoccerAgainstKeeper initializes the agent most of the way down the
    field with the ball and tasks it with scoring on a keeper.

    Rewards in this task are the same as SoccerEmptyGoal: reward
    is given for kicking the ball close to the goal and extra reward is
    given for scoring a goal.

    """
    def __init__(self):
        super(SoccerAgainstKeeperEnv, self).__init__()

    def _configure_environment(self):
        super(SoccerAgainstKeeperEnv, self)._start_hfo_server(defense_npcs=1,
                                                              offense_on_ball=1,
                                                              ball_x_min=0.6)
