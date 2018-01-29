from copy import deepcopy
from gym.spaces import Dict


class GoalDict(Dict):
    """
    A goal-based space, in which every state has a goal, an achieved goal, and the current observation.
    You can use any other space type for the goal and observation spaces, respectively.

    Example usage:
    self.goal_space = spaces.Goal(observation_space=(spaces.Discrete(2), goal_space=spaces.Discrete(3)))
    """
    def __init__(self, observation_space, goal_space):
        self.observation_space = observation_space
        self.goal_space = goal_space
        self.achieved_goal_space = deepcopy(goal_space)
        spaces = {
            'observation': self.observation_space,
            'goal': self.goal_space,
            'achieved_goal': self.achieved_goal_space,
        }
        super(GoalDict, self).__init__(spaces=spaces)

    def __repr__(self):
        return "GoalDict(" + ", ". join([k + ":" + str(s) for k, s in self.spaces.items()]) + ")"

    def from_jsonable(self, sample_n):
        super(Goal, self).from_jsonable(sample_n)
        self.observation_space = self.spaces['observation']
        self.goal_space = self.spaces['goal']
        self.achieved_goal = self.spaces['achieved_goal']
