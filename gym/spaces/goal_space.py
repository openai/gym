from copy import deepcopy
from gym.spaces import Dict


class GoalDict(Dict):
    """
    A goal-based space, in which every state has a desired goal, an achieved goal, and the
    current observation. More concretely, the desired goal is the goal which the agent should
    attempt to achieve. The achieved goal is the one that it has currently achieved instead.
    The observation is your typical Gym observation, i.e. it contains the relevant state of
    the environment. You can use any other space type for the goal and observation spaces,
    respectively.

    Example usage:
        self.goal_space = spaces.Goal(
            observation_space=spaces.Discrete(2),
            goal_space=spaces.Discrete(3)
        )
    """
    def __init__(self, observation_space, goal_space):
        self.observation_space = observation_space
        self.goal_space = goal_space

        spaces = {
            'observation': self.observation_space,
            'desired_goal': self.goal_space,
            'achieved_goal': self.goal_space,
        }
        super(GoalDict, self).__init__(spaces=spaces)

    def __repr__(self):
        return "GoalDict(" + ", ". join([k + ":" + str(s) for k, s in self.spaces.items()]) + ")"

    def from_jsonable(self, sample_n):
        super(GoalDict, self).from_jsonable(sample_n)
        self.observation_space = self.spaces['observation']
        self.goal_space = self.spaces['desired_goal']
