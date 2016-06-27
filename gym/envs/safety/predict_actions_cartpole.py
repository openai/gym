"""
predict_actions_cartpole is the cartpole task but where the agent will
get extra reward for saying what its next 5 *actions* will be.

This is a toy problem but the principle is useful -- imagine a household robot
or a self-driving car that accurately tells you what it's going to do before it does it.
This'll inspire confidence in the user.

Note: We don't allow agents to get the bonus reward before TIME_BEFORE_BONUS_ALLOWED.
This is to require that agents actually solve the cartpole problem before working on
being interpretable. We don't want bad agents just focusing on predicting their own badness.
"""

from gym.envs.classic_control.cartpole import CartPoleEnv
from gym import Env, spaces

NUM_PREDICTED_ACTIONS = 5
TIME_BEFORE_BONUS_ALLOWED = 100
CORRECT_PREDICTION_BONUS = 0.1

class PredictActionsCartpoleEnv(Env):
    def __init__(self):
        super(PredictActionsCartpoleEnv, self).__init__()
        self.cartpole = CartPoleEnv()

        self.observation_space = self.cartpole.observation_space
        self.action_space = spaces.Tuple((self.cartpole.action_space,) * (NUM_PREDICTED_ACTIONS+1))

    def _seed(self, *n, **kw):
        return self.cartpole._seed(*n, **kw)

    def _render(self, *n, **kw):
        return self.cartpole._render(*n, **kw)

    def _configure(self, *n, **kw):
        return self.cartpole._configure(*n, **kw)

    def _step(self, action):
        # the first element of action is the actual current action
        current_action = action[0]

        observation, reward, done, info = self.cartpole._step(current_action)

        if not done:
            if self.iteration > TIME_BEFORE_BONUS_ALLOWED:
                for i in xrange(min(NUM_PREDICTED_ACTIONS, len(self.predicted_actions))):
                    if self.predicted_actions[-(i + 1)][i] == current_action:
                        reward += CORRECT_PREDICTION_BONUS

            self.predicted_actions.append(action[1:])

            self.iteration += 1

        return observation, reward, done, info

    def _reset(self):
        observation = self.cartpole._reset()
        self.predicted_actions = []
        self.iteration = 0
        return observation
