"""
predict_obs_cartpole is the cartpole task but where the agent will
get extra reward for saying what it expects its next 5 *observations* will be.

This is a toy problem but the principle is useful -- imagine a household robot
or a self-driving car that accurately tells you what it expects to percieve after
taking a certain plan of action. This'll inspire confidence in the user.

Note: We don't allow agents to get the bonus reward before TIME_BEFORE_BONUS_ALLOWED.
This is to require that agents actually solve the cartpole problem before working on
being interpretable. We don't want bad agents just focusing on predicting their own badness.
"""

from gym.envs.classic_control.cartpole import CartPoleEnv
from gym import Env, spaces

import numpy as np
import math

NUM_PREDICTED_OBSERVATIONS = 5
TIME_BEFORE_BONUS_ALLOWED = 100

# this is the bonus reward for perfectly predicting one observation
# bonus decreases smoothly as prediction gets farther from actual observation
CORRECT_PREDICTION_BONUS = 0.1

class PredictObsCartpoleEnv(Env):
    def __init__(self):
        super(PredictObsCartpoleEnv, self).__init__()
        self.cartpole = CartPoleEnv()

        self.observation_space = self.cartpole.observation_space
        self.action_space = spaces.Tuple((self.cartpole.action_space,) + (self.cartpole.observation_space,) * (NUM_PREDICTED_OBSERVATIONS))

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
            # We add the newly predicted observations to the list before checking predictions
            # in order to give the agent a chance to predict the observations that they
            # are going to get _this_ round.
            self.predicted_observations.append(action[1:])

            if self.iteration > TIME_BEFORE_BONUS_ALLOWED:
                for i in xrange(min(NUM_PREDICTED_OBSERVATIONS, len(self.predicted_observations))):
                    l2dist = np.sqrt(np.sum(np.square(np.subtract(
                        self.predicted_observations[-(i + 1)][i],
                        observation
                    ))))

                    bonus = CORRECT_PREDICTION_BONUS * (1 - math.erf(l2dist))

                    reward += bonus

            self.iteration += 1

        return observation, reward, done, info

    def _reset(self):
        observation = self.cartpole._reset()
        self.predicted_observations = []
        self.iteration = 0
        return observation
