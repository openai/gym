"""Implementation of StepAPICompatibility wrapper class for transforming envs between new and old step API."""
import gym
from gym.logger import deprecation
from gym.utils.step_api_compatibility import step_to_new_api, step_to_old_api


class StepAPICompatibility(gym.Wrapper):
    r"""A wrapper which can transform an environment from new step API to old and vice-versa.

    Old step API refers to step() method returning (observation, reward, done, info)
    New step API refers to step() method returning (observation, reward, terminated, truncated, info)
    (Refer to docs for details on the API change)

    Args:
        env (gym.Env): the env to wrap. Can be in old or new API
        new_step_api (bool): True to use env with new step API, False to use env with old step API. (True by default)

    Examples:
        >>> env = gym.make("CartPole-v1")
        >>> env # wrapper not applied by default, set to new API
        <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
        >>> env = gym.make("CartPole-v1", new_step_api=False) # set to old API
        <StepAPICompatibility<TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>>
        >>> env = StepAPICompatibility(CustomEnv(), new_step_api=True) # manually using wrapper on unregistered envs

    """

    def __init__(self, env: gym.Env, new_step_api=True):
        """A wrapper which can transform an environment from new step API to old and vice-versa.

        Args:
            env (gym.Env): the env to wrap. Can be in old or new API
            new_step_api (bool): Whether the wrapper's step method outputs two booleans (new API) or one boolean (old API)
        """
        super().__init__(env)
        self.new_step_api = new_step_api
        if not self.new_step_api:
            deprecation(
                "Initializing environment in old step API which returns one bool instead of two."
            )

    def step(self, action):
        """Steps through the environment, returning 5 or 4 items depending on `new_step_api`.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info) or (observation, reward, done, info)
        """
        step_returns = self.env.step(action)
        if self.new_step_api:
            return step_to_new_api(step_returns)
        else:
            return step_to_old_api(step_returns)
