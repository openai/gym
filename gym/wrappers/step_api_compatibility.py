"""Implementation of StepAPICompatibility wrapper class for transforming envs between new and old step API."""
import gym
from gym.logger import deprecation
from gym.utils.step_api_compatibility import (
    convert_to_done_step_api,
    convert_to_terminated_truncated_step_api,
)


class StepAPICompatibility(gym.Wrapper):
    r"""A wrapper which can transform an environment from new step API to old and vice-versa.

    Old step API refers to step() method returning (observation, reward, done, info)
    New step API refers to step() method returning (observation, reward, terminated, truncated, info)
    (Refer to docs for details on the API change)

    Args:
        env (gym.Env): the env to wrap. Can be in old or new API
        apply_step_compatibility (bool): Apply to convert environment to use new step API that returns two bools. (False by default)

    Examples:
        >>> env = gym.make("CartPole-v1")
        >>> env # wrapper not applied by default, set to new API
        <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
        >>> env = gym.make("CartPole-v1", apply_api_compatibility=True) # set to old API
        <StepAPICompatibility<TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>>
        >>> env = StepAPICompatibility(CustomEnv(), apply_step_compatibility=False) # manually using wrapper on unregistered envs

    """

    def __init__(self, env: gym.Env, output_truncation_bool: bool = True):
        """A wrapper which can transform an environment from new step API to old and vice-versa.

        Args:
            env (gym.Env): the env to wrap. Can be in old or new API
            output_truncation_bool (bool): Whether the wrapper's step method outputs two booleans (new API) or one boolean (old API)
        """
        super().__init__(env)
        self.output_truncation_bool = output_truncation_bool
        if not self.output_truncation_bool:
            deprecation(
                "Initializing environment in old step API which returns one bool instead of two."
            )

    def step(self, action):
        """Steps through the environment, returning 5 or 4 items depending on `apply_step_compatibility`.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info) or (observation, reward, done, info)
        """
        step_returns = self.env.step(action)
        if self.output_truncation_bool:
            return convert_to_terminated_truncated_step_api(step_returns)
        else:
            return convert_to_done_step_api(step_returns)
