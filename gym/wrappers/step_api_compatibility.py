import gym
from gym.logger import deprecation
from gym.utils.step_api_compatibility import step_to_new_api, step_to_old_api


class StepAPICompatibility(gym.Wrapper):
    r"""A wrapper which can transform an environment from new step API to old and vice-versa.

    Old step API refers to step() method returning (observation, reward, done, info)
    New step API refers to step() method returning (observation, reward, terminated, truncated, info)
    (Refer to docs for details on the API change)

    This wrapper is to be used to ease transition to new API and for backward compatibility.

    Args:
        env (gym.Env): the env to wrap. Can be in old or new API
        new_step_api (bool): True to use env with new step API, False to use env with old step API. (False by default)

    Examples:
        >>> env = gym.make("CartPole-v1")
        >>> env # wrapper applied by default, set to old API
        <TimeLimit<OrderEnforcing<StepAPICompatibility<CartPoleEnv<CartPole-v1>>>>>
        >>> env = gym.make("CartPole-v1", new_step_api=True) # set to new API
        >>> env = StepAPICompatibility(CustomEnv(), new_step_api=True) # manually using wrapper on unregistered envs

    """

    def __init__(self, env: gym.Env, new_step_api=False):
        super().__init__(env)
        self.new_step_api = new_step_api
        if not self.new_step_api:
            deprecation(
                "Initializing environment in old step API which returns one bool instead of two. "
                "It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future. "
            )

    def step(self, action):
        step_returns = self.env.step(action)
        if self.new_step_api:
            return step_to_new_api(step_returns)
        else:
            return step_to_old_api(step_returns)
