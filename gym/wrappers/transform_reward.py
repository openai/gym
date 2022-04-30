"""Wrapper for transforming the reward."""
from typing import Callable

import gym
from gym import RewardWrapper


class TransformReward(RewardWrapper):
    """Transform the reward via an arbitrary function.

    Example::

        >>> import gym
        >>> env = gym.make('CartPole-v1')
        >>> env = TransformReward(env, lambda r: 0.01*r)
        >>> env.reset()
        >>> observation, reward, done, info = env.step(env.action_space.sample())
        >>> reward
        0.01

    Args:
        env (Env): The environment to apply the wrapper
        f (callable): A function that transforms the reward

    """

    def __init__(self, env: gym.Env, f: Callable[[float], float]):
        """Initialise the transform_reward with an environment and reward transform function f.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the reward
        """
        super().__init__(env)
        assert callable(f)
        self.f = f

    def reward(self, reward):
        """Transforms the reward using callable f.

        Args:
            reward: The reward to transform

        Returns: The transformed reward
        """
        return self.f(reward)
