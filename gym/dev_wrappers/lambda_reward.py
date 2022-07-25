"""Lambda reward wrappers that uses jumpy for compatibility with jax (i.e. brax) and numpy environments."""
from typing import Callable, Optional, Union

import jax.numpy as jnp
import jumpy as jp
import numpy as np

import gym
from gym.dev_wrappers import FuncArgType
from gym.error import InvalidBound


class LambdaRewardV0(gym.RewardWrapper):
    """A reward wrapper that allows a custom function to modify the step reward.

    Example:
        >>> import gym
        >>> env = gym.make("CartPole-v1")
        >>> env = LambdaRewardV0(env, lambda r: 2 * r + 1)
        >>> env.reset()
        >>> _, rew, _, _ = env.step(0)
        >>> rew
        3.0
    """

    def __init__(
        self,
        env: gym.Env,
        fn: Callable[[FuncArgType], Union[float, int, jp.ndarray]],
    ):
        """Initialize LambdaRewardV0 wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            fn: (Callable): The function to apply to reward
        """
        super().__init__(env)

        self.fn = fn

    def reward(self, reward: Union[float, int, jp.ndarray]):
        """Apply function to reward.

        Args:
            reward (Union[float, int, jp.ndarray]): environment's reward
        """
        return self.fn(reward)


class ClipRewardsV0(LambdaRewardV0):
    """A wrapper that clips the rewards for an environment between an upper and lower bound.

    Example with an upper and lower bound:
        >>> import gym
        >>> env = gym.make("CartPole-v1")
        >>> env = ClipRewardsV0(env, 0, 0.5)
        >>> env.reset()
        >>> _, rew, _, _ = env.step(1)
        >>> rew
        0.5
    """

    def __init__(
        self,
        env: gym.Env,
        min_reward: Optional[Union[float, jp.ndarray]] = None,
        max_reward: Optional[Union[float, jp.ndarray]] = None,
    ):
        """Initialize ClipRewardsV0 wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            min_reward (Union[float, jp.ndarray]): lower bound to apply
            max_reward (Union[float, jp.ndarray]): higher bound to apply
        """
        if min_reward is None and max_reward is None:
            raise InvalidBound("Both `min_reward` and `max_reward` cannot be None")

        elif max_reward is not None and min_reward is not None:
            array_bounds = isinstance(
                min_reward, (np.ndarray, jnp.ndarray)
            ) or isinstance(max_reward, (np.ndarray, jnp.ndarray))

            invalid_bounds = (
                any(max_reward < min_reward)
                if array_bounds
                else max_reward < min_reward
            )

            if invalid_bounds:
                raise InvalidBound(
                    f"Min reward ({min_reward}) must be smaller than max reward ({max_reward})"
                )

        super().__init__(env, lambda x: jp.clip(x, a_min=min_reward, a_max=max_reward))
