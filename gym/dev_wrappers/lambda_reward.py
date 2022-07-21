"""Lambda reward wrappers that uses jumpy for compatibility with jax (i.e. brax) and numpy environments."""
from typing import Callable, Optional, Union

import jumpy as jp

import gym
from gym.error import InvalidBound


class lambda_reward_v0(gym.RewardWrapper):
    """A reward wrapper that allows a custom function to modify the step reward.

    Example:
        >>> import gym
        >>> env = gym.make("CartPole-v1")
        >>> env = lambda_reward_v0(env, lambda r: 2 * r + 1)
        >>> env.reset()
        >>> _, rew, _, _ = env.step(0)
        >>> rew
        3.0
    """

    def __init__(
        self,
        env: gym.Env,
        fn: Callable[[Union[float, int, jp.ndarray]], Union[float, int, jp.ndarray]],
    ):
        """Initialize lambda_reward_v0 wrapper.

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


class clip_rewards_v0(lambda_reward_v0):
    """A wrapper that clips the rewards for an environment between an upper and lower bound.

    Example with an upper and lower bound:
        >>> import gym
        >>> env = gym.make("CartPole-v1")
        >>> env = clip_rewards_v0(env, 0, 0.5)
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
        """Initialize clip_reward_v0 wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            min_reward (Union[float, jp.ndarray]): lower bound to apply
            max_reward (Union[float, jp.ndarray]): higher bound to apply
        """
        if min_reward is None and max_reward is None:
            raise InvalidBound("Both `min_reward` and `max_reward` cannot be None")
        elif max_reward and min_reward and max_reward < min_reward:
            raise InvalidBound(
                f"Min reward ({min_reward}) must be less than max reward ({max_reward})"
            )
        else:
            super().__init__(
                env, lambda x: jp.clip(x, a_min=min_reward, a_max=max_reward)
            )
