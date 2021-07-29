from gym import RewardWrapper
import warnings


class TransformReward(RewardWrapper):
    r"""Transform the reward via an arbitrary function.

    Example::

        >>> import gym
        >>> env = gym.make('CartPole-v1')
        >>> env = TransformReward(env, lambda r: 0.01*r)
        >>> env.reset()
        >>> observation, reward, done, info = env.step(env.action_space.sample())
        >>> reward
        0.01

    Args:
        env (Env): environment
        f (callable): a function that transforms the reward

    """

    def __init__(self, env, f):
        super(TransformReward, self).__init__(env)
        assert callable(f)
        warnings.warn("Gym\'s internal preprocessing wrappers are now deprecated. While they will continue to work for the foreseeable future, we strongly recommend using SuperSuit instead: https://github.com/PettingZoo-Team/SuperSuit")
        self.f = f

    def reward(self, reward):
        return self.f(reward)
