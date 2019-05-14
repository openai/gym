from gym import RewardWrapper


class ScaleReward(RewardWrapper):
    r"""Scale the reward.

    Example::

        >>> import gym
        >>> env = gym.make('CartPole-v1')
        >>> env = ScaleReward(env, scale=0.1)
        >>> env.reset()
        >>> observation, reward, done, info = env.step(env.action_space.sample())
        >>> reward
        0.1

    Args:
            env (Env): environment
            scale (float): reward scaling factor

    """
    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return self.scale*reward
