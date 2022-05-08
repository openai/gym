import gym


class BraxInfoToClassic(gym.Wrapper):
    """This wrapper converts the `Brax` info format of a
    vector environment to the `classic` info format.

    Example::

    >>> # brax
    ...  {
    ...      k: np.array[0., 0., 0.5, 0.3],
    ...      _k: np.array[False, False, True, True]
    ...  }
    ...
    ... # classic
    ... [{}, {}, {k: 0.5}, {k: 0.3}]

    """

    def __init__(self, env):
        assert getattr(
            env, "is_vector_env", False
        ), "This wrapper can only be used in vectorized environments."
        super().__init__(env)

    def step(self, action):
        observation, reward, done, infos = self.env.step(action)
        # TODO
        return observation, reward, done, infos

    def reset(self, **kwargs):
        # TODO
        ...
