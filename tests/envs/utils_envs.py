import gym


class RegisterDuringMakeEnv(gym.Env):
    """Used in `test_registration.py` to check if `env.make` can import and register an env"""

    def __init__(self):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(1)


class ArgumentEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
    action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

    def __init__(self, arg1, arg2, arg3):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
