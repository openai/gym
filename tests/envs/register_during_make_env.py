import gym


class RegisterDuringMakeEnv(gym.Env):
    """Used in `test_registration.py` to check if `env.make` can import and register an env"""

    def __init__(self):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, return_info=True, options=None):
        return 0, {}

    def step(self, action):
        return 0, 0, False, {}


gym.register(
    "RegisterDuringMakeEnv-v0",
    entry_point="tests.envs.register_during_make_env:RegisterDuringMakeEnv",
)
