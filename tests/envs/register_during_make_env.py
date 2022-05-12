import gym


class RegisterDuringMakeEnv(gym.Env):
    """Used in `test_registration.py` to check if `env.make` can import and register an env"""

    def __init__(self):
        super().__init__()
        self.act_space = gym.spaces.Discrete(2)
        self.obs_space = gym.spaces.Discrete(2)

    def reset(self, return_info=True):
        return self.obs_space.sample(), {}

    def step(self, action):
        return self.obs_space.sample(), 0, False, {}


gym.register(
    "RegisterDuringMakeEnv-v0",
    entry_point="tests.envs.register_during_make_env:RegisterDuringMakeEnv",
)
