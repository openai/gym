import gym


class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.act_space = gym.spaces.Discrete(2)
        self.obs_space = gym.spaces.Discrete(2)

    def reset(self):
        return self.obs_space.sample()

    def step(self, action):
        return self.obs_space.sample(), 0, False, {}


gym.register("Dummy-v0", entry_point="tests.envs.dummy_env:DummyEnv")
