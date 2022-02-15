import gym


def test_onehot_observation():
    env = gym.make("Taxi-v3")
    env = gym.wrappers.OnehotObservation(env)
    obs = env.reset()
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (500,)
    assert env.observation_space.contains(obs)
