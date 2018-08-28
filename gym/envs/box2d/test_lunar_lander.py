from .lunar_lander import LunarLander, LunarLanderContinuous, enjoy_lander

def test_lunar_lander():
    _test_lander(LunarLander(), seed=0)

def test_lunar_lander_continuous():
    _test_lander(LunarLanderContinuous(), seed=0)

def _test_lander(env, seed=None, render=False):
    total_reward = enjoy_lander(env, seed=seed, render=render)
    assert total_reward > 100


