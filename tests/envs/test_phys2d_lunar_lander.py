import pytest

try:
    import brax

    from gym.envs.phys2d.lunar_lander import LunarLander, demo_heuristic_lander
except ImportError:
    brax = None


@pytest.mark.skipif(brax is None, reason="brax not installed")
def test_lunar_lander():
    _test_lander(LunarLander(), seed=0)


@pytest.mark.skipif(brax is None, reason="brax not installed")
def test_lunar_lander_continuous():
    _test_lander(LunarLander(continuous=True), seed=0)


@pytest.mark.skipif(brax is None, reason="brax not installed")
def _test_lander(env, seed=None, render=False):
    total_reward = demo_heuristic_lander(env, seed=seed, render=render)
    assert total_reward > 100
