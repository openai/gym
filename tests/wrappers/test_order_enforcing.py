import pytest

import gym
from gym.envs.classic_control import CartPoleEnv
from gym.wrappers import OrderEnforcing
from tests.envs.spec_list import spec_list
from tests.wrappers.utils import has_wrapper


@pytest.mark.parametrize("spec", spec_list, ids=[spec.id for spec in spec_list])
def test_gym_make_order_enforcing(spec):
    env = gym.make(spec.id)

    assert has_wrapper(env, OrderEnforcing)


def test_order_enforcing():
    # The reason for not using gym.make is that all environments are by default wrapped in the order enforcing wrapper
    env = CartPoleEnv()
    assert not has_wrapper(env, OrderEnforcing)

    # Assert that the order enforcing works for step and render before reset
    order_enforced_env = OrderEnforcing(env)
    assert order_enforced_env._has_reset is False
    with pytest.raises(AssertionError):
        order_enforced_env.step(0)
    with pytest.raises(AssertionError):
        order_enforced_env.render()

    # Assert that the Assertion errors are not raised after reset
    order_enforced_env.reset()
    assert order_enforced_env._has_reset is True
    order_enforced_env.step(0)
    order_enforced_env.render()

    # Assert that with disable_render_order_enforcing works
    env = CartPoleEnv()
    env = OrderEnforcing(env, disable_render_order_enforcing=True)
    env.render()  # no assertion error
