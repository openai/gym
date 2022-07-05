import pytest

import gym
from gym.envs.classic_control import CartPoleEnv
from gym.error import ResetNeeded
from gym.wrappers import OrderEnforcing
from tests.envs.utils import all_testing_env_specs
from tests.wrappers.utils import has_wrapper


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_gym_make_order_enforcing(spec):
    """Checks that gym.make wrappers the environment with the OrderEnforcing wrapper."""
    env = gym.make(spec.id, disable_env_checker=True)

    assert has_wrapper(env, OrderEnforcing)


def test_order_enforcing():
    """Checks that the order enforcing works as expected, raising an error before reset is called and not after."""
    # The reason for not using gym.make is that all environments are by default wrapped in the order enforcing wrapper
    env = CartPoleEnv()
    assert not has_wrapper(env, OrderEnforcing)

    # Assert that the order enforcing works for step and render before reset
    order_enforced_env = OrderEnforcing(env)
    assert order_enforced_env.has_reset is False
    with pytest.raises(ResetNeeded):
        order_enforced_env.step(0)
    with pytest.raises(ResetNeeded):
        order_enforced_env.render(mode="rgb_array")
    assert order_enforced_env.has_reset is False

    # Assert that the Assertion errors are not raised after reset
    order_enforced_env.reset()
    assert order_enforced_env.has_reset is True
    order_enforced_env.step(0)
    order_enforced_env.render(mode="rgb_array")

    # Assert that with disable_render_order_enforcing works, the environment has already been reset
    env = CartPoleEnv()
    env = OrderEnforcing(env, disable_render_order_enforcing=True)
    env.render(mode="rgb_array")  # no assertion error
