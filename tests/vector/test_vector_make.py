import pytest

import gym
from gym.vector import AsyncVectorEnv, SyncVectorEnv
from gym.wrappers import OrderEnforcing, TimeLimit, TransformObservation
from gym.wrappers.env_checker import PassiveEnvChecker
from tests.wrappers.utils import has_wrapper


def test_vector_make_id():
    env = gym.vector.make("CartPole-v1")
    assert isinstance(env, AsyncVectorEnv)
    assert env.num_envs == 1
    env.close()


@pytest.mark.parametrize("num_envs", [1, 3, 10])
def test_vector_make_num_envs(num_envs):
    env = gym.vector.make("CartPole-v1", num_envs=num_envs)
    assert env.num_envs == num_envs
    env.close()


def test_vector_make_asynchronous():
    env = gym.vector.make("CartPole-v1", asynchronous=True)
    assert isinstance(env, AsyncVectorEnv)
    env.close()

    env = gym.vector.make("CartPole-v1", asynchronous=False)
    assert isinstance(env, SyncVectorEnv)
    env.close()


def test_vector_make_wrappers():
    env = gym.vector.make("CartPole-v1", num_envs=2, asynchronous=False)
    assert isinstance(env, SyncVectorEnv)
    assert len(env.envs) == 2

    sub_env = env.envs[0]
    assert isinstance(sub_env, gym.Env)
    if sub_env.spec.order_enforce:
        assert has_wrapper(sub_env, OrderEnforcing)
    if sub_env.spec.max_episode_steps is not None:
        assert has_wrapper(sub_env, TimeLimit)

    assert all(
        has_wrapper(sub_env, TransformObservation) is False for sub_env in env.envs
    )
    env.close()

    env = gym.vector.make(
        "CartPole-v1",
        num_envs=2,
        asynchronous=False,
        wrappers=lambda _env: TransformObservation(_env, lambda obs: obs * 2),
    )
    # As asynchronous environment are inaccessible, synchronous vector must be used
    assert isinstance(env, SyncVectorEnv)
    assert all(has_wrapper(sub_env, TransformObservation) for sub_env in env.envs)

    env.close()


def test_vector_make_disable_env_checker():
    # As asynchronous environment are inaccessible, synchronous vector must be used
    env = gym.vector.make("CartPole-v1", num_envs=1, asynchronous=False)
    assert isinstance(env, SyncVectorEnv)
    assert has_wrapper(env.envs[0], PassiveEnvChecker)
    env.close()

    env = gym.vector.make("CartPole-v1", num_envs=5, asynchronous=False)
    assert isinstance(env, SyncVectorEnv)
    assert has_wrapper(env.envs[0], PassiveEnvChecker)
    assert all(
        has_wrapper(env.envs[i], PassiveEnvChecker) is False for i in [1, 2, 3, 4]
    )
    env.close()

    env = gym.vector.make(
        "CartPole-v1", num_envs=3, asynchronous=False, disable_env_checker=True
    )
    assert isinstance(env, SyncVectorEnv)
    assert all(has_wrapper(sub_env, PassiveEnvChecker) is False for sub_env in env.envs)
    env.close()
