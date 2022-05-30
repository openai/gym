import numpy as np
import pytest

from gym.spaces import Box
from gym.utils.env_checker import check_env


# This runs a smoketest on each official registered env. We may want
# to try also running environments which are not officially registered
# envs.
@pytest.mark.filterwarnings(
    "ignore:.*We recommend you to use a symmetric and normalized Box action space.*"
)
@pytest.mark.parametrize(
    "spec", spec_list_no_mujoco_py, ids=[spec.id for spec in spec_list_no_mujoco_py]
)
def test_env_gym_api(spec):
    # Capture warnings
    with pytest.warns(None):
        env = spec.make(disable_env_check=True)

    # Test if env adheres to Gym API
    check_env(env, warn=True, skip_render_check=True)

    ob_space = env.observation_space
    act_space = env.action_space
    ob = env.reset()
    assert ob_space.contains(ob), f"Reset observation: {ob!r} not in space"
    if isinstance(ob_space, Box):
        # Only checking dtypes for Box spaces to avoid iterating through tuple entries
        assert (
            ob.dtype == ob_space.dtype
        ), f"Reset observation dtype: {ob.dtype}, expected: {ob_space.dtype}"

    a = act_space.sample()
    observation, reward, done, _info = env.step(a)
    assert ob_space.contains(
        observation
    ), f"Step observation: {observation!r} not in space"
    assert np.isscalar(reward), f"{reward} is not a scalar for {env}"
    assert isinstance(done, bool), f"Expected {done} to be a boolean"
    if isinstance(ob_space, Box):
        assert (
            observation.dtype == ob_space.dtype
        ), f"Step observation dtype: {ob.dtype}, expected: {ob_space.dtype}"
    for mode in env.metadata.get("render_modes", []):
        if not (mode == "human" and spec.entry_point.startswith("gym.envs.mujoco")):
            env.render(mode=mode)

    # Make sure we can render the environment after close.
    for mode in env.metadata.get("render_modes", []):
        if not (mode == "human" and spec.entry_point.startswith("gym.envs.mujoco")):

            env.render(mode=mode)

    env.close()


@pytest.mark.parametrize("spec", spec_list, ids=[spec.id for spec in spec_list])
def test_reset_info(spec):
    with pytest.warns(None):
        env = spec.make(disable_env_check=True)

    obs = env.reset()
    assert obs in env.observation_space

    obs = env.reset(return_info=False)
    assert obs in env.observation_space

    obs, info = env.reset(return_info=True)
    assert obs in env.observation_space
    assert isinstance(info, dict)

    env.close()


# Note that this precludes running this test in multiple threads.
# However, we probably already can't do multithreading due to some environments.
SEED = 0
NUM_STEPS = 50


@pytest.mark.parametrize("env", testing_envs, ids=[env.id for env in testing_envs])
def test_env_determinism_rollout(env):
    """Run a rollout with two environments and assert equality.

    This test run a rollout of NUM_STEPS steps with two environments
    initialized with the same seed and assert that:

    - observation after first reset are the same
    - same actions are sampled by the two envs
    - observations are contained in the observation space
    - obs, rew, done and info are equals between the two envs

    Args:
        env (gym.Env): Environment
    """
    # Don't check rollout equality if it's a nondeterministic environment.
    if env.spec.nondeterministic is True:
        return

    env_1 = env.spec.make(disable_env_checker=True)
    env_2 = env.spec.make(disable_env_checker=True)

    initial_obs_1 = env_1.reset(seed=SEED)
    initial_obs_2 = env_2.reset(seed=SEED)
    assert_equals(initial_obs_1, initial_obs_2)

    env_1.action_space.seed(SEED)

    for time_step in range(NUM_STEPS):
        # We don't evaluate the determinism of actions
        action = env_1.action_space.sample()

        obs_1, rew_1, done_1, info_1 = env_1.step(action)
        obs_2, rew_2, done_2, info_2 = env_2.step(action)

        assert_equals(obs_1, obs_2, f"{time_step=} ")
        assert env_1.observation_space.contains(obs_1)  # obs_2 verified by previous assertion

        assert rew_1 == rew_2, f"{time_step=} {rew_1=}, {rew_2=}"
        assert done_1 == done_2, f"{time_step=} {done_1=}, {done_2=}"
        assert_equals(info_1, info_2, f"{time_step=} ")

        if done_1:  # done_2 verified by previous assertion
            env_1.reset(seed=SEED)
            env_2.reset(seed=SEED)

    env_1.close()
    env_2.close()