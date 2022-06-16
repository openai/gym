import pytest

import gym
from gym.envs.registration import EnvSpec
from gym.utils.env_checker import check_env
from tests.envs.utils import all_testing_env_specs, assert_equals, gym_testing_env_specs

# This runs a smoketest on each official registered env. We may want
# to try also running environments which are not officially registered
# envs.
IGNORE_WARNINGS = [
    "Agent's minimum observation space value is -infinity. This is probably too low.",
    "Agent's maximum observation space value is infinity. This is probably too high.",
    "We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html",
]
IGNORE_WARNINGS = [f"\x1b[33mWARN: {message}\x1b[0m" for message in IGNORE_WARNINGS]


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_env(spec):
    # Capture warnings
    env = spec.make(disable_env_checker=True)

    # Test if env adheres to Gym API
    with pytest.warns(None) as warnings:
        check_env(env)

    for warning in warnings.list:
        if warning.message.args[0] not in IGNORE_WARNINGS:
            raise gym.error.Error(f"Unexpected warning: {warning.message}")


# Note that this precludes running this test in multiple threads.
# However, we probably already can't do multithreading due to some environments.
SEED = 0
NUM_STEPS = 50


@pytest.mark.parametrize(
    "env_spec", all_testing_env_specs, ids=[env.id for env in all_testing_env_specs]
)
def test_env_determinism_rollout(env_spec: EnvSpec):
    """Run a rollout with two environments and assert equality.

    This test run a rollout of NUM_STEPS steps with two environments
    initialized with the same seed and assert that:

    - observation after first reset are the same
    - same actions are sampled by the two envs
    - observations are contained in the observation space
    - obs, rew, done and info are equals between the two envs
    """
    # Don't check rollout equality if it's a nondeterministic environment.
    if env_spec.nondeterministic is True:
        return

    env_1 = env_spec.make(disable_env_checker=True)
    env_2 = env_spec.make(disable_env_checker=True)

    initial_obs_1 = env_1.reset(seed=SEED)
    initial_obs_2 = env_2.reset(seed=SEED)
    assert_equals(initial_obs_1, initial_obs_2)

    env_1.action_space.seed(SEED)

    for time_step in range(NUM_STEPS):
        # We don't evaluate the determinism of actions
        action = env_1.action_space.sample()

        obs_1, rew_1, done_1, info_1 = env_1.step(action)
        obs_2, rew_2, done_2, info_2 = env_2.step(action)

        assert_equals(obs_1, obs_2, f"[{time_step}] ")
        assert env_1.observation_space.contains(
            obs_1
        )  # obs_2 verified by previous assertion

        assert rew_1 == rew_2, f"[{time_step}] reward 1={rew_1}, reward 2={rew_2}"
        assert done_1 == done_2, f"[{time_step}] done 1={done_1}, done 2={done_2}"
        assert_equals(info_1, info_2, f"[{time_step}] ")

        if done_1:  # done_2 verified by previous assertion
            env_1.reset(seed=SEED)
            env_2.reset(seed=SEED)

    env_1.close()
    env_2.close()


@pytest.mark.parametrize(
    "spec", gym_testing_env_specs, ids=[spec.id for spec in gym_testing_env_specs]
)
def test_render_modes(spec):
    env = spec.make()

    for mode in env.metadata.get("render_modes", []):
        if mode != "human":
            new_env = spec.make(render_mode=mode)

            new_env.reset()
            new_env.step(new_env.action_space.sample())
            new_env.render()
