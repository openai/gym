"""Test environment determinism by performing a rollout."""

import pytest

from gym.utils.env_checker import data_equivalence
from tests.envs.spec_list import spec_list


@pytest.mark.parametrize("spec", spec_list, ids=[spec.id for spec in spec_list])
def test_env(spec):
    """Run a rollout with two environments and assert equality.

    This test run a rollout of NUM_STEPS steps with two environments
    initialized with the same seed and assert that:

    - observation after first reset are the same
    - same actions are sampled by the two envs
    - observations are contained in the observation space
    - obs, rew, done and info are equals between the two envs

    Args:
        spec (EnvSpec): Environment specification

    """
    # Note that this precludes running this test in multiple
    # threads. However, we probably already can't do multithreading
    # due to some environments.
    SEED = 0
    NUM_STEPS = 50

    env1, env2 = spec.make(), spec.make()

    initial_observation1 = env1.reset(seed=SEED)
    initial_observation2 = env2.reset(seed=SEED)

    env1.action_space.seed(SEED)
    env2.action_space.seed(SEED)

    assert data_equivalence(
        initial_observation1, initial_observation2
    ), f"Initial Observations 1 and 2 are not equivalent. initial obs 1={initial_observation1}, initial obs 2={initial_observation2}"

    for i in range(NUM_STEPS):
        action1 = env1.action_space.sample()
        action2 = env2.action_space.sample()

        try:
            assert data_equivalence(
                action1, action2
            ), f"Action 1 and 2 are not equivalent. action 1={action1}, action 2={action2}"
        except AssertionError:
            print(f"env 1 action space={env1.action_space}")
            print(f"env 2 action space={env2.action_space}")
            print(f"[{i}] action sample 1={action1}, action sample 2={action2}")
            raise

        # Don't check rollout equality if it's a nondeterministic
        # environment.
        if spec.nondeterministic:
            return

        obs1, rew1, done1, info1 = env1.step(action1)
        obs2, rew2, done2, info2 = env2.step(action2)

        assert data_equivalence(
            obs1, obs2
        ), f"Observation 1 and 2 are not equivalent. obs 1={obs1}, obs 2={obs2}"

        assert env1.observation_space.contains(obs1)
        assert env2.observation_space.contains(obs2)

        assert rew1 == rew2, f"[{i}] reward1: {rew1}, reward2: {rew2}"
        assert done1 == done2, f"[{i}] done1: {done1}, done2: {done2}"
        assert data_equivalence(
            info1, info2
        ), f"Info 1 and 2 are not equivalent. info 1={info1}, info 2={info2}"

        if done1:  # done2 verified in previous assertion
            env1.reset(seed=SEED)
            env2.reset(seed=SEED)

    env1.close()
    env2.close()
