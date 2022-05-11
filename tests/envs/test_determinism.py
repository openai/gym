import numpy as np
import pytest

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

    assert_equals(initial_observation1, initial_observation2)

    for i in range(NUM_STEPS):
        action1 = env1.action_space.sample()
        action2 = env2.action_space.sample()
        
        try:
            assert_equals(action1, action2)
        except AssertionError:
            print("env1.action_space=", env1.action_space)
            print("env2.action_space=", env2.action_space)
            print("action_samples1=", action1)
            print("action_samples2=", action2)
            print(
                f"[{i}] action_sample1: {action1}, action_sample2: {action2}"
            )
            raise

        # Don't check rollout equality if it's a a nondeterministic
        # environment.
        if spec.nondeterministic:
            return

        obs1, rew1, done1, info1 = env1.step(action1)
        obs2, rew2, done2, info2 = env2.step(action2)

        assert_equals(obs1, obs2, f"[{i}] ")
        
        assert env1.observation_space.contains(obs1)
        assert env2.observation_space.contains(obs2)
        
        assert rew1 == rew2, f"[{i}] r1: {rew1}, r2: {rew2}"
        assert done1 == done2, f"[{i}] d1: {done1}, d2: {done2}"

        # Go returns a Pachi game board in info, which doesn't
        # properly check equality. For now, we hack around this by
        # just skipping Go.
        if spec.id not in ["Go9x9-v0", "Go19x19-v0"]:
            assert_equals(info1, info2, f"[{i}] ")

        if done1: # done2 verified in previous assertion
            env1.reset(seed=SEED)
            env2.reset(seed=SEED)
    
    env1.close()
    env2.close()

    
def assert_equals(a, b, prefix=None):
    assert type(a) == type(b), f"{prefix}Differing types: {a} and {b}"
    if isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), f"{prefix}Key sets differ: {a} and {b}"

        for k in a.keys():
            v_a = a[k]
            v_b = b[k]
            assert_equals(v_a, v_b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, tuple):
        for elem_from_a, elem_from_b in zip(a, b):
            assert_equals(elem_from_a, elem_from_b)
    else:
        assert a == b
