import pytest

import gym
import numpy as np


ENV_ID = "CartPole-v1"
NUM_ENVS = 3
ENV_STEPS = 50


@pytest.mark.parametrize("asynchronous", [True, False])
def test_vector_env_info(asynchronous):
    env = gym.vector.make(ENV_ID, num_envs=NUM_ENVS, asynchronous=asynchronous)
    env.reset()
    for _ in range(ENV_STEPS):
        action = env.action_space.sample()
        _, _, dones, infos = env.step(action)
        if any(dones):
            assert len(infos['terminal_observation']) == NUM_ENVS
            assert len(infos['_terminal_observation']) == NUM_ENVS
            
            assert isinstance(infos['terminal_observation'], np.ndarray)
            assert isinstance(infos['_terminal_observation'], np.ndarray)

            for i, done in enumerate(dones):
                if done:
                    assert infos['_terminal_observation'][i]
                else:
                    assert not infos['_terminal_observation'][i]
                    assert infos['terminal_observation'][i] == None
