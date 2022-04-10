from dataclasses import dataclass
import pytest
import numpy as np
import gym

from gym.utils.play import PlayableGame
from gym.utils.play import play



@dataclass
class DummyEnvSpec():
    id: str


class DummyPlayEnv(gym.Env):
    
    def step(self, action):
        ...
        
    def reset(self):
        ...
        
    def render(self, mode):
        return np.zeros((1,1))


def dummy_keys_to_action():
    return {(ord('a'),): 0, (ord('d'),): 1}


def test_play_relvant_keys():
    env = DummyPlayEnv()
    keys_to_action = {
         (ord('a'),): 0,
         (ord('d'),): 1
    }
    game = PlayableGame(env)
    relevant_keys = game.get_relevant_keys(keys_to_action)
    assert relevant_keys == {97, 100}


def test_play_revant_keys_no_mapping():
    env = DummyPlayEnv()
    env.spec = DummyEnvSpec("DummyPlayEnv")
    game = PlayableGame(env)
    
    with pytest.raises(AssertionError) as info:
        game.get_relevant_keys()


def test_play_relevant_keys_with_env_attribute():
    """Env has a keys_to_action attribute
    """
    env = DummyPlayEnv()
    env.get_keys_to_action = dummy_keys_to_action
    game = PlayableGame(env)
    relevant_keys = game.get_relevant_keys()
    assert relevant_keys == {97, 100}






# def test_play_loop():
#     env = DummyPlayEnv()
#     keys_to_action = {
#         (ord('a'),): 0,
#         (ord('d'),): 1
#     }
#     play(env, keys_to_action=keys_to_action)