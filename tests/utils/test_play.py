from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import pygame
import pytest
from pygame import KEYDOWN, QUIT, event
from pygame.event import Event

import gym
from gym.utils.play import MissingKeysToAction, PlayableGame, play

RELEVANT_KEY = 100
IRRELEVANT_KEY = 1


@dataclass
class DummyEnvSpec:
    id: str


class DummyPlayEnv(gym.Env):
    def step(self, action):
        obs = np.zeros((1, 1))
        rew, done, info = 1, False, {}
        return obs, rew, done, info

    def reset(self):
        ...

    def render(self, mode="rgb_array"):
        return np.zeros((1, 1))


class PlayStatus:
    def __init__(self, callback: Callable):
        self.data_callback = callback
        self.cumulative_reward = 0

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        self.cumulative_reward += self.data_callback(
            obs_t, obs_tp1, action, rew, done, info
        )


# set of key events to inject into the play loop as callback
callback_events = [
    Event(KEYDOWN, {"key": RELEVANT_KEY}),
    Event(KEYDOWN, {"key": RELEVANT_KEY}),
    Event(QUIT),
]


def callback(obs_t, obs_tp1, action, rew, done, info):
    event.post(callback_events.pop(0))
    return rew


def dummy_keys_to_action():
    return {(ord("a"),): 0, (ord("d"),): 1}


@pytest.fixture(autouse=True)
def close_pygame():
    yield
    pygame.quit()


def test_play_relevant_keys():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    assert game.relevant_keys == {97, 100}


def test_play_relevant_keys_no_mapping():
    env = DummyPlayEnv()
    env.spec = DummyEnvSpec("DummyPlayEnv")

    with pytest.raises(MissingKeysToAction) as info:
        PlayableGame(env)


def test_play_relevant_keys_with_env_attribute():
    """Env has a keys_to_action attribute"""
    env = DummyPlayEnv()
    env.get_keys_to_action = dummy_keys_to_action
    game = PlayableGame(env)
    assert game.relevant_keys == {97, 100}


def test_video_size_no_zoom():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    assert game.video_size == list(env.render().shape)


def test_video_size_zoom():
    env = DummyPlayEnv()
    zoom = 2.2
    game = PlayableGame(env, dummy_keys_to_action(), zoom)
    assert game.video_size == tuple(int(shape * zoom) for shape in env.render().shape)


def test_keyboard_quit_event():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    event = Event(pygame.KEYDOWN, {"key": 27})
    assert game.running == True
    game.process_event(event)
    assert game.running == False


def test_pygame_quit_event():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    event = Event(pygame.QUIT)
    assert game.running == True
    game.process_event(event)
    assert game.running == False


def test_keyboard_relevant_keydown_event():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    event = Event(pygame.KEYDOWN, {"key": RELEVANT_KEY})
    game.process_event(event)
    assert game.pressed_keys == [RELEVANT_KEY]


def test_keyboard_irrelevant_keydown_event():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    event = Event(pygame.KEYDOWN, {"key": IRRELEVANT_KEY})
    game.process_event(event)
    assert game.pressed_keys == []


def test_keyboard_keyup_event():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    event = Event(pygame.KEYDOWN, {"key": RELEVANT_KEY})
    game.process_event(event)
    event = Event(pygame.KEYUP, {"key": RELEVANT_KEY})
    game.process_event(event)
    assert game.pressed_keys == []


def test_play_loop():
    env = DummyPlayEnv()
    cumulative_env_reward = 0
    for s in range(
        len(callback_events)
    ):  # we run the same number of steps executed with play()
        _, rew, _, _ = env.step(None)
        cumulative_env_reward += rew

    env_play = DummyPlayEnv()
    status = PlayStatus(callback)
    play(env_play, callback=status.callback, keys_to_action=dummy_keys_to_action())

    assert status.cumulative_reward == cumulative_env_reward
