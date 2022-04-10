from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pygame
import pytest

import gym
from gym.utils.play import PlayableGame, play

RELEVANT_KEY = 100
IRRELEVANT_KEY = 1


@dataclass
class MockKeyEvent:
    type: pygame.event.Event
    key: Optional[int] = field(default=None)
    size: Optional[Tuple[int, int]] = field(default=None)


@dataclass
class DummyEnvSpec:
    id: str


class DummyPlayEnv(gym.Env):
    def step(self, action):
        ...

    def reset(self):
        ...

    def render(self, mode="rgb_array"):
        return np.zeros((1, 1))


def dummy_keys_to_action():
    return {(ord("a"),): 0, (ord("d"),): 1}


def test_play_relvant_keys():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    assert game.relevant_keys == {97, 100}


def test_play_revant_keys_no_mapping():
    env = DummyPlayEnv()
    env.spec = DummyEnvSpec("DummyPlayEnv")

    with pytest.raises(AssertionError) as info:
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
    video_size = game.get_video_size()
    assert video_size == list(env.render().shape)


def test_video_size_zoom():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    zoom_value = 2.2
    video_size = game.get_video_size(zoom=zoom_value)
    assert video_size == tuple(int(shape * zoom_value) for shape in env.render().shape)


def test_keyboard_quit_event():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    event = MockKeyEvent(pygame.KEYDOWN, 27)
    assert game.running == True
    game.process_event(event)
    assert game.running == False


def test_pygame_quit_event():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    event = MockKeyEvent(pygame.QUIT)
    assert game.running == True
    game.process_event(event)
    assert game.running == False


def test_keyboard_relevant_keydown_event():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    event = MockKeyEvent(pygame.KEYDOWN, RELEVANT_KEY)
    game.process_event(event)
    assert game.pressed_keys == [RELEVANT_KEY]


def test_keyboard_irrelevant_keydown_event():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    event = MockKeyEvent(pygame.KEYDOWN, IRRELEVANT_KEY)
    game.process_event(event)
    assert game.pressed_keys == []


def test_keyboard_keyup_event():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    event = MockKeyEvent(pygame.KEYDOWN, RELEVANT_KEY)
    game.process_event(event)
    event = MockKeyEvent(pygame.KEYUP, RELEVANT_KEY)
    game.process_event(event)
    assert game.pressed_keys == []


# def test_play_loop():
#     env = DummyPlayEnv()
#     keys_to_action = {
#         (ord('a'),): 0,
#         (ord('d'),): 1
#     }
#     play(env, keys_to_action=keys_to_action)
