from dataclasses import dataclass
from typing import Callable

import numpy as np
import pygame
import pytest
from pygame import KEYDOWN, KEYUP, QUIT, event
from pygame.event import Event

import gym
from gym.utils.play import MissingKeysToAction, PlayableGame, play

RELEVANT_KEY_1 = ord("a")  # 97
RELEVANT_KEY_2 = ord("d")  # 100
IRRELEVANT_KEY = 1


@dataclass
class DummyEnvSpec:
    id: str


class DummyPlayEnv(gym.Env):
    def step(self, action):
        obs = np.zeros((1, 1))
        rew, done, info = 1, False, {}
        return obs, rew, done, info

    def reset(self, seed=None):
        ...

    def render(self, mode="rgb_array"):
        return np.zeros((1, 1))


class PlayStatus:
    def __init__(self, callback: Callable):
        self.data_callback = callback
        self.cumulative_reward = 0
        self.last_observation = None

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        _, obs_tp1, _, rew, _, _ = self.data_callback(
            obs_t, obs_tp1, action, rew, done, info
        )
        self.cumulative_reward += rew
        self.last_observation = obs_tp1


def dummy_keys_to_action():
    return {(RELEVANT_KEY_1,): 0, (RELEVANT_KEY_2,): 1}


@pytest.fixture(autouse=True)
def close_pygame():
    yield
    pygame.quit()


def test_play_relevant_keys():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    assert game.relevant_keys == {RELEVANT_KEY_1, RELEVANT_KEY_2}


def test_play_relevant_keys_no_mapping():
    env = DummyPlayEnv()
    env.spec = DummyEnvSpec("DummyPlayEnv")

    with pytest.raises(MissingKeysToAction):
        PlayableGame(env)


def test_play_relevant_keys_with_env_attribute():
    """Env has a keys_to_action attribute"""
    env = DummyPlayEnv()
    env.get_keys_to_action = dummy_keys_to_action
    game = PlayableGame(env)
    assert game.relevant_keys == {RELEVANT_KEY_1, RELEVANT_KEY_2}


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
    event = Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE})
    assert game.running is True
    game.process_event(event)
    assert game.running is False


def test_pygame_quit_event():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    event = Event(pygame.QUIT)
    assert game.running is True
    game.process_event(event)
    assert game.running is False


def test_keyboard_relevant_keydown_event():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    event = Event(pygame.KEYDOWN, {"key": RELEVANT_KEY_1})
    game.process_event(event)
    assert game.pressed_keys == [RELEVANT_KEY_1]


def test_keyboard_irrelevant_keydown_event():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    event = Event(pygame.KEYDOWN, {"key": IRRELEVANT_KEY})
    game.process_event(event)
    assert game.pressed_keys == []


def test_keyboard_keyup_event():
    env = DummyPlayEnv()
    game = PlayableGame(env, dummy_keys_to_action())
    event = Event(pygame.KEYDOWN, {"key": RELEVANT_KEY_1})
    game.process_event(event)
    event = Event(pygame.KEYUP, {"key": RELEVANT_KEY_1})
    game.process_event(event)
    assert game.pressed_keys == []


def test_play_loop_real_env():
    SEED = 42
    ENV = "CartPole-v1"

    # set of key events to inject into the play loop as callback
    callback_events = [
        Event(KEYDOWN, {"key": RELEVANT_KEY_1}),
        Event(KEYUP, {"key": RELEVANT_KEY_1}),
        Event(KEYDOWN, {"key": RELEVANT_KEY_2}),
        Event(KEYUP, {"key": RELEVANT_KEY_2}),
        Event(KEYDOWN, {"key": RELEVANT_KEY_1}),
        Event(KEYUP, {"key": RELEVANT_KEY_1}),
        Event(KEYDOWN, {"key": RELEVANT_KEY_1}),
        Event(KEYUP, {"key": RELEVANT_KEY_1}),
        Event(KEYDOWN, {"key": RELEVANT_KEY_2}),
        Event(KEYUP, {"key": RELEVANT_KEY_2}),
        Event(QUIT),
    ]
    keydown_events = [k for k in callback_events if k.type == KEYDOWN]

    def callback(obs_t, obs_tp1, action, rew, done, info):
        pygame_event = callback_events.pop(0)
        event.post(pygame_event)

        # after releasing a key, post new events until
        # we have one keydown
        while pygame_event.type == KEYUP:
            pygame_event = callback_events.pop(0)
            event.post(pygame_event)

        return obs_t, obs_tp1, action, rew, done, info

    env = gym.make(ENV)
    env.reset(seed=SEED)
    keys_to_action = dummy_keys_to_action()

    # first action is 0 because at the first iteration
    # we can not inject a callback event into play()
    env.step(0)
    for e in keydown_events:
        action = keys_to_action[(e.key,)]
        obs, _, _, _ = env.step(action)

    env_play = gym.make(ENV)
    status = PlayStatus(callback)
    play(env_play, callback=status.callback, keys_to_action=keys_to_action, seed=SEED)

    assert (status.last_observation == obs).all()
