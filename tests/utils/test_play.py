from functools import partial
from itertools import product
from typing import Callable

import numpy as np
import pygame
import pytest
from pygame import KEYDOWN, KEYUP, QUIT, event
from pygame.event import Event

import gym
from gym.utils.play import MissingKeysToAction, PlayableGame, play
from tests.testing_env import GenericTestEnv

RELEVANT_KEY_1 = ord("a")  # 97
RELEVANT_KEY_2 = ord("d")  # 100
IRRELEVANT_KEY = 1


PlayableEnv = partial(
    GenericTestEnv,
    metadata={"render_modes": ["rgb_array"]},
    render_fn=lambda self: np.ones((10, 10, 3)),
)


class KeysToActionWrapper(gym.Wrapper):
    def __init__(self, env, keys_to_action):
        super().__init__(env)
        self.keys_to_action = keys_to_action

    def get_keys_to_action(self):
        return self.keys_to_action


class PlayStatus:
    def __init__(self, callback: Callable):
        self.data_callback = callback
        self.cumulative_reward = 0
        self.last_observation = None

    def callback(self, obs_t, obs_tp1, action, rew, terminated, truncated, info):
        _, obs_tp1, _, rew, _, _, _ = self.data_callback(
            obs_t, obs_tp1, action, rew, terminated, truncated, info
        )
        self.cumulative_reward += rew
        self.last_observation = obs_tp1


def dummy_keys_to_action():
    return {(RELEVANT_KEY_1,): 0, (RELEVANT_KEY_2,): 1}


def dummy_keys_to_action_str():
    """{'a': 0, 'd': 1}"""
    return {chr(RELEVANT_KEY_1): 0, chr(RELEVANT_KEY_2): 1}


@pytest.fixture(autouse=True)
def close_pygame():
    yield
    pygame.quit()


def test_play_relevant_keys():
    env = PlayableEnv(render_mode="rgb_array")
    game = PlayableGame(env, dummy_keys_to_action())
    assert game.relevant_keys == {RELEVANT_KEY_1, RELEVANT_KEY_2}


def test_play_relevant_keys_no_mapping():
    env = PlayableEnv(render_mode="rgb_array")

    with pytest.raises(MissingKeysToAction):
        PlayableGame(env)


def test_play_relevant_keys_with_env_attribute():
    """Env has a keys_to_action attribute"""
    env = PlayableEnv(render_mode="rgb_array")
    env.get_keys_to_action = dummy_keys_to_action
    game = PlayableGame(env)
    assert game.relevant_keys == {RELEVANT_KEY_1, RELEVANT_KEY_2}


def test_video_size_no_zoom():
    env = PlayableEnv(render_mode="rgb_array")
    game = PlayableGame(env, dummy_keys_to_action())
    assert game.video_size == env.render().shape[:2]


def test_video_size_zoom():
    env = PlayableEnv(render_mode="rgb_array")
    zoom = 2.2
    game = PlayableGame(env, dummy_keys_to_action(), zoom)
    assert game.video_size == tuple(int(dim * zoom) for dim in env.render().shape[:2])


def test_keyboard_quit_event():
    env = PlayableEnv(render_mode="rgb_array")
    game = PlayableGame(env, dummy_keys_to_action())
    event = Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE})
    assert game.running is True
    game.process_event(event)
    assert game.running is False


def test_pygame_quit_event():
    env = PlayableEnv(render_mode="rgb_array")
    game = PlayableGame(env, dummy_keys_to_action())
    event = Event(pygame.QUIT)
    assert game.running is True
    game.process_event(event)
    assert game.running is False


def test_keyboard_relevant_keydown_event():
    env = PlayableEnv(render_mode="rgb_array")
    game = PlayableGame(env, dummy_keys_to_action())
    event = Event(pygame.KEYDOWN, {"key": RELEVANT_KEY_1})
    game.process_event(event)
    assert game.pressed_keys == [RELEVANT_KEY_1]


def test_keyboard_irrelevant_keydown_event():
    env = PlayableEnv(render_mode="rgb_array")
    game = PlayableGame(env, dummy_keys_to_action())
    event = Event(pygame.KEYDOWN, {"key": IRRELEVANT_KEY})
    game.process_event(event)
    assert game.pressed_keys == []


def test_keyboard_keyup_event():
    env = PlayableEnv(render_mode="rgb_array")
    game = PlayableGame(env, dummy_keys_to_action())
    event = Event(pygame.KEYDOWN, {"key": RELEVANT_KEY_1})
    game.process_event(event)
    event = Event(pygame.KEYUP, {"key": RELEVANT_KEY_1})
    game.process_event(event)
    assert game.pressed_keys == []


def test_play_loop_real_env():
    SEED = 42
    ENV = "CartPole-v1"

    # If apply_wrapper is true, we provide keys_to_action through the environment. If str_keys is true, the
    # keys_to_action dictionary will have strings as keys
    for apply_wrapper, str_keys in product([False, True], [False, True]):
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

        def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
            pygame_event = callback_events.pop(0)
            event.post(pygame_event)

            # after releasing a key, post new events until
            # we have one keydown
            while pygame_event.type == KEYUP:
                pygame_event = callback_events.pop(0)
                event.post(pygame_event)

            return obs_t, obs_tp1, action, rew, terminated, truncated, info

        env = gym.make(ENV, render_mode="rgb_array", disable_env_checker=True)
        env.reset(seed=SEED)
        keys_to_action = (
            dummy_keys_to_action_str() if str_keys else dummy_keys_to_action()
        )

        # first action is 0 because at the first iteration
        # we can not inject a callback event into play()
        obs, _, _, _, _ = env.step(0)
        for e in keydown_events:
            action = keys_to_action[chr(e.key) if str_keys else (e.key,)]
            obs, _, _, _, _ = env.step(action)

        env_play = gym.make(ENV, render_mode="rgb_array", disable_env_checker=True)
        if apply_wrapper:
            env_play = KeysToActionWrapper(env, keys_to_action=keys_to_action)
            assert hasattr(env_play, "get_keys_to_action")

        status = PlayStatus(callback)
        play(
            env_play,
            callback=status.callback,
            keys_to_action=None if apply_wrapper else keys_to_action,
            seed=SEED,
        )

        assert (status.last_observation == obs).all()


def test_play_no_keys():
    with pytest.raises(MissingKeysToAction):
        play(gym.make("CartPole-v1"))
