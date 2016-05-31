import logging
from time import sleep

import numpy

import gym
from gym import utils

try:
    import doom_py
except ImportError as e:
    raise gym.error.DependencyNotInstalled("{}. (HINT: you can install Doom dependencies with 'pip install gym[doom].)'".format(e))

logger = logging.getLogger(__name__)

class DoomEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        utils.EzPickle.__init__(self)

    def _step(self, action):
        # action is a np array but DoomGame.make_action expects a list of ints
        list_action = [int(x) for x in action]
        try:
            state = self.game.get_state()
            reward = self.game.make_action(list_action)
            if self.game.is_episode_finished():
                is_finished = True
            else:
                is_finished = False
            return state.image_buffer.copy(), reward, is_finished, {}

        except doom_py.vizdoom.ViZDoomIsNotRunningException:
            return [], 0, True, {}

    def _reset(self):
        self.game.new_episode()
        return self.game.get_state().image_buffer.copy()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                # If we don't None out this reference pyglet becomes unhappy
                self.viewer = None
            return
        try:
            state = self.game.get_state()
            img = state.image_buffer
            # VizDoom returns None if the episode is finished, let's make it
            # an empty image so the recorder doesn't stop
            if img is None:
                img = numpy.zeros((self.screen_height, self.screen_width, 3), dtype=numpy.uint8)
            if mode == 'rgb_array':
                return img
            elif mode is 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
                sleep(0.02857)  # 35 fps = 0.02857 sleep between frames
        except doom_py.vizdoom.ViZDoomIsNotRunningException:
            pass # Doom has been closed

    def _close(self):
        self.game.close()
