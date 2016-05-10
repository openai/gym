import logging
import gym
from gym import error
from time import sleep

try:
    import doom_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install Doom dependencies with 'pip install gym[doom].)'".format(e))

logger = logging.getLogger(__name__)

class DoomEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def _step(self, action):
        action = action
        list_actions = []
        for i in self.allowed_actions:
            list_actions.append(int(action[i]))
        try:
            state = self.game.get_state()
            reward = self.game.make_action(list_actions)
            if self.sleep_time > 0: sleep(self.sleep_time)
            if self.game.is_episode_finished():
                is_finished = True
                self.game.close()
            else:
                is_finished = False
            return state.image_buffer.copy(), reward, is_finished, state.game_variables

        except doom_py.vizdoom.doom_is_not_running_exception:
            return [], 0, True, {}

    def _reset(self):
        self.game.new_episode()
        return self.game.get_state().image_buffer.copy()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return
        try:
            state = self.game.get_state()
            img = state.image_buffer
            if mode == 'rgb_array':
                return img
            elif mode is 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
        except doom_py.vizdoom.doom_is_not_running_exception:
            pass # Doom has been closed
