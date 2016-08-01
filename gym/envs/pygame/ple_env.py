import gym
from gym import spaces
from ple import PLE

class PLEEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name='FlappyBird', obs_type='ram', display_screen=True):
        # open up a game state to communicate with emulator
        import importlib
        game_module_name = ('ple.games.%s' % game_name).lower()
        game_module = importlib.import_module(game_module_name)
        game = getattr(game_module, game_name)()
        self.game_state = PLE(game, fps=30, display_screen=display_screen)
        self.game_state.init()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.screen_width, self.screen_height = self.game_state.getScreenDims()
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.viewer = None


    def _step(self, a):
        reward = self.game_state.act(self._action_set[a])
        state = self.game_state.getScreenRGB()
        terminal = self.game_state.game_over()
        return state, reward, terminal, {}

    def _get_image(self):
        return self.game_state.getScreenRGB()

    @property
    def _n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def _reset(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.game_state.reset_game()
        state = self.game_state.getScreenRGB()
        return state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

