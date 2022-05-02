from typing import Callable, Dict, Optional, Tuple

import pygame
from numpy.typing import NDArray
from pygame import Surface
from pygame.event import Event

from gym import Env, logger
from gym.logger import deprecation

try:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warn(f"failed to set matplotlib backend, plotting will not work: {str(e)}")
    plt = None

from collections import deque

from pygame.locals import VIDEORESIZE


class MissingKeysToAction(Exception):
    """Raised when the environment does not have
    a default keys_to_action mapping
    """


class PlayableGame:
    def __init__(
        self,
        env: Env,
        keys_to_action: Optional[Dict[Tuple[int], int]] = None,
        zoom: Optional[float] = None,
    ):
        self.env = env
        self.relevant_keys = self._get_relevant_keys(keys_to_action)
        self.video_size = self._get_video_size(zoom)
        self.screen = pygame.display.set_mode(self.video_size)
        self.pressed_keys = []
        self.running = True

    def _get_relevant_keys(
        self, keys_to_action: Optional[Dict[Tuple[int], int]] = None
    ) -> set:
        if keys_to_action is None:
            if hasattr(self.env, "get_keys_to_action"):
                keys_to_action = self.env.get_keys_to_action()
            elif hasattr(self.env.unwrapped, "get_keys_to_action"):
                keys_to_action = self.env.unwrapped.get_keys_to_action()
            else:
                raise MissingKeysToAction(
                    "%s does not have explicit key to action mapping, "
                    "please specify one manually" % self.env.spec.id
                )
        relevant_keys = set(sum((list(k) for k in keys_to_action.keys()), []))
        return relevant_keys

    def _get_video_size(self, zoom: Optional[float] = None) -> Tuple[int, int]:
        # TODO: this needs to be updated when the render API change goes through
        rendered = self.env.render(mode="rgb_array")
        video_size = [rendered.shape[1], rendered.shape[0]]

        if zoom is not None:
            video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

        return video_size

    def process_event(self, event: Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in self.relevant_keys:
                self.pressed_keys.append(event.key)
            elif event.key == pygame.K_ESCAPE:
                self.running = False
        elif event.type == pygame.KEYUP:
            if event.key in self.relevant_keys:
                self.pressed_keys.remove(event.key)
        elif event.type == pygame.QUIT:
            self.running = False
        elif event.type == VIDEORESIZE:
            self.video_size = event.size
            self.screen = pygame.display.set_mode(self.video_size)


def display_arr(
    screen: Surface, arr: NDArray, video_size: Tuple[int, int], transpose: bool
):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))


def play(
    env: Env,
    transpose: Optional[bool] = True,
    fps: Optional[int] = 30,
    zoom: Optional[float] = None,
    callback: Optional[Callable] = None,
    keys_to_action: Optional[Dict[Tuple[int], int]] = None,
    seed: Optional[int] = None,
):
    """Allows one to play the game using keyboard.

    To simply play the game use:

        play(gym.make("Pong-v4"))

    Above code works also if env is wrapped, so it's particularly useful in
    verifying that the frame-level preprocessing does not render the game
    unplayable.

    If you wish to plot real time statistics as you play, you can use
    gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
    for last 5 second of gameplay.

        def callback(obs_t, obs_tp1, action, rew, done, info):
            return [rew,]
        plotter = PlayPlot(callback, 30 * 5, ["reward"])

        env = gym.make("Pong-v4")
        play(env, callback=plotter.callback)


    Arguments
    ---------
    env: gym.Env
        Environment to use for playing.
    transpose: bool
        If True the output of observation is transposed.
        Defaults to true.
    fps: int
        Maximum number of steps of the environment to execute every second.
        Defaults to 30.
    zoom: float
        Make screen edge this many times bigger
    callback: lambda or None
        Callback if a callback is provided it will be executed after
        every step. It takes the following input:
            obs_t: observation before performing action
            obs_tp1: observation after performing action
            action: action that was executed
            rew: reward that was received
            done: whether the environment is done or not
            info: debug info
    keys_to_action: dict: tuple(int) -> int or None
        Mapping from keys pressed to action performed.
        For example if pressed 'w' and space at the same time is supposed
        to trigger action number 2 then key_to_action dict would look like this:

            {
                # ...
                sorted(ord('w'), ord(' ')) -> 2
                # ...
            }
        If None, default key_to_action mapping for that env is used, if provided.
    seed: bool or None
        Random seed used when resetting the environment. If None, no seed is used.
    """
    env.reset(seed=seed)
    game = PlayableGame(env, keys_to_action, zoom)

    done = True
    clock = pygame.time.Clock()

    while game.running:
        if done:
            done = False
            obs = env.reset(seed=seed)
        else:
            action = keys_to_action.get(tuple(sorted(game.pressed_keys)), 0)
            prev_obs = obs
            obs, rew, done, info = env.step(action)
            if callback is not None:
                callback(prev_obs, obs, action, rew, done, info)
        if obs is not None:
            # TODO: this needs to be updated when the render API change goes through
            rendered = env.render(mode="rgb_array")
            display_arr(
                game.screen, rendered, transpose=transpose, video_size=game.video_size
            )

        # process pygame events
        for event in pygame.event.get():
            game.process_event(event)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


class PlayPlot:
    def __init__(self, callback, horizon_timesteps, plot_names):
        deprecation(
            "`PlayPlot` is marked as deprecated and will be removed in the near future."
        )
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names

        assert plt is not None, "matplotlib backend failed, plotting will not work"

        num_plots = len(self.plot_names)
        self.fig, self.ax = plt.subplots(num_plots)
        if num_plots == 1:
            self.ax = [self.ax]
        for axis, name in zip(self.ax, plot_names):
            axis.set_title(name)
        self.t = 0
        self.cur_plot = [None for _ in range(num_plots)]
        self.data = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        points = self.data_callback(obs_t, obs_tp1, action, rew, done, info)
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1

        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                plot.remove()
            self.cur_plot[i] = self.ax[i].scatter(
                range(xmin, xmax), list(self.data[i]), c="blue"
            )
            self.ax[i].set_xlim(xmin, xmax)
        plt.pause(0.000001)
