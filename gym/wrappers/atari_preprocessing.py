"""Implementation of Atari 2600 Preprocessing following the guidelines of Machado et al., 2018."""
import numpy as np

import gym
from gym.spaces import Box
from gym.utils.step_api_compatibility import step_api_compatibility

try:
    import cv2
except ImportError:
    cv2 = None


class AtariPreprocessing(gym.Wrapper):
    """Atari 2600 preprocessing wrapper.

    This class follows the guidelines in Machado et al. (2018),
    "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents".

    Specifically, the following preprocess stages applies to the atari environment:
    - Noop Reset: Obtains the initial state by taking a random number of no-ops on reset, default max 30 no-ops.
    - Frame skipping: The number of frames skipped between steps, 4 by default
    - Max-pooling: Pools over the most recent two observations from the frame skips
    - Termination signal when a life is lost: When the agent losses a life during the environment, then the environment is terminated.
        Turned off by default. Not recommended by Machado et al. (2018).
    - Resize to a square image: Resizes the atari environment original observation shape from 210x180 to 84x84 by default
    - Grayscale observation: If the observation is colour or greyscale, by default, greyscale.
    - Scale observation: If to scale the observation between [0, 1) or [0, 255), by default, not scaled.
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = False,
        grayscale_obs: bool = True,
        grayscale_newaxis: bool = False,
        scale_obs: bool = False,
        new_step_api: bool = False,
    ):
        """Wrapper for Atari 2600 preprocessing.

        Args:
            env (Env): The environment to apply the preprocessing
            noop_max (int): For No-op reset, the max number no-ops actions are taken at reset, to turn off, set to 0.
            frame_skip (int): The number of frames between new observation the agents observations effecting the frequency at which the agent experiences the game.
            screen_size (int): resize Atari frame
            terminal_on_life_loss (bool): `if True`, then :meth:`step()` returns `terminated=True` whenever a
                life is lost.
            grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
                is returned.
            grayscale_newaxis (bool): `if True and grayscale_obs=True`, then a channel axis is added to
                grayscale observations to make them 3-dimensional.
            scale_obs (bool): if True, then observation normalized in range [0,1) is returned. It also limits memory
                optimization benefits of FrameStack Wrapper.

        Raises:
            DependencyNotInstalled: opencv-python package not installed
            ValueError: Disable frame-skipping in the original env
        """
        super().__init__(env, new_step_api)
        if cv2 is None:
            raise gym.error.DependencyNotInstalled(
                "opencv-python package not installed, run `pip install gym[other]` to get dependencies for atari"
            )
        assert frame_skip > 0
        assert screen_size > 0
        assert noop_max >= 0
        if frame_skip > 1:
            if (
                "NoFrameskip" not in env.spec.id
                and getattr(env.unwrapped, "_frameskip", None) != 1
            ):
                raise ValueError(
                    "Disable frame-skipping in the original env. Otherwise, more than one "
                    "frame-skip will happen as through this wrapper"
                )
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale_obs = grayscale_obs
        self.grayscale_newaxis = grayscale_newaxis
        self.scale_obs = scale_obs

        # buffer of most recent two observations for max pooling
        assert isinstance(env.observation_space, Box)
        if grayscale_obs:
            self.obs_buffer = [
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
            ]
        else:
            self.obs_buffer = [
                np.empty(env.observation_space.shape, dtype=np.uint8),
                np.empty(env.observation_space.shape, dtype=np.uint8),
            ]

        self.ale = env.unwrapped.ale
        self.lives = 0
        self.game_over = False

        _low, _high, _obs_dtype = (
            (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
        )
        _shape = (screen_size, screen_size, 1 if grayscale_obs else 3)
        if grayscale_obs and not grayscale_newaxis:
            _shape = _shape[:-1]  # Remove channel axis
        self.observation_space = Box(
            low=_low, high=_high, shape=_shape, dtype=_obs_dtype
        )

    def step(self, action):
        """Applies the preprocessing for an :meth:`env.step`."""
        total_reward, terminated, truncated, info = 0.0, False, False, {}

        for t in range(self.frame_skip):
            _, reward, terminated, truncated, info = step_api_compatibility(
                self.env.step(action), True
            )
            total_reward += reward
            self.game_over = terminated

            if self.terminal_on_life_loss:
                new_lives = self.ale.lives()
                terminated = terminated or new_lives < self.lives
                self.game_over = terminated
                self.lives = new_lives

            if terminated or truncated:
                break
            if t == self.frame_skip - 2:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[1])
                else:
                    self.ale.getScreenRGB(self.obs_buffer[1])
            elif t == self.frame_skip - 1:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[0])
                else:
                    self.ale.getScreenRGB(self.obs_buffer[0])
        return step_api_compatibility(
            (self._get_obs(), total_reward, terminated, truncated, info),
            self.new_step_api,
        )

    def reset(self, **kwargs):
        """Resets the environment using preprocessing."""
        # NoopReset
        if kwargs.get("return_info", False):
            _, reset_info = self.env.reset(**kwargs)
        else:
            _ = self.env.reset(**kwargs)
            reset_info = {}

        noops = (
            self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
            if self.noop_max > 0
            else 0
        )
        for _ in range(noops):
            _, _, terminated, truncated, step_info = step_api_compatibility(
                self.env.step(0), True
            )
            reset_info.update(step_info)
            if terminated or truncated:
                if kwargs.get("return_info", False):
                    _, reset_info = self.env.reset(**kwargs)
                else:
                    _ = self.env.reset(**kwargs)
                    reset_info = {}

        self.lives = self.ale.lives()
        if self.grayscale_obs:
            self.ale.getScreenGrayscale(self.obs_buffer[0])
        else:
            self.ale.getScreenRGB(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)

        if kwargs.get("return_info", False):
            return self._get_obs(), reset_info
        else:
            return self._get_obs()

    def _get_obs(self):
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        assert cv2 is not None
        obs = cv2.resize(
            self.obs_buffer[0],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )

        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)

        if self.grayscale_obs and self.grayscale_newaxis:
            obs = np.expand_dims(obs, axis=-1)  # Add a channel axis
        return obs
