"""Wrapper for augmenting observations by pixel values."""
import collections
import copy
from collections.abc import MutableMapping
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gym
from gym import logger, spaces

STATE_KEY = "state"


class PixelObservationWrapper(gym.ObservationWrapper):
    """Augment observations by pixel values.

    Observations of this wrapper will be dictionaries of images.
    You can also choose to add the observation of the base environment to this dictionary.
    In that case, if the base environment has an observation space of type :class:`Dict`, the dictionary
    of rendered images will be updated with the base environment's observation. If, however, the observation
    space is of type :class:`Box`, the base environment's observation (which will be an element of the :class:`Box`
    space) will be added to the dictionary under the key "state".

    Example:
        >>> import gym
        >>> env = PixelObservationWrapper(gym.make('CarRacing-v1', render_mode="single_rgb_array"))
        >>> obs = env.reset()
        >>> obs.keys()
        odict_keys(['pixels'])
        >>> obs['pixels'].shape
        (400, 600, 3)
        >>> env = PixelObservationWrapper(gym.make('CarRacing-v1', render_mode="single_rgb_array"), pixels_only=False)
        >>> obs = env.reset()
        >>> obs.keys()
        odict_keys(['state', 'pixels'])
        >>> obs['state'].shape
        (96, 96, 3)
        >>> obs['pixels'].shape
        (400, 600, 3)
        >>> env = PixelObservationWrapper(gym.make('CarRacing-v1', render_mode="single_rgb_array"), pixel_keys=('obs',))
        >>> obs = env.reset()
        >>> obs.keys()
        odict_keys(['obs'])
        >>> obs['obs'].shape
        (400, 600, 3)
    """

    def __init__(
        self,
        env: gym.Env,
        pixels_only: bool = True,
        render_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        pixel_keys: Tuple[str, ...] = ("pixels",),
    ):
        """Initializes a new pixel Wrapper.

        Args:
            env: The environment to wrap.
            pixels_only (bool): If ``True`` (default), the original observation returned
                by the wrapped environment will be discarded, and a dictionary
                observation will only include pixels. If ``False``, the
                observation dictionary will contain both the original
                observations and the pixel observations.
            render_kwargs (dict): Optional dictionary containing that maps elements of ``pixel_keys``to
                keyword arguments passed to the :meth:`self.render` method.
            pixel_keys: Optional custom string specifying the pixel
                observation's key in the ``OrderedDict`` of observations.
                Defaults to ``(pixels,)``.

        Raises:
            AssertionError: If any of the keys in ``render_kwargs``do not show up in ``pixel_keys``.
            ValueError: If ``env``'s observation space is not compatible with the
                wrapper. Supported formats are a single array, or a dict of
                arrays.
            ValueError: If ``env``'s observation already contains any of the
                specified ``pixel_keys``.
            TypeError: When an unexpected pixel type is used
        """
        super().__init__(env, new_step_api=True)

        # Avoid side-effects that occur when render_kwargs is manipulated
        render_kwargs = copy.deepcopy(render_kwargs)
        self.render_history = []

        if render_kwargs is None:
            render_kwargs = {}

        for key in render_kwargs:
            assert key in pixel_keys, (
                "The argument render_kwargs should map elements of "
                "pixel_keys to dictionaries of keyword arguments. "
                f"Found key '{key}' in render_kwargs but not in pixel_keys."
            )

        default_render_kwargs = {}
        if not env.render_mode:
            default_render_kwargs = {"mode": "rgb_array"}
            logger.warn(
                "env.render_mode must be specified to use PixelObservationWrapper:"
                "`gym.make(env_name, render_mode='single_rgb_array')`."
            )

        for key in pixel_keys:
            render_kwargs.setdefault(key, default_render_kwargs)

        wrapped_observation_space = env.observation_space

        if isinstance(wrapped_observation_space, spaces.Box):
            self._observation_is_dict = False
            invalid_keys = {STATE_KEY}
        elif isinstance(wrapped_observation_space, (spaces.Dict, MutableMapping)):
            self._observation_is_dict = True
            invalid_keys = set(wrapped_observation_space.spaces.keys())
        else:
            raise ValueError("Unsupported observation space structure.")

        if not pixels_only:
            # Make sure that now keys in the `pixel_keys` overlap with
            # `observation_keys`
            overlapping_keys = set(pixel_keys) & set(invalid_keys)
            if overlapping_keys:
                raise ValueError(
                    f"Duplicate or reserved pixel keys {overlapping_keys!r}."
                )

        if pixels_only:
            self.observation_space = spaces.Dict()
        elif self._observation_is_dict:
            self.observation_space = copy.deepcopy(wrapped_observation_space)
        else:
            self.observation_space = spaces.Dict({STATE_KEY: wrapped_observation_space})

        # Extend observation space with pixels.

        self.env.reset()
        pixels_spaces = {}
        for pixel_key in pixel_keys:
            pixels = self._render(**render_kwargs[pixel_key])
            pixels: np.ndarray = pixels[-1] if isinstance(pixels, List) else pixels

            if not hasattr(pixels, "dtype") or not hasattr(pixels, "shape"):
                raise TypeError(
                    f"Render method returns a {pixels.__class__.__name__}, but an array with dtype and shape is expected."
                    "Be sure to specify the correct render_mode."
                )

            if np.issubdtype(pixels.dtype, np.integer):
                low, high = (0, 255)
            elif np.issubdtype(pixels.dtype, np.float):
                low, high = (-float("inf"), float("inf"))
            else:
                raise TypeError(pixels.dtype)

            pixels_space = spaces.Box(
                shape=pixels.shape, low=low, high=high, dtype=pixels.dtype
            )
            pixels_spaces[pixel_key] = pixels_space

        self.observation_space.spaces.update(pixels_spaces)

        self._pixels_only = pixels_only
        self._render_kwargs = render_kwargs
        self._pixel_keys = pixel_keys

    def observation(self, observation):
        """Updates the observations with the pixel observations.

        Args:
            observation: The observation to add pixel observations for

        Returns:
            The updated pixel observations
        """
        pixel_observation = self._add_pixel_observation(observation)
        return pixel_observation

    def _add_pixel_observation(self, wrapped_observation):
        if self._pixels_only:
            observation = collections.OrderedDict()
        elif self._observation_is_dict:
            observation = type(wrapped_observation)(wrapped_observation)
        else:
            observation = collections.OrderedDict()
            observation[STATE_KEY] = wrapped_observation

        pixel_observations = {
            pixel_key: self._render(**self._render_kwargs[pixel_key])
            for pixel_key in self._pixel_keys
        }

        observation.update(pixel_observations)

        return observation

    def render(self, *args, **kwargs):
        """Renders the environment."""
        render = self.env.render(*args, **kwargs)
        if isinstance(render, list):
            render = self.render_history + render
            self.render_history = []
        return render

    def _render(self, *args, **kwargs):
        render = self.env.render(*args, **kwargs)
        if isinstance(render, list):
            self.render_history += render
        return render
