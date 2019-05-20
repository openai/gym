"""An observation wrapper that augments observations by pixel values."""

import collections
import copy

import numpy as np

from gym import spaces
from gym import ObservationWrapper

STATE_KEY = 'state'


class PixelObservationWrapper(ObservationWrapper):
    """Augment observations by pixel values."""
    def __init__(self,
                 env,
                 pixels_only=True,
                 render_kwargs=None,
                 observation_key='pixels'):
        """Initializes a new pixel Wrapper.

        Args:
            env: The environment to wrap.
            pixels_only: If `True` (default), the original observation returned
                by the wrapped environment will be discarded, and a dictionary
                observation will only include pixels. If `False`, the
                observation dictionary will contain both the original
                observations and the pixel observations.
            render_kwargs: Optional `dict` containing keyword arguments passed
                to the `self.render` method.
            observation_key: Optional custom string specifying the pixel
                observation's key in the `OrderedDict` of observations.
                Defaults to 'pixels'.

        Raises:
            ValueError: If `env`'s observation spec is not compatible with the
                wrapper. Supported formats are a single array, or a dict of
                arrays.
            ValueError: If `env`'s observation already contains the specified
                `observation_key`.
        """

        super(PixelObservationWrapper, self).__init__(env)
        if render_kwargs is None:
            render_kwargs = {}

        render_mode = render_kwargs.pop('mode', 'rgb_array')
        assert render_mode == 'rgb_array', render_mode
        render_kwargs['mode'] = 'rgb_array'

        wrapped_observation_space = env.observation_space

        if isinstance(wrapped_observation_space, spaces.Box):
            self._observation_is_dict = False
            invalid_keys = set([STATE_KEY])
        elif isinstance(wrapped_observation_space,
                        (spaces.Dict, collections.MutableMapping)):
            self._observation_is_dict = True
            invalid_keys = set(wrapped_observation_space.spaces.keys())
        else:
            raise ValueError("Unsupported observation space structure.")

        if not pixels_only and observation_key in invalid_keys:
            raise ValueError("Duplicate or reserved observation key {!r}."
                             .format(observation_key))

        if pixels_only:
            self.observation_space = spaces.Dict()
        elif self._observation_is_dict:
            self.observation_space = copy.deepcopy(wrapped_observation_space)
        else:
            self.observation_space = spaces.Dict()
            self.observation_space.spaces[STATE_KEY] = wrapped_observation_space

        # Extend observation space with pixels.
        pixels = self.env.render(**render_kwargs)

        if np.issubdtype(pixels.dtype, np.integer):
            low, high = (0, 255)
        elif np.issubdtype(pixels.dtype, np.float):
            low, high = (-float('inf'), float('inf'))
        else:
            raise TypeError(pixels.dtype)

        pixels_space = spaces.Box(
            shape=pixels.shape, low=low, high=high, dtype=pixels.dtype)
        self.observation_space.spaces[observation_key] = pixels_space

        self._env = env
        self._pixels_only = pixels_only
        self._render_kwargs = render_kwargs
        self._observation_key = observation_key

    def observation(self, observation):
        pixel_observation = self._add_pixel_observation(observation)
        return pixel_observation

    def _add_pixel_observation(self, observation):
        if self._pixels_only:
            observation = collections.OrderedDict()
        elif self._observation_is_dict:
            observation = type(observation)(observation)
        else:
            observation = collections.OrderedDict()
            observation[STATE_KEY] = observation

        pixels = self.env.render(**self._render_kwargs)
        observation[self._observation_key] = pixels

        return observation
