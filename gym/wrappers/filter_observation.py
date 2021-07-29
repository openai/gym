import copy
from gym import spaces
from gym import ObservationWrapper


class FilterObservation(ObservationWrapper):
    """Filter dictionary observations by their keys.

    Args:
        env: The environment to wrap.
        filter_keys: List of keys to be included in the observations.

    Raises:
        ValueError: If observation keys in not instance of None or
            iterable.
        ValueError: If any of the `filter_keys` are not included in
            the original `env`'s observation space

    """

    def __init__(self, env, filter_keys=None):
        super(FilterObservation, self).__init__(env)

        wrapped_observation_space = env.observation_space
        assert isinstance(
            wrapped_observation_space, spaces.Dict
        ), "FilterObservationWrapper is only usable with dict observations."

        observation_keys = wrapped_observation_space.spaces.keys()

        if filter_keys is None:
            filter_keys = tuple(observation_keys)

        missing_keys = set(key for key in filter_keys if key not in observation_keys)

        if missing_keys:
            raise ValueError(
                "All the filter_keys must be included in the "
                "original obsrevation space.\n"
                "Filter keys: {filter_keys}\n"
                "Observation keys: {observation_keys}\n"
                "Missing keys: {missing_keys}".format(
                    filter_keys=filter_keys,
                    observation_keys=observation_keys,
                    missing_keys=missing_keys,
                )
            )

        self.observation_space = type(wrapped_observation_space)(
            [
                (name, copy.deepcopy(space))
                for name, space in wrapped_observation_space.spaces.items()
                if name in filter_keys
            ]
        )

        self._env = env
        self._filter_keys = tuple(filter_keys)

    def observation(self, observation):
        filter_observation = self._filter_observation(observation)
        return filter_observation

    def _filter_observation(self, observation):
        observation = type(observation)(
            [
                (name, value)
                for name, value in observation.items()
                if name in self._filter_keys
            ]
        )
        return observation
