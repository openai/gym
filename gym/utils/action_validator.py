import warnings

import numpy as np

from gym.error import InvalidAction
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete


def validate_action_discrete(func):
    def wrapper(self, action, *args, **kwargs):
        if isinstance(self.action_space, Discrete) and not self.action_space.contains(
            action
        ):
            raise InvalidAction(
                f"you passed the action `{action}` with dtype "
                f"{type(action)} while the supported action space is "
                f"{self.action_space} with dtype {self.action_space.dtype}"
            )
        return func(self, action, *args, **kwargs)

    return wrapper


def validate_action_continuous(func):
    """Continuous action space do not raise any exception if an out of bound
    action is performed. Typically clipping occurs inside the step call.
    Also wrong shape input are allowed and managed internally.
    To maintain backward compatibility we are only raising a warning
    if a potentially invalid action is performed.
    """

    def wrapper(self, action, *args, **kwargs):
        if isinstance(action, np.ndarray):
            action_type = action.dtype
        else:
            action_type = type(action)

        if isinstance(self.action_space, Box) and not self.action_space.contains(
            action
        ):
            warnings.warn(
                f"you passed the action `{action}` with dtype "
                f"{action_type} while the supported action space is "
                f"{self.action_space} with dtype {self.action_space.dtype}"
            )
        return func(self, action, *args, **kwargs)

    return wrapper
