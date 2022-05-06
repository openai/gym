from typing import Union

import gym


def has_wrapper(wrapped_env: Union[gym.Wrapper, gym.Env], wrapper_type: type):
    while type(wrapped_env) != gym.Env:
        if isinstance(wrapped_env, wrapper_type):
            return True
        wrapped_env = wrapped_env.env
    return False
