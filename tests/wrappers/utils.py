import gym


def has_wrapper(wrapped_env: gym.Env, wrapper_type: type) -> bool:
    while isinstance(wrapped_env, gym.Wrapper):
        if isinstance(wrapped_env, wrapper_type):
            return True
        wrapped_env = wrapped_env.env
    return False
