"""Module for vector environments."""
from typing import Iterable, List, Optional, Union

import gym
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.vector_env import VectorEnv, VectorEnvWrapper

__all__ = ["AsyncVectorEnv", "SyncVectorEnv", "VectorEnv", "VectorEnvWrapper", "make"]


def make(
    id: str,
    num_envs: int = 1,
    asynchronous: bool = True,
    wrappers: Optional[Union[callable, List[callable]]] = None,
    disable_env_checker: bool = False,
    **kwargs,
) -> VectorEnv:
    """Create a vectorized environment from multiple copies of an environment, from its id.

    Example::

        >>> import gym
        >>> env = gym.vector.make('CartPole-v1', num_envs=3)
        >>> env.reset()
        array([[-0.04456399,  0.04653909,  0.01326909, -0.02099827],
               [ 0.03073904,  0.00145001, -0.03088818, -0.03131252],
               [ 0.03468829,  0.01500225,  0.01230312,  0.01825218]],
              dtype=float32)

    Args:
        id: The environment ID. This must be a valid ID from the registry.
        num_envs: Number of copies of the environment.
        asynchronous: If `True`, wraps the environments in an :class:`AsyncVectorEnv` (which uses `multiprocessing`_ to run the environments in parallel). If ``False``, wraps the environments in a :class:`SyncVectorEnv`.
        wrappers: If not ``None``, then apply the wrappers to each internal environment during creation.
        disable_env_checker: If to disable the env checker, if True it will only run on the first environment created.
        **kwargs: Keywords arguments applied during gym.make

    Returns:
        The vectorized environment.
    """

    def create_env(_disable_env_checker):
        """Creates an environment that can enable or disable the environment checker."""

        def _make_env():
            env = gym.envs.registration.make(
                id, disable_env_checker=_disable_env_checker, **kwargs
            )
            if wrappers is not None:
                if callable(wrappers):
                    env = wrappers(env)
                elif isinstance(wrappers, Iterable) and all(
                    [callable(w) for w in wrappers]
                ):
                    for wrapper in wrappers:
                        env = wrapper(env)
                else:
                    raise NotImplementedError
            return env

        return _make_env

    env_fns = [
        create_env(env_num == 0 and disable_env_checker is False)
        for env_num in range(num_envs)
    ]
    return AsyncVectorEnv(env_fns) if asynchronous else SyncVectorEnv(env_fns)
