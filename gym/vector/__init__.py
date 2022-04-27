try:
    from collections.abc import Iterable
except ImportError:
    Iterable = (tuple, list)

from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.vector_env import VectorEnv, VectorEnvWrapper

__all__ = ["AsyncVectorEnv", "SyncVectorEnv", "VectorEnv", "VectorEnvWrapper", "make"]


def make(
    id, num_envs=1, asynchronous=True, wrappers=None, info_format="classic", **kwargs
):
    """Create a vectorized environment from multiple copies of an environment,
    from its id.

    Parameters
    ----------
    id : str
        The environment ID. This must be a valid ID from the registry.

    num_envs : int
        Number of copies of the environment.

    asynchronous : bool
        If `True`, wraps the environments in an :class:`AsyncVectorEnv` (which uses
        `multiprocessing`_ to run the environments in parallel). If ``False``,
        wraps the environments in a :class:`SyncVectorEnv`.

    wrappers : callable, or iterable of callables, optional
        If not ``None``, then apply the wrappers to each internal
        environment during creation.

    info_format : str, optional
        Choose one of the available info formatting strategies. Default behaviour
        is returning a list of dictionaries where each dictionary represents the
        info of the environment at index i.

    Returns
    -------
    :class:`gym.vector.VectorEnv`
        The vectorized environment.

    Example
    -------
    >>> env = gym.vector.make('CartPole-v1', num_envs=3)
    >>> env.reset()
    array([[-0.04456399,  0.04653909,  0.01326909, -0.02099827],
           [ 0.03073904,  0.00145001, -0.03088818, -0.03131252],
           [ 0.03468829,  0.01500225,  0.01230312,  0.01825218]],
          dtype=float32)
    """
    from gym.envs import make as make_

    def _make_env():
        env = make_(id, **kwargs)
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

    env_fns = [_make_env for _ in range(num_envs)]
    return (
        AsyncVectorEnv(env_fns, info_format=info_format)
        if asynchronous
        else SyncVectorEnv(env_fns, info_format=info_format)
    )
