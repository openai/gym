from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.vector_env import VectorEnv

__all__ = ['AsyncVectorEnv', 'SyncVectorEnv', 'VectorEnv', 'make']

def make(id, num_envs=1, asynchronous=True, **kwargs):
    """Create a vectorized environment from multiple copies of an environment,
    from its id

    Parameters
    ----------
    id : str
        The environment ID. This must be a valid ID from the registry.

    num_envs : int
        Number of copies of the environment. If `1`, then it returns an
        unwrapped (i.e. non-vectorized) environment.

    asynchronous : bool (default: `True`)
        If `True`, wraps the environments in an `AsyncVectorEnv` (which uses 
        `multiprocessing` to run the environments in parallel). If `False`,
        wraps the environments in a `SyncVectorEnv`.

    Returns
    -------
    env : `gym.vector.VectorEnv` instance
        The vectorized environment.

    Example
    -------
    >>> import gym
    >>> env = gym.vector.make('CartPole-v1', 3)
    >>> env.reset()
    array([[-0.04456399,  0.04653909,  0.01326909, -0.02099827],
           [ 0.03073904,  0.00145001, -0.03088818, -0.03131252],
           [ 0.03468829,  0.01500225,  0.01230312,  0.01825218]],
          dtype=float32)
    """
    from gym.envs import make as make_
    def _make_env():
        return make_(id, **kwargs)
    if num_envs == 1:
        return _make_env()
    env_fns = [_make_env for _ in range(num_envs)]

    return AsyncVectorEnv(env_fns) if asynchronous else SyncVectorEnv(env_fns)
