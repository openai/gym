import gym
from gym.spaces import Tuple
from gym.vector.utils.spaces import batch_space
from gym.vector.utils.tile_images import tile_images

__all__ = ['VectorEnv']


class VectorEnv(gym.Env):
    """Base class for vectorized environments.

    Parameters
    ----------
    num_envs : int
        Number of environments in the vectorized environment.

    observation_space : `gym.spaces.Space` instance
        Observation space of a single environment.

    action_space : `gym.spaces.Space` instance
        Action space of a single environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        super(VectorEnv, self).__init__()
        self.num_envs = num_envs
        self.observation_space = batch_space(observation_space, n=num_envs)
        self.action_space = Tuple((action_space,) * num_envs)

        self.closed = False
        self.viewer = None

        # The observation and action spaces of a single environment are
        # kept in separate properties
        self.single_observation_space = observation_space
        self.single_action_space = action_space

    def reset_async(self):
        pass

    def reset_wait(self, **kwargs):
        raise NotImplementedError()

    def reset(self):
        self.reset_async()
        return self.reset_wait()

    def step_async(self, actions):
        pass

    def step_wait(self, **kwargs):
        raise NotImplementedError()

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        raise NotImplementedError

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer

    def __del__(self):
        if hasattr(self, 'closed'):
            if not self.closed:
                self.close()
