import gym

__all__ = ['HumanPlayer', 'Skip1', 'Skip2', 'Skip3', 'Skip4', 'Skip5']

# Helper functions

def skip_n(wrapper, action, skip_steps):
    done = False
    rew_total = 0
    current_step = 0
    while current_step < (skip_steps + 1) and not done:
        obs, rew, done, info = wrapper.env.step(action)
        rew_total += rew
        current_step += 1
    return obs, rew_total, done, info

# Wrappers

class HumanPlayer(gym.Wrapper):
    """ Allows a human to play the level """
    def __init__(self, env):
        self._uploadable = False
        self._unwrapped._mode = 'human'

class Skip1(gym.Wrapper):
    """ Sends action, and repeats it for 1 additional step """
    def _step(self, action):
        return skip_n(self, action, 1)

class Skip2(gym.Wrapper):
    """ Sends action, and repeats it for 2 additional steps """
    def _step(self, action):
        return skip_n(self, action, 2)

class Skip3(gym.Wrapper):
    """ Sends action, and repeats it for 3 additional steps """
    def _step(self, action):
        return skip_n(self, action, 3)

class Skip4(gym.Wrapper):
    """ Sends action, and repeats it for 4 additional steps """
    def _step(self, action):
        return skip_n(self, action, 4)

class Skip5(gym.Wrapper):
    """ Sends action, and repeats it for 5 additional steps """
    def _step(self, action):
        return skip_n(self, action, 5)
