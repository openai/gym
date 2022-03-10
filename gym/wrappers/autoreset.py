"""
A class for providing an automatic reset functionality
for gym environments when calling self.step().

When calling step causes self.env.step() to return done,
self.env.reset() is called,
and the return format of self.step() is as follows:

new_obs, final_reward, final_done, info

new_obs is the first observation after calling self.env.reset(),

final_reward is the reward after calling self.env.step(),
prior to calling self.env.reset()

final_done is always True

info is a dict of the form {info:{<self.env info>}, "final_obs":<the
observation after calling self.env.step(), prior to calling
self.env.reset()>}

If done is not true when self.env.step() is called, self.step() returns
obs, reward, and done as normal, and wraps the info returned
by from self.env.step() in another layer of dictionary with the key
"info", to preserve the structure of the info return across cases.

Warning: When using this wrapper to collect rollouts, note
that the when self.env.step() returns done, a
new observation from after calling reset is returned
alongside the final reward and done state from the
previous episode by self.step(). It's important to retrieve the final state
from the "final_obs" key in the info dict. Make sure you know what you're
doing if you use this wrapper!
"""


import gym


class AutoResetWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            new_obs, new_info = self.env.reset(return_info=True)
            return new_obs, done, reward, {"info": new_info, "final_obs": obs}
        else:
            return obs, reward, done, {"info": info}
