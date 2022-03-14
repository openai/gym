import gym


class AutoResetWrapper(gym.Wrapper):
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

    info is a dict containing all the keys from the info dict returned by
    the call to self.env.reset(), with an additional key 'final_obs"
    containing the observation returned by the last call to self.env.step()
    and "final_info" containing the info dict returned by the last call
    to self.env.step().

    If done is not true when self.env.step() is called, self.step() returns
    obs, reward, done, and info as normal.

    Warning: When using this wrapper to collect rollouts, note
    that the when self.env.step() returns done, a
    new observation from after calling self.env.reset() is returned
    by self.step() alongside the final reward and done state from the
    previous episode . If you need the final state from the previous
    episode, you need to retrieve it via the the "final_obs" key
    in the info dict. Make sure you know what you're doing if you
    use this wrapper!
    """

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if done:

            new_obs, new_info = self.env.reset(return_info=True)
            assert "final_obs" not in new_info
            assert "final_info" not in new_info

            new_info["final_obs"] = obs
            new_info["final_info"] = info

            obs = new_obs
            info = new_info

        return obs, reward, done, info
