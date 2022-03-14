import gym


class AutoResetWrapper(gym.Wrapper):
    """
    A class for providing an automatic reset functionality
    for gym environments when calling self.step().

    When calling step causes self.env.step() to return done,
    self.env.reset() is called,
    and the return format of self.step() is as follows:

    new_obs, terminal_reward, terminal_done, info

    new_obs is the first observation after calling self.env.reset(),

    terminal_reward is the reward after calling self.env.step(),
    prior to calling self.env.reset()

    terminal_done is always True

    info is a dict containing all the keys from the info dict returned by
    the call to self.env.reset(), with an additional key "terminal_observation"
    containing the observation returned by the last call to self.env.step()
    and "terminal_info" containing the info dict returned by the last call
    to self.env.step().

    If done is not true when self.env.step() is called, self.step() returns
    obs, reward, done, and info as normal.

    Warning: When using this wrapper to collect rollouts, note
    that the when self.env.step() returns done, a
    new observation from after calling self.env.reset() is returned
    by self.step() alongside the terminal reward and done state from the
    previous episode . If you need the terminal state from the previous
    episode, you need to retrieve it via the the "terminal_observation" key
    in the info dict. Make sure you know what you're doing if you
    use this wrapper!
    """

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if done:

            new_obs, new_info = self.env.reset(return_info=True)
            assert (
                "terminal_observation" not in new_info
            ), 'info dict cannot contain key "terminal_observation" '
            assert (
                "terminal_info" not in new_info
            ), 'info dict cannot contain key "terminal_info" '

            new_info["terminal_observation"] = obs
            new_info["terminal_info"] = info

            obs = new_obs
            info = new_info

        return obs, reward, done, info
