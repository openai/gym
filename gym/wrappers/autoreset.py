"""Wrapper that autoreset environments when `done=True`."""
import gym


class AutoResetWrapper(gym.Wrapper):
    """A class for providing an automatic reset functionality for gym environments when calling :meth:`self.step`.

    When calling step causes :meth:`Env.step` to return done, :meth:`Env.reset` is called,
    and the return format of :meth:`self.step` is as follows: ``(new_obs, terminal_reward, terminal_done, info)``
     - ``new_obs`` is the first observation after calling :meth:`self.env.reset`
     - ``terminal_reward`` is the reward after calling :meth:`self.env.step`, prior to calling :meth:`self.env.reset`.
     - ``terminal_done`` is always True
     - ``info`` is a dict containing all the keys from the info dict returned by the call to :meth:`self.env.reset`,
       with an additional key "terminal_observation" containing the observation returned by the last call to :meth:`self.env.step`
       and "terminal_info" containing the info dict returned by the last call to :meth:`self.env.step`.

    Warning: When using this wrapper to collect rollouts, note that when :meth:`Env.step` returns done, a
        new observation from after calling :meth:`Env.reset` is returned by :meth:`Env.step` alongside the
        terminal reward and done state from the previous episode.
        If you need the terminal state from the previous episode, you need to retrieve it via the
        "terminal_observation" key in the info dict.
        Make sure you know what you're doing if you use this wrapper!
    """

    def step(self, action):
        """Steps through the environment with action and resets the environment if a done-signal is encountered.

        Args:
            action: The action to take

        Returns:
            The autoreset environment :meth:`step`
        """
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
