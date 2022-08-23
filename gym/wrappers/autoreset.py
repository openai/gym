"""Wrapper that autoreset environments when `terminated=True` or `truncated=True`."""
import gym


class AutoResetWrapper(gym.Wrapper):
    """A class for providing an automatic reset functionality for gym environments when calling :meth:`self.step`.

    When calling step causes :meth:`Env.step` to return `terminated=True` or `truncated=True`, :meth:`Env.reset` is called,
    and the return format of :meth:`self.step` is as follows: ``(new_obs, final_reward, final_terminated, final_truncated, info)``
    with new step API and ``(new_obs, final_reward, final_done, info)`` with the old step API.
     - ``new_obs`` is the first observation after calling :meth:`self.env.reset`
     - ``final_reward`` is the reward after calling :meth:`self.env.step`, prior to calling :meth:`self.env.reset`.
     - ``final_terminated`` is the terminated value before calling :meth:`self.env.reset`.
     - ``final_truncated`` is the truncated value before calling :meth:`self.env.reset`. Both `final_terminated` and `final_truncated` cannot be False.
     - ``info`` is a dict containing all the keys from the info dict returned by the call to :meth:`self.env.reset`,
       with an additional key "final_observation" containing the observation returned by the last call to :meth:`self.env.step`
       and "final_info" containing the info dict returned by the last call to :meth:`self.env.step`.

    Warning: When using this wrapper to collect rollouts, note that when :meth:`Env.step` returns `terminated` or `truncated`, a
        new observation from after calling :meth:`Env.reset` is returned by :meth:`Env.step` alongside the
        final reward, terminated and truncated state from the previous episode.
        If you need the final state from the previous episode, you need to retrieve it via the
        "final_observation" key in the info dict.
        Make sure you know what you're doing if you use this wrapper!
    """

    def __init__(self, env: gym.Env):
        """A class for providing an automatic reset functionality for gym environments when calling :meth:`self.step`.

        Args:
            env (gym.Env): The environment to apply the wrapper
        """
        super().__init__(env)

    def step(self, action):
        """Steps through the environment with action and resets the environment if a terminated or truncated signal is encountered.

        Args:
            action: The action to take

        Returns:
            The autoreset environment :meth:`step`
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:

            new_obs, new_info = self.env.reset()
            assert (
                "final_observation" not in new_info
            ), 'info dict cannot contain key "final_observation" '
            assert (
                "final_info" not in new_info
            ), 'info dict cannot contain key "final_info" '

            new_info["final_observation"] = obs
            new_info["final_info"] = info

            obs = new_obs
            info = new_info

        return obs, reward, terminated, truncated, info
