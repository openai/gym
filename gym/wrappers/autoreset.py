import gym
from gym.wrappers.step_compatibility import step_api_compatibility


@step_api_compatibility
class AutoResetWrapper(gym.Wrapper):
    """
    A class for providing an automatic reset functionality
    for gym environments when calling self.step().

    When calling step causes self.env.step() to return terminated or truncated,
    self.env.reset() is called,
    and the return format of self.step() is as follows:

    new_obs, closing_reward, closing_terminated, closing_truncated, closing_info

    new_obs is the first observation after calling self.env.reset(),

    closing_reward is the reward after calling self.env.step(),
    prior to calling self.env.reset()

    (closing_terminated or closing_truncated) is True

    info is a dict containing all the keys from the info dict returned by
    the call to self.env.reset(), with an additional key "closing_observation"
    containing the observation returned by the last call to self.env.step()
    and "closing_info" containing the info dict returned by the last call
    to self.env.step().

    If (terminated or truncated) is not true when self.env.step() is called, self.step() returns
    obs, reward, terminated, truncated, and info as normal.

    Warning: When using this wrapper to collect rollouts, note
    that the when self.env.step() returns terminated=True or truncated=True, a
    new observation from after calling self.env.reset() is returned
    by self.step() alongside the closing reward and done state from the
    previous episode . If you need the closing state from the previous
    episode, you need to retrieve it via the the "closing_observation" key
    in the info dict. Make sure you know what you're doing if you
    use this wrapper!
    """

    new_step_api = True  # whether this wrapper is written in new API (assumed old API if not present)

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.new_step_api = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self._get_env_step_returns(action)

        if terminated or truncated:

            new_obs, new_info = self.env.reset(return_info=True)
            assert (
                "closing_observation" not in new_info
            ), 'info dict cannot contain key "closing_observation" '
            assert (
                "closing_info" not in new_info
            ), 'info dict cannot contain key "closing_info" '

            new_info["closing_observation"] = obs
            new_info["closing_info"] = info

            obs = new_obs
            info = new_info

        return obs, reward, terminated, truncated, info
