from typing import List

import gym


class BraxInfoToClassic(gym.Wrapper):
    """Converts `Brax` info format to `classic`.

    This wrapper converts the `Brax` info format of a
    vector environment to the `classic` info format.
    This wrapper is intended to be used around vectorized
    environments. If using other wrappers that perform
    operation on info like `RecordEpisodeStatistics` this
    need to be the outermost wrapper.

    i.e. BraxInfoToClassic(RecordEpisodeStatistics(envs))

    Example::

    >>> # brax
    ...  {
    ...      k: np.array[0., 0., 0.5, 0.3],
    ...      _k: np.array[False, False, True, True]
    ...  }
    ...
    ... # classic
    ... [{}, {}, {k: 0.5}, {k: 0.3}]

    """

    def __init__(self, env):
        assert getattr(
            env, "is_vector_env", False
        ), "This wrapper can only be used in vectorized environments."
        super().__init__(env)

    def step(self, action):
        observation, reward, done, infos = self.env.step(action)
        classic_info = self._convert_brax_info_to_classic(infos)

        return observation, reward, done, classic_info

    def reset(self, **kwargs):
        if not kwargs.get("return_info"):
            obs = self.env.reset(**kwargs)
            return obs

        obs, infos = self.env.reset(**kwargs)
        classic_info = self._convert_brax_info_to_classic(infos)
        return obs, classic_info

    def _convert_brax_info_to_classic(self, infos: dict) -> List[dict]:
        classic_info = [{} for _ in range(self.num_envs)]
        classic_info = self._process_episode_statistics(infos, classic_info)
        for k in infos:
            if k.startswith("_"):
                continue
            for i, has_info in enumerate(infos[f"_{k}"]):
                if has_info:
                    classic_info[i][k] = infos[k][i]
        return classic_info

    def _process_episode_statistics(self, infos, classic_info):
        episode_statistics = infos.pop("episode", False)
        if not episode_statistics:
            return classic_info

        episode_statistics_mask = infos.pop("_episode")
        for i, has_info in enumerate(episode_statistics_mask):
            if has_info:
                classic_info[i]["episode"] = {}
                classic_info[i]["episode"]["r"] = episode_statistics["r"][i]
                classic_info[i]["episode"]["l"] = episode_statistics["l"][i]
                classic_info[i]["episode"]["t"] = episode_statistics["t"][i]

        return classic_info
