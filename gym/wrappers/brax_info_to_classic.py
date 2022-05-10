from typing import List

import gym


class BraxInfoToClassic(gym.Wrapper):
    """This wrapper converts the `Brax` info format of a
    vector environment to the `classic` info format.

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
        for k in infos:
            if k.startswith("_"):
                continue
            for i, has_info in enumerate(infos[f"_{k}"]):

                # TODO: simplify this block
                # used when this wrapper wraps also RecordEpisodeStatistic
                if k == "episode":
                    for statistic in ["r", "l", "t"]:
                        if "episode" not in classic_info[i] and has_info:
                            classic_info[i]["episode"] = {}
                        if has_info:
                            classic_info[i]["episode"][statistic] = infos[k][statistic][
                                i
                            ]
                    continue

                if has_info:
                    classic_info[i][k] = infos[k][i]
        return classic_info
