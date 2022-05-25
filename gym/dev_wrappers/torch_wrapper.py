from typing import Union, Tuple

from brax.io.torch import (
    torch_to_jax,
    jax_to_torch
)
import torch

from gym import Wrapper
class TorchWrapper(Wrapper):
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        jax_action = torch_to_jax(action)

        # TODO: look at the device argument for moving returned observation back to device
        return jax_to_torch(self.env.step(jax_action))

    def reset(self, **kwargs) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        return_info = kwargs.get("return_info", False)
        if return_info:
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)

        obs = jax_to_torch(obs)

        # TODO: handle vector_env here?
        if return_info:
            return obs, info
        else:
            return obs

    @property
    def action_space(self):
        act_space = self.env.action_space if self._action_space is None else self._action_space
        return jax_to_torch(act_space)

    @action_space.setter
    def action_space(self, space: torch.Tensor):
        self._action_space = torch_to_jax(space)

    @property
    def observation_space(self):
        obs_space = self.env.observation_space if self._observation_space is None else self._observation_space
        return jax_to_torch(obs_space)
