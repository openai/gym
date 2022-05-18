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

