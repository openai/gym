from typing import Optional, Tuple, Union

from brax.io.torch import (
    torch_to_jax,
    jax_to_torch
)
import torch

from gym import Env, Wrapper

class TorchWrapper(Wrapper):
    def __init__(self,
        env: Union[Wrapper, Env],
        device: Optional[torch.device] = None):

        super().__init__(env)
        self.device: Optional[torch.device] = device

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        jax_action = torch_to_jax(action)
        obs, reward, done, info = super().step(jax_action)

        return jax_to_torch(obs, device=self.device), reward, done, info

    def reset(self, **kwargs) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        return_info = kwargs.get("return_info", False)
        if return_info:
            obs, info = super().reset(**kwargs)
        else:
            obs = super().reset(**kwargs)

        obs = jax_to_torch(obs, device=self.device)

        # TODO: handle vector_env here?
        if return_info:
            return obs, info
        else:
            return obs

