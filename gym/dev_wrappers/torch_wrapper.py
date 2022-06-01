from typing import Optional, Tuple, Union

import torch
from brax.io.torch import jax_to_torch, torch_to_jax

from gym import Env, Wrapper


class TorchWrapper(Wrapper):
    """This wrapper will convert torch inputs for the actions and observations to Jax arrays
    for an underlying Jax environment then convert the return observations from Jax arrays
    back to torch tensors.

    Args:
        env: The Jax-based environment to wrap
        device: The device the torch Tensors should be moved to
    """

    def __init__(self, env: Union[Wrapper, Env], device: Optional[torch.device] = None):

        super().__init__(env)
        self.device: Optional[torch.device] = device
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        jax_action = torch_to_jax(action)
        obs, reward, done, info = self.env.step(jax_action)

        # TODO: Verify this captures vector envs and converts everything from jax to torch
        if self.is_vector_env:
            obs = [jax_to_torch(o, device=self.device) for o in obs]
        else:
            obs = jax_to_torch(obs, device=self.device)

        return obs, reward, done, info

    def reset(self, **kwargs) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        return_info = kwargs.get("return_info", False)
        if return_info:
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)

        # TODO: Verify this captures vector envs and converts everything from jax to torch
        if self.is_vector_env:
            obs = [jax_to_torch(o, device=self.device) for o in obs]
        else:
            obs = jax_to_torch(obs, device=self.device)

        if return_info:
            return obs, info
        else:
            return obs
