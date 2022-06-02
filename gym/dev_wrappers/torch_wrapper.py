from typing import Optional, Tuple, Union

from gym import Env, Wrapper
from gym.error import DependencyNotInstalled

try:
    import torch
except ImportError:
    raise DependencyNotInstalled("torch is not installed, run `pip install torch`")

from brax.io.torch import jax_to_torch, torch_to_jax


class jax_to_torch_v0(Wrapper):
    def __init__(self, env: Union[Wrapper, Env], device: Optional[torch.device] = None):
        """This wrapper will convert torch inputs for the actions and observations to Jax arrays
        for an underlying Jax environment then convert the return observations from Jax arrays
        back to torch tensors.

        Args:
            env: The Jax-based environment to wrap
            device: The device the torch Tensors should be moved to
        """

        super().__init__(env)
        self.device: Optional[torch.device] = device

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        jax_action = torch_to_jax(action)
        obs, reward, done, info = self.env.step(jax_action)

        # TODO: Verify this captures vector envs and converts everything from jax to torch
        obs = jax_to_torch(obs, device=self.device)

        return obs, reward, done, info

    def reset(self, **kwargs) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        result = self.env.reset(**kwargs)
        if kwargs.get("return_info", False):
            return jax_to_torch(result[0], device=self.device), result[1]
        else:
            return jax_to_torch(result, device=self.device)
