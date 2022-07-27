"""
This wrapper will convert torch inputs for the actions and observations to Jax arrays
for an underlying Jax environment then convert the return observations from Jax arrays
back to torch tensors.

Functionality for converting between torch and jax types originally copied from
https://github.com/google/brax/blob/9d6b7ced2a13da0d074b5e9fbd3aad8311e26997/brax/io/torch.py
Under the Apache 2.0 license. Copyright is held by the authors
"""

from typing import Any, Dict, Optional, Tuple, Union

from jax._src import dlpack as jax_dlpack
from jax.interpreters.xla import DeviceArray

from gym import Env, Wrapper
from gym.error import DependencyNotInstalled

try:
    import torch
    from torch.utils import dlpack as torch_dlpack
except ImportError:
    raise DependencyNotInstalled("torch is not installed, run `pip install torch`")


Device = Union[str, torch.device]

def torch_to_jax(value: torch.Tensor) -> DeviceArray:
    """Converts a PyTorch Tensor into a Jax DeviceArray."""
    tensor = torch_dlpack.to_dlpack(value)
    tensor = jax_dlpack.from_dlpack(tensor)
    return tensor


def jax_to_torch(value: DeviceArray, device: Device = None) -> torch.Tensor:
    """Converts a Jax DeviceArray into a PyTorch Tensor."""
    dlpack = jax_dlpack.to_dlpack(value.astype("float32"))
    tensor = torch_dlpack.from_dlpack(dlpack)
    if device:
        return tensor.to(device=device)
    else:
        return tensor


def torch_dict_to_jax(
    value: Dict[str, Union[torch.Tensor, Any]]
) -> Dict[str, Union[DeviceArray, Any]]:
    """Converts a dictionary of PyTorch Tensors into a Dictionary of Jax DeviceArrays."""
    return type(value)(**{k: torch_to_jax(v) for k, v in value.items()})


def jax_dict_to_torch(
     value: Dict[str, Union[DeviceArray, Any]], device: Device = None
) -> Dict[str, Union[torch.Tensor, Any]]:
    """Converts a dictionary of Jax DeviceArrays into a Dictionary of PyTorch Tensors."""
    return type(value)(
        **{k: jax_to_torch(v, device) for k, v in value.items()}
    )


class JaxToTorchV0(Wrapper):
    def __init__(self, env: Union[Wrapper, Env], device: Optional[torch.device] = None):
        """
        Args:
            env: The Jax-based environment to wrap
            device: The device the torch Tensors should be moved to
        """

        super().__init__(env)
        self.device: Optional[torch.device] = device

    def step(
        self, action: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], float, bool, Dict]:
        jax_action = torch_to_jax(action)
        obs, reward, done, info = self.env.step(jax_action)

        # TODO: Verify this captures vector envs and converts everything from jax to torch
        if type(obs) == dict:
            obs = jax_dict_to_torch(obs, device=self.device)
        else:
            obs = jax_to_torch(obs, device=self.device)

        return obs, reward, done, info

    def reset(self, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        result = self.env.reset(**kwargs)
        if kwargs.get("return_info", False):
            if type(result[0]) == dict:
                return jax_dict_to_torch(result[0], device=self.device), result[1]
            else:
                return jax_to_torch(result[0], device=self.device), result[1]
        else:
            if type(result) == dict:
                return jax_dict_to_torch(result, device=self.device)
            else:
                return jax_to_torch(result, device=self.device)

