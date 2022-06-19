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


class jax_to_torch_v0(Wrapper):
    def __init__(self, env: Union[Wrapper, Env], device: Optional[torch.device] = None):
        """This wrapper will convert torch inputs for the actions and observations to Jax arrays
        for an underlying Jax environment then convert the return observations from Jax arrays
        back to torch tensors.

        Functionality for converting between torch and jax types originally copied from
        https://github.com/google/brax/blob/9d6b7ced2a13da0d074b5e9fbd3aad8311e26997/brax/io/torch.py
        Under the Apache 2.0 license. Copyright is held by the authors

        Args:
            env: The Jax-based environment to wrap
            device: The device the torch Tensors should be moved to
        """

        super().__init__(env)
        self.device: Optional[torch.device] = device

    def step(
        self, action: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], float, bool, dict]:
        jax_action = self._torch_to_jax(action)
        obs, reward, done, info = self.env.step(jax_action)

        # TODO: Verify this captures vector envs and converts everything from jax to torch
        if type(obs) == dict:
            obs = self._jax_dict_torch(obs, device=self.device)
        else:
            obs = self._jax_to_torch(obs, device=self.device)

        return obs, reward, done, info

    def reset(self, **kwargs) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        result = self.env.reset(**kwargs)
        if kwargs.get("return_info", False):
            if type(result[0]) == dict:
                return self._jax_dict_to_torch(result[0], device=self.device), result[1]
            else:
                return self._jax_to_torch(result[0], device=self.device), result[1]
        else:
            if type(result) == dict:
                return self._jax_dict_to_torch(result, device=self.device)
            else:
                return self._jax_to_torch(result, device=self.device)

    def _torch_to_jax(self, value: torch.Tensor) -> DeviceArray:
        """Converts a PyTorch Tensor into a Jax DeviceArray."""
        tensor = torch_dlpack.to_dlpack(value)
        tensor = jax_dlpack.from_dlpack(tensor)
        return tensor

    def _jax_to_torch(self, value: DeviceArray, device: Device = None) -> torch.Tensor:
        dlpack = jax_dlpack.to_dlpack(value.astype("float32"))
        tensor = torch_dlpack.from_dlpack(dlpack)
        if device:
            return tensor.to(device=device)
        else:
            return tensor

    def _torch_dict_to_jax(
        self, value: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[DeviceArray, Any]]:
        return type(value)(**{k: self._torch_to_jax(v) for k, v in value.items()})

    def _jax_dict_to_torch(
        self, value: Dict[str, Union[DeviceArray, Any]], device: Device = None
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        return type(value)(
            **{k: self._jax_to_torch(v, device) for k, v in value.items()}
        )
