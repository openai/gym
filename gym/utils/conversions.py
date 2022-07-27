from typing import Any, Dict, Optional, Tuple, Union

from jax._src import dlpack as jax_dlpack
from jax.interpreters.xla import DeviceArray

from gym.error import DependencyNotInstalled

try:
    import torch
    from torch.utils import dlpack as torch_dlpack
except ImportError:
    raise DependencyNotInstalled("torch is not installed, run `pip install torch`")

try:
    import tensorflow as tf
except ImportError:
    raise DependencyNotInstalled(
        "tensorflow is not installed, run `pip install tensorflow`"
    )


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


def tf_to_jax(arr: tf.Tensor):
    # TODO: Verify this captures vector envs and converts everything from tensorflow to jax
    return jax_dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(arr))

def jax_to_tf(arr):
    # TODO: Verify this captures vector envs and converts everything from jax to tensorflow
    return tf.experimental.dlpack.from_dlpack(jax_dlpack.to_dlpack(arr))

def tf_dict_to_jax(
     value: Dict[str, Union[tf.Tensor, Any]]
) -> Dict[str, Union[DeviceArray, Any]]:
    return type(value)(**{k: tf_to_jax(v) for k, v in value.items()})

def jax_dict_to_tf(
    value: Dict[str, Union[DeviceArray, Any]]
) -> Dict[str, Union[tf.Tensor, Any]]:
    return type(value)(**{k: tf_to_jax(v) for k, v in value.items()})
