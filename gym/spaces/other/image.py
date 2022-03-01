from __future__ import annotations
from typing import Optional, SupportsFloat
import numpy as np
from gym.spaces.box import Box


class ImageSpace(Box):
    """ Space of image arrays.

    This is just a `Box` space with some added convenience methods and properties.
    
    >>> mnist_images = ImageSpace(0, 255, (1, 28, 28), dtype=np.uint8)
    >>> mnist_images
    ImageSpace(0, 255, (1, 28, 28), uint8)
    >>> mnist_images.is_channels_first
    True
    >>> mnist_images.channels, mnist_images.height, mnist_images.width
    (1, 28, 28)
    
    >>> image_batch = ImageSpace(0, 1, (32, 3, 256, 256), dtype=np.float64)
    >>> image_batch
    ImageSpace(0.0, 1.0, (32, 3, 256, 256), float64)
    >>> image_batch.is_channels_first
    True
    >>> image_batch.batch_size
    32
    >>> image_batch.channels
    3
    >>> image_batch.height, image_batch.width
    (256, 256)
    """

    def __init__(
        self,
        low: SupportsFloat | np.ndarray,
        high: SupportsFloat | np.ndarray,
        shape: tuple[int, int, int] | tuple[int, int, int, int] | None = None,
        dtype: type = np.float32,
        seed: int | None = None,
    ) -> None:
        super().__init__(low, high, shape=shape, dtype=dtype, seed=seed)
        if len(self.shape) not in {3, 4}:
            raise RuntimeError(
                f"Can only create ImageSpaces when shape has 3 or 4 dimensions."
            )

    def contains(self, x) -> bool:
        return super().contains(x)

    @property
    def batch_size(self) -> Optional[int]:
        """ Gives the size of the batch_size dimension if present, else `None`. """
        return self.shape[0] if len(self.shape) == 4 else None

    @property
    def channels(self) -> int:
        """ Gives the number of channels. """
        return self.shape[self._format.index("C")]

    @property
    def height(self) -> int:
        """ Gives the size of the `height` dimension. """
        return self.shape[self._format.index("H")]

    @property
    def width(self) -> int:
        """ Gives the size of the `width` dimension. """
        return self.shape[self._format.index("W")]

    @property
    def is_channels_first(self) -> bool:
        """ Returns wether the images are in the channels-first format (i.e. "CHW" or "BCHW").

        Raises a RuntimeError if the format can't be determined.
        """

        def _is_channels_dim(d: int) -> bool:
            """ Simple test to check if the given dimension is a 'channels' dimension. """
            return d in {1, 3, 4}

        if self.ndim == 3:
            d1, _, d2 = self.shape
        else:
            # NOTE: Should only have 3 or 4 dimensions for now.
            assert self.ndim == 4
            _, d1, _, d2 = self.shape

        if _is_channels_dim(d1) and not _is_channels_dim(d2):
            return True
        elif not _is_channels_dim(d1) and _is_channels_dim(d2):
            return False
        else:
            raise RuntimeError(
                f"Can't tell if channels-first or channels-last given shape {self.shape}"
            )

    @property
    def ndim(self) -> int:
        """ Returns the number of dimensions. """
        return len(self.shape)

    @property
    def _format(self) -> str:
        """Returns the format string."""
        # NOTE: Limiting to just 3 or 4 dims for now.
        # if len(self.shape) == 2:
        #     return "HW"
        format = "CHW" if self.is_channels_first else "HWC"
        if self.batch_size is not None:
            format = "N" + format
        return format
