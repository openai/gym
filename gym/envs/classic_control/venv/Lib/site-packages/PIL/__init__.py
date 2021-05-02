"""Pillow (Fork of the Python Imaging Library)

Pillow is the friendly PIL fork by Alex Clark and Contributors.
    https://github.com/python-pillow/Pillow/

Pillow is forked from PIL 1.1.7.

PIL is the Python Imaging Library by Fredrik Lundh and Contributors.
Copyright (c) 1999 by Secret Labs AB.

Use PIL.__version__ for this Pillow version.

;-)
"""

import sys
import warnings

from . import _version

# VERSION was removed in Pillow 6.0.0.
__version__ = _version.__version__


# PILLOW_VERSION is deprecated and will be removed in a future release.
# Use __version__ instead.
def _raise_version_warning():
    warnings.warn(
        "PILLOW_VERSION is deprecated and will be removed in a future release. "
        "Use __version__ instead.",
        DeprecationWarning,
        stacklevel=3,
    )


if sys.version_info >= (3, 7):

    def __getattr__(name):
        if name == "PILLOW_VERSION":
            _raise_version_warning()
            return __version__
        raise AttributeError("module '{}' has no attribute '{}'".format(__name__, name))


else:

    class _Deprecated_Version(str):
        def __str__(self):
            _raise_version_warning()
            return super().__str__()

        def __getitem__(self, key):
            _raise_version_warning()
            return super().__getitem__(key)

        def __eq__(self, other):
            _raise_version_warning()
            return super().__eq__(other)

        def __ne__(self, other):
            _raise_version_warning()
            return super().__ne__(other)

        def __gt__(self, other):
            _raise_version_warning()
            return super().__gt__(other)

        def __lt__(self, other):
            _raise_version_warning()
            return super().__lt__(other)

        def __ge__(self, other):
            _raise_version_warning()
            return super().__gt__(other)

        def __le__(self, other):
            _raise_version_warning()
            return super().__lt__(other)

    PILLOW_VERSION = _Deprecated_Version(__version__)

del _version


_plugins = [
    "BlpImagePlugin",
    "BmpImagePlugin",
    "BufrStubImagePlugin",
    "CurImagePlugin",
    "DcxImagePlugin",
    "DdsImagePlugin",
    "EpsImagePlugin",
    "FitsStubImagePlugin",
    "FliImagePlugin",
    "FpxImagePlugin",
    "FtexImagePlugin",
    "GbrImagePlugin",
    "GifImagePlugin",
    "GribStubImagePlugin",
    "Hdf5StubImagePlugin",
    "IcnsImagePlugin",
    "IcoImagePlugin",
    "ImImagePlugin",
    "ImtImagePlugin",
    "IptcImagePlugin",
    "JpegImagePlugin",
    "Jpeg2KImagePlugin",
    "McIdasImagePlugin",
    "MicImagePlugin",
    "MpegImagePlugin",
    "MpoImagePlugin",
    "MspImagePlugin",
    "PalmImagePlugin",
    "PcdImagePlugin",
    "PcxImagePlugin",
    "PdfImagePlugin",
    "PixarImagePlugin",
    "PngImagePlugin",
    "PpmImagePlugin",
    "PsdImagePlugin",
    "SgiImagePlugin",
    "SpiderImagePlugin",
    "SunImagePlugin",
    "TgaImagePlugin",
    "TiffImagePlugin",
    "WebPImagePlugin",
    "WmfImagePlugin",
    "XbmImagePlugin",
    "XpmImagePlugin",
    "XVThumbImagePlugin",
]


class UnidentifiedImageError(OSError):
    """
    Raised in :py:meth:`PIL.Image.open` if an image cannot be opened and identified.
    """

    pass
