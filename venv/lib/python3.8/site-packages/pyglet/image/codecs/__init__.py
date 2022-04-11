# ----------------------------------------------------------------------------
# pyglet
# Copyright (c) 2006-2008 Alex Holkner
# Copyright (c) 2008-2021 pyglet contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

"""Collection of image encoders and decoders.

Modules must subclass ImageDecoder and ImageEncoder for each method of
decoding/encoding they support.

Modules must also implement the two functions::

    def get_decoders():
        # Return a list of ImageDecoder instances or []
        return []

    def get_encoders():
        # Return a list of ImageEncoder instances or []
        return []

"""

import os.path

from pyglet.util import Codecs, Decoder, Encoder, DecodeException, EncodeException
from pyglet import compat_platform


class _ImageCodecs(Codecs):
    """Subclass of Codecs that adds support for animation methods."""

    def __init__(self):
        self._decoder_animation_extensions = {}
        super().__init__()

    def add_decoders(self, module):
        """Override the default method to also add animation decoders.
        """
        super().add_decoders(module)
        for decoder in module.get_decoders():
            for extension in decoder.get_animation_file_extensions():
                if extension not in self._decoder_animation_extensions:
                    self._decoder_animation_extensions[extension] = []
                self._decoder_animation_extensions[extension].append(decoder)

    def get_animation_decoders(self, filename=None):
        """Get an ordered list of all animation decoders. If a `filename` is
        provided, decoders supporting that extension will be ordered first
        in the list.
        """
        decoders = []
        if filename:
            extension = os.path.splitext(filename)[1].lower()
            decoders += self._decoder_animation_extensions.get(extension, [])
        decoders += [e for e in self._decoders if e not in decoders]
        return decoders


_codecs = _ImageCodecs()

add_decoders = _codecs.add_decoders
get_decoders = _codecs.get_decoders
add_encoders = _codecs.add_encoders
get_encoders = _codecs.get_encoders
get_animation_decoders = _codecs.get_animation_decoders


class ImageDecodeException(DecodeException):
    pass


class ImageEncodeException(EncodeException):
    pass


class ImageDecoder(Decoder):

    def get_animation_file_extensions(self):
        """Return a list of accepted file extensions, e.g. ['.gif', '.flc']
        Lower-case only.
        """
        return []

    def decode(self, file, filename):
        """Decode the given file object and return an instance of `Image`.
        Throws ImageDecodeException if there is an error.  filename
        can be a file type hint.
        """
        raise NotImplementedError()

    def decode_animation(self, file, filename):
        """Decode the given file object and return an instance of :py:class:`~pyglet.image.Animation`.
        Throws ImageDecodeException if there is an error.  filename
        can be a file type hint.
        """
        raise ImageDecodeException('This decoder cannot decode animations.')

    def __repr__(self):
        return "{0}{1}".format(self.__class__.__name__,
                               self.get_animation_file_extensions() +
                               self.get_file_extensions())


class ImageEncoder(Encoder):

    def encode(self, image, file, filename):
        """Encode the given image to the given file.  filename
        provides a hint to the file format desired.
        """
        raise NotImplementedError()

    def __repr__(self):
        return "{0}{1}".format(self.__class__.__name__, self.get_file_extensions())


def add_default_image_codecs():
    # Add the codecs we know about.  These should be listed in order of
    # preference.  This is called automatically by pyglet.image.

    # Compressed texture in DDS format
    try:
        from pyglet.image.codecs import dds
        add_encoders(dds)
        add_decoders(dds)
    except ImportError:
        pass

    # Mac OS X default: Quartz
    if compat_platform == 'darwin':
        try:
            from pyglet.image.codecs import quartz
            add_encoders(quartz)
            add_decoders(quartz)
        except ImportError:
            pass

    # Windows 7 default: Windows Imaging Component
    if compat_platform in ('win32', 'cygwin'):
        from pyglet.libs.win32.constants import WINDOWS_7_OR_GREATER
        if WINDOWS_7_OR_GREATER:  # Supports Vista and above.
            try:
                from pyglet.image.codecs import wic
                add_encoders(wic)
                add_decoders(wic)
            except ImportError:
                pass

    # Windows XP default: GDI+
    if compat_platform in ('win32', 'cygwin'):
        try:
            from pyglet.image.codecs import gdiplus
            add_encoders(gdiplus)
            add_decoders(gdiplus)
        except ImportError:
            pass

    # Linux default: GdkPixbuf 2.0
    if compat_platform.startswith('linux'):
        try:
            from pyglet.image.codecs import gdkpixbuf2
            add_encoders(gdkpixbuf2)
            add_decoders(gdkpixbuf2)
        except ImportError:
            pass

    # Fallback: PIL
    try:
        from pyglet.image.codecs import pil
        add_encoders(pil)
        add_decoders(pil)
    except ImportError:
        pass

    # Fallback: PNG loader (slow)
    try:
        from pyglet.image.codecs import png
        add_encoders(png)
        add_decoders(png)
    except ImportError:
        pass

    # Fallback: BMP loader (slow)
    try:
        from pyglet.image.codecs import bmp
        add_encoders(bmp)
        add_decoders(bmp)
    except ImportError:
        pass
