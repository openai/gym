# ----------------------------------------------------------------------------
# pyglet
# Copyright (c) 2006-2008 Alex Holkner
# Copyright (c) 2008-2020 pyglet contributors
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
import os.path


_decoders = []              # List of registered ModelDecoders
_encoders = []              # List of registered ModelEncoders
_decoder_extensions = {}    # Map str -> list of matching ModelDecoders
_encoder_extensions = {}    # Map str -> list of matching ModelEncoders


class ModelDecodeException(Exception):
    exception_priority = 10


class ModelEncodeException(Exception):
    pass


class ModelDecoder:
    def get_file_extensions(self):
        """Return a list of accepted file extensions, e.g. ['.obj', '.gox']
        Lower-case only.
        """
        return []

    def decode(self, file, filename, batch):
        """Decode the given file object and return an instance of `Model`.
        Throws ModelDecodeException if there is an error.  filename
        can be a file type hint.
        """
        raise NotImplementedError()

    def __repr__(self):
        return "{0}{1}".format(self.__class__.__name__, self.get_file_extensions())


class ModelEncoder:
    def get_file_extensions(self):
        """Return a list of accepted file extensions, e.g. ['.obj', '.gox']
        Lower-case only.
        """
        return []

    def encode(self, model, file, filename):
        """Encode the given model to the given file.  filename
        provides a hint to the file format desired.  options are
        encoder-specific, and unknown options should be ignored or
        issue warnings.
        """
        raise NotImplementedError()

    def __repr__(self):
        return "{0}{1}".format(self.__class__.__name__, self.get_file_extensions())


def get_encoders(filename=None):
    """Get an ordered list of all encoders. If a `filename` is provided,
    encoders supporting that extension will be ordered first in the list.
    """
    encoders = []
    if filename:
        extension = os.path.splitext(filename)[1].lower()
        encoders += _encoder_extensions.get(extension, [])
    encoders += [e for e in _encoders if e not in encoders]
    return encoders


def get_decoders(filename=None):
    """Get an ordered list of all decoders. If a `filename` is provided,
    decoders supporting that extension will be ordered first in the list.
    """
    decoders = []
    if filename:
        extension = os.path.splitext(filename)[1].lower()
        decoders += _decoder_extensions.get(extension, [])
    return decoders


def add_decoders(module):
    """Add a decoder module.  The module must define `get_decoders`.  Once
    added, the appropriate decoders defined in the codec will be returned by
    pyglet.model.codecs.get_decoders.
    """
    for decoder in module.get_decoders():
        _decoders.append(decoder)
        for extension in decoder.get_file_extensions():
            if extension not in _decoder_extensions:
                _decoder_extensions[extension] = []
            _decoder_extensions[extension].append(decoder)


def add_encoders(module):
    """Add an encoder module.  The module must define `get_encoders`.  Once
    added, the appropriate encoders defined in the codec will be returned by
    pyglet.model.codecs.get_encoders.
    """
    for encoder in module.get_encoders():
        _encoders.append(encoder)
        for extension in encoder.get_file_extensions():
            if extension not in _encoder_extensions:
                _encoder_extensions[extension] = []
            _encoder_extensions[extension].append(encoder)


def add_default_model_codecs():
    # Add all bundled codecs. These should be listed in order of
    # preference. This is called automatically by pyglet.model.

    try:
        from pyglet.model.codecs import obj
        add_decoders(obj)
    except ImportError:
        pass

    # TODO: complete this decoder, and enable it by default
    # try:
    #     from pyglet.model.codecs import gltf
    #     add_decoders(gltf)
    # except ImportError:
    #     pass
