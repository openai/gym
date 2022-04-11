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

"""Decoder for RIFF Wave files, using the standard library wave module.
"""

import wave

from ..exceptions import MediaDecodeException, MediaEncodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder


class WAVEDecodeException(MediaDecodeException):
    pass


class WaveSource(StreamingSource):
    def __init__(self, filename, file=None):
        if file is None:
            file = open(filename, 'rb')

        self._file = file

        try:
            self._wave = wave.open(file)
        except wave.Error as e:
            raise WAVEDecodeException(e)

        nchannels, sampwidth, framerate, nframes, comptype, compname = self._wave.getparams()

        self.audio_format = AudioFormat(channels=nchannels, sample_size=sampwidth * 8, sample_rate=framerate)

        self._bytes_per_frame = nchannels * sampwidth
        self._duration = nframes / framerate
        self._duration_per_frame = self._duration / nframes
        self._num_frames = nframes

        self._wave.rewind()

    def __del__(self):
        if hasattr(self, '_file'):
            self._file.close()

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        num_frames = max(1, num_bytes // self._bytes_per_frame)

        data = self._wave.readframes(num_frames)
        if not data:
            return None

        timestamp = self._wave.tell() / self.audio_format.sample_rate
        duration = num_frames / self.audio_format.sample_rate
        return AudioData(data, len(data), timestamp, duration, [])

    def seek(self, timestamp):
        timestamp = max(0.0, min(timestamp, self._duration))
        position = int(timestamp / self._duration_per_frame)
        self._wave.setpos(position)


#########################################
#   Decoder class:
#########################################

class WaveDecoder(MediaDecoder):

    def get_file_extensions(self):
        return '.wav', '.wave', '.riff'

    def decode(self, file, filename, streaming=True):
        if streaming:
            return WaveSource(filename, file)
        else:
            return StaticSource(WaveSource(filename, file))


class WaveEncoder(MediaEncoder):

    def get_file_extensions(self):
        return '.wav', '.wave', '.riff'

    def encode(self, source, file, filename):
        """Save the Source to disk as a standard RIFF Wave.

        A standard RIFF wave header will be added to the raw PCM
        audio data when it is saved to disk.

        :Parameters:
            `filename` : str
                The file name to save as.

        """

        extension = filename.split('.')[-1].lower()
        if f".{extension}" not in self.get_file_extensions():
            raise MediaDecodeException("Invalid Format")

        source.seek(0)
        wave_writer = wave.open(file, mode='wb')
        wave_writer.setnchannels(source.audio_format.channels)
        wave_writer.setsampwidth(source.audio_format.sample_size // 8)
        wave_writer.setframerate(source.audio_format.sample_rate)

        # Save the data in 1-second chunks:
        chunksize = source.audio_format.bytes_per_second
        audiodata = source.get_audio_data(chunksize)
        while audiodata:
            wave_writer.writeframes(audiodata.data)
            audiodata = source.get_audio_data(chunksize)


def get_decoders():
    return [WaveDecoder()]


def get_encoders():
    return [WaveEncoder()]
