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

import io

import pyglet

from pyglet.media.exceptions import MediaException, CannotSeekException, MediaEncodeException


class AudioFormat:
    """Audio details.

    An instance of this class is provided by sources with audio tracks.  You
    should not modify the fields, as they are used internally to describe the
    format of data provided by the source.

    Args:
        channels (int): The number of channels: 1 for mono or 2 for stereo
            (pyglet does not yet support surround-sound sources).
        sample_size (int): Bits per sample; only 8 or 16 are supported.
        sample_rate (int): Samples per second (in Hertz).
    """

    def __init__(self, channels, sample_size, sample_rate):
        self.channels = channels
        self.sample_size = sample_size
        self.sample_rate = sample_rate

        # Convenience
        self.bytes_per_sample = (sample_size >> 3) * channels
        self.bytes_per_second = self.bytes_per_sample * sample_rate

    def __eq__(self, other):
        if other is None:
            return False
        return (self.channels == other.channels and
                self.sample_size == other.sample_size and
                self.sample_rate == other.sample_rate)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '%s(channels=%d, sample_size=%d, sample_rate=%d)' % (
            self.__class__.__name__, self.channels, self.sample_size,
            self.sample_rate)


class VideoFormat:
    """Video details.

    An instance of this class is provided by sources with a video stream. You
    should not modify the fields.

    Note that the sample aspect has no relation to the aspect ratio of the
    video image.  For example, a video image of 640x480 with sample aspect 2.0
    should be displayed at 1280x480.  It is the responsibility of the
    application to perform this scaling.

    Args:
        width (int): Width of video image, in pixels.
        height (int): Height of video image, in pixels.
        sample_aspect (float): Aspect ratio (width over height) of a single
            video pixel.
        frame_rate (float): Frame rate (frames per second) of the video.

            .. versionadded:: 1.2
    """

    def __init__(self, width, height, sample_aspect=1.0):
        self.width = width
        self.height = height
        self.sample_aspect = sample_aspect
        self.frame_rate = None

    def __eq__(self, other):
        if isinstance(other, VideoFormat):
            return (self.width == other.width and
                    self.height == other.height and
                    self.sample_aspect == other.sample_aspect and
                    self.frame_rate == other.frame_rate)
        return False


class AudioData:
    """A single packet of audio data.

    This class is used internally by pyglet.

    Args:
        data (str or ctypes array or pointer): Sample data.
        length (int): Size of sample data, in bytes.
        timestamp (float): Time of the first sample, in seconds.
        duration (float): Total data duration, in seconds.
        events (List[:class:`pyglet.media.events.MediaEvent`]): List of events
            contained within this packet. Events are timestamped relative to
            this audio packet.
    """

    __slots__ = 'data', 'length', 'timestamp', 'duration', 'events'

    def __init__(self, data, length, timestamp, duration, events):
        self.data = data
        self.length = length
        self.timestamp = timestamp
        self.duration = duration
        self.events = events

    def __eq__(self, other):
        if isinstance(other, AudioData):
            return (self.data == other.data and
                    self.length == other.length and
                    self.timestamp == other.timestamp and
                    self.duration == other.duration and
                    self.events == other.events)
        return False

    def consume(self, num_bytes, audio_format):
        """Remove some data from the beginning of the packet.

        All events are cleared.

        Args:
            num_bytes (int): The number of bytes to consume from the packet.
            audio_format (:class:`.AudioFormat`): The packet audio format.
        """
        self.events = ()
        if num_bytes >= self.length:
            self.data = None
            self.length = 0
            self.timestamp += self.duration
            self.duration = 0.
            return
        elif num_bytes == 0:
            return

        self.data = self.data[num_bytes:]
        self.length -= num_bytes
        self.duration -= num_bytes / audio_format.bytes_per_second
        self.timestamp += num_bytes / audio_format.bytes_per_second

    def get_string_data(self):
        """Return data as a bytestring.

        Returns:
            bytes: Data as a (byte)string.
        """
        if self.data is None:
            return b''

        return memoryview(self.data).tobytes()[:self.length]


class SourceInfo:
    """Source metadata information.

    Fields are the empty string or zero if the information is not available.

    Args:
        title (str): Title
        author (str): Author
        copyright (str): Copyright statement
        comment (str): Comment
        album (str): Album name
        year (int): Year
        track (int): Track number
        genre (str): Genre

    .. versionadded:: 1.2
    """

    title = ''
    author = ''
    copyright = ''
    comment = ''
    album = ''
    year = 0
    track = 0
    genre = ''


class Source:
    """An audio and/or video source.

    Args:
        audio_format (:class:`.AudioFormat`): Format of the audio in this
            source, or ``None`` if the source is silent.
        video_format (:class:`.VideoFormat`): Format of the video in this
            source, or ``None`` if there is no video.
        info (:class:`.SourceInfo`): Source metadata such as title, artist,
            etc; or ``None`` if the` information is not available.

            .. versionadded:: 1.2

    Attributes:
        is_player_source (bool): Determine if this source is a player
            current source.

            Check on a :py:class:`~pyglet.media.player.Player` if this source
            is the current source.
    """

    _duration = None
    _players = []  # List of players when calling Source.play

    audio_format = None
    video_format = None
    info = None
    is_player_source = False

    @property
    def duration(self):
        """float: The length of the source, in seconds.

        Not all source durations can be determined; in this case the value
        is ``None``.

        Read-only.
        """
        return self._duration

    def play(self):
        """Play the source.

        This is a convenience method which creates a Player for
        this source and plays it immediately.

        Returns:
            :class:`.Player`
        """
        from pyglet.media.player import Player  # XXX Nasty circular dependency
        player = Player()
        player.queue(self)
        player.play()
        Source._players.append(player)

        def _on_player_eos():
            Source._players.remove(player)
            # There is a closure on player. To get the refcount to 0,
            # we need to delete this function.
            player.on_player_eos = None

        player.on_player_eos = _on_player_eos
        return player

    def get_animation(self):
        """
        Import all video frames into memory.

        An empty animation will be returned if the source has no video.
        Otherwise, the animation will contain all unplayed video frames (the
        entire source, if it has not been queued on a player). After creating
        the animation, the source will be at EOS (end of stream).

        This method is unsuitable for videos running longer than a
        few seconds.

        .. versionadded:: 1.1

        Returns:
            :class:`pyglet.image.Animation`
        """
        from pyglet.image import Animation, AnimationFrame
        if not self.video_format:
            # XXX: This causes an assertion in the constructor of Animation
            return Animation([])
        else:
            frames = []
            last_ts = 0
            next_ts = self.get_next_video_timestamp()
            while next_ts is not None:
                image = self.get_next_video_frame()
                if image is not None:
                    delay = next_ts - last_ts
                    frames.append(AnimationFrame(image, delay))
                    last_ts = next_ts
                next_ts = self.get_next_video_timestamp()
            return Animation(frames)

    def get_next_video_timestamp(self):
        """Get the timestamp of the next video frame.

        .. versionadded:: 1.1

        Returns:
            float: The next timestamp, or ``None`` if there are no more video
            frames.
        """
        pass

    def get_next_video_frame(self):
        """Get the next video frame.

        .. versionadded:: 1.1

        Returns:
            :class:`pyglet.image.AbstractImage`: The next video frame image,
            or ``None`` if the video frame could not be decoded or there are
            no more video frames.
        """
        pass

    def save(self, filename, file=None, encoder=None):
        """Save this Source to a file.

        :Parameters:
            `filename` : str
                Used to set the file format, and to open the output file
                if `file` is unspecified.
            `file` : file-like object or None
                File to write audio data to.
            `encoder` : MediaEncoder or None
                If unspecified, all encoders matching the filename extension
                are tried.  If all fail, the exception from the first one
                attempted is raised.

        """
        if not file:
            file = open(filename, 'wb')

        if encoder:
            encoder.encode(self, file, filename)
        else:
            first_exception = None
            for encoder in pyglet.media.get_encoders(filename):

                try:
                    encoder.encode(self, file, filename)
                    return
                except MediaEncodeException as e:
                    first_exception = first_exception or e
                    file.seek(0)

            if not first_exception:
                raise MediaEncodeException(f"No Encoders are available for this extension: '{filename}'")
            raise first_exception

        file.close()

    # Internal methods that Player calls on the source:

    def seek(self, timestamp):
        """Seek to given timestamp.

        Args:
            timestamp (float): Time where to seek in the source. The
                ``timestamp`` will be clamped to the duration of the source.
        """
        raise CannotSeekException()

    def get_queue_source(self):
        """Return the ``Source`` to be used as the queue source for a player.

        Default implementation returns self.
        """
        return self

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        """Get next packet of audio data.

        Args:
            num_bytes (int): Maximum number of bytes of data to return.
            compensation_time (float): Time in sec to compensate due to a
                difference between the master clock and the audio clock.

        Returns:
            :class:`.AudioData`: Next packet of audio data, or ``None`` if
            there is no (more) data.
        """
        return None


class StreamingSource(Source):
    """A source that is decoded as it is being played.

    The source can only be played once at a time on any
    :class:`~pyglet.media.player.Player`.
    """

    def get_queue_source(self):
        """Return the ``Source`` to be used as the source for a player.

        Default implementation returns self.

        Returns:
            :class:`.Source`
        """
        if self.is_player_source:
            raise MediaException('This source is already queued on a player.')
        self.is_player_source = True
        return self

    def delete(self):
        """Release the resources held by this StreamingSource."""
        pass


class StaticSource(Source):
    """A source that has been completely decoded in memory.

    This source can be queued onto multiple players any number of times.

    Construct a :py:class:`~pyglet.media.StaticSource` for the data in
    ``source``.

    Args:
        source (Source):  The source to read and decode audio and video data
            from.
    """

    def __init__(self, source):
        source = source.get_queue_source()
        if source.video_format:
            raise NotImplementedError('Static sources not supported for video.')

        self.audio_format = source.audio_format
        if not self.audio_format:
            self._data = None
            self._duration = 0.
            return

        # Arbitrary: number of bytes to request at a time.
        buffer_size = 1 << 20  # 1 MB

        # Naive implementation.  Driver-specific implementations may override
        # to load static audio data into device (or at least driver) memory.
        data = io.BytesIO()
        while True:
            audio_data = source.get_audio_data(buffer_size)
            if not audio_data:
                break
            data.write(audio_data.get_string_data())
        self._data = data.getvalue()

        self._duration = len(self._data) / self.audio_format.bytes_per_second

    def get_queue_source(self):
        if self._data is not None:
            return StaticMemorySource(self._data, self.audio_format)

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        """The StaticSource does not provide audio data.

        When the StaticSource is queued on a
        :class:`~pyglet.media.player.Player`, it creates a
        :class:`.StaticMemorySource` containing its internal audio data and
        audio format.

        Raises:
            RuntimeError
        """
        raise RuntimeError('StaticSource cannot be queued.')


class StaticMemorySource(StaticSource):
    """
    Helper class for default implementation of :class:`.StaticSource`.

    Do not use directly. This class is used internally by pyglet.

    Args:
        data (AudioData): The audio data.
        audio_format (AudioFormat): The audio format.
    """

    def __init__(self, data, audio_format):
        """Construct a memory source over the given data buffer."""
        self._file = io.BytesIO(data)
        self._max_offset = len(data)
        self.audio_format = audio_format
        self._duration = len(data) / float(audio_format.bytes_per_second)

    def seek(self, timestamp):
        """Seek to given timestamp.

        Args:
            timestamp (float): Time where to seek in the source.
        """
        offset = int(timestamp * self.audio_format.bytes_per_second)

        # Align to sample
        if self.audio_format.bytes_per_sample == 2:
            offset &= 0xfffffffe
        elif self.audio_format.bytes_per_sample == 4:
            offset &= 0xfffffffc

        self._file.seek(offset)

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        """Get next packet of audio data.

        Args:
            num_bytes (int): Maximum number of bytes of data to return.
            compensation_time (float): Not used in this class.

        Returns:
            :class:`.AudioData`: Next packet of audio data, or ``None`` if
            there is no (more) data.
        """
        offset = self._file.tell()
        timestamp = float(offset) / self.audio_format.bytes_per_second

        # Align to sample size
        if self.audio_format.bytes_per_sample == 2:
            num_bytes &= 0xfffffffe
        elif self.audio_format.bytes_per_sample == 4:
            num_bytes &= 0xfffffffc

        data = self._file.read(num_bytes)
        if not len(data):
            return None

        duration = float(len(data)) / self.audio_format.bytes_per_second
        return AudioData(data, len(data), timestamp, duration, [])


class SourceGroup:
    """Group of like sources to allow gapless playback.

    Seamlessly read data from a group of sources to allow for
    gapless playback. All sources must share the same audio format.
    The first source added sets the format.
    """

    def __init__(self):
        self.audio_format = None
        self.video_format = None
        self.duration = 0.0
        self._timestamp_offset = 0.0
        self._dequeued_durations = []
        self._sources = []

    def seek(self, time):
        if self._sources:
            self._sources[0].seek(time)

    def add(self, source):
        self.audio_format = self.audio_format or source.audio_format
        source = source.get_queue_source()
        assert (source.audio_format == self.audio_format), "Sources must share the same audio format."
        self._sources.append(source)
        self.duration += source.duration

    def has_next(self):
        return len(self._sources) > 1

    def get_queue_source(self):
        return self

    def _advance(self):
        if self._sources:
            self._timestamp_offset += self._sources[0].duration
            self._dequeued_durations.insert(0, self._sources[0].duration)
            old_source = self._sources.pop(0)
            self.duration -= old_source.duration

            if isinstance(old_source, StreamingSource):
                old_source.delete()
                del old_source

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        """Get next audio packet.

        :Parameters:
            `num_bytes` : int
                Hint for preferred size of audio packet; may be ignored.

        :rtype: `AudioData`
        :return: Audio data, or None if there is no more data.
        """

        if not self._sources:
            return None

        buffer = b""
        duration = 0.0
        timestamp = 0.0

        while len(buffer) < num_bytes and self._sources:
            audiodata = self._sources[0].get_audio_data(num_bytes)
            if audiodata:
                buffer += audiodata.data
                duration += audiodata.duration
                timestamp += self._timestamp_offset
            else:
                self._advance()

        return AudioData(buffer, len(buffer), timestamp, duration, [])
