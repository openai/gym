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

import math

import pyglet
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.events import MediaEvent
from pyglet.util import debug_print
from . import interface

_debug = debug_print('debug_media')


def _convert_coordinates(coordinates):
    x, y, z = coordinates
    return x, y, -z


class XAudio2AudioPlayer(AbstractAudioPlayer):
    # Need to cache these because pyglet API allows update separately, but
    # DSound requires both to be set at once.
    _cone_inner_angle = 360
    _cone_outer_angle = 360

    min_buffer_size = 9600

    max_buffer_count = 3  # Max in queue at once, increasing may impact performance depending on buffer size.

    def __init__(self, driver, xa2_driver, source, player):
        super(XAudio2AudioPlayer, self).__init__(source, player)
        # We keep here a strong reference because the AudioDriver is anyway
        # a singleton object which will only be deleted when the application
        # shuts down. The AudioDriver does not keep a ref to the AudioPlayer.
        self.driver = driver
        self._xa2_driver = xa2_driver

        # If cleared, we need to check when it's done clearing.
        self._flushing = False

        # If deleted, we need to make sure it's done deleting.
        self._deleted = False

        # Desired play state (may be actually paused due to underrun -- not
        # implemented yet).
        self._playing = False

        # Theoretical write and play cursors for an infinite buffer.  play
        # cursor is always <= write cursor (when equal, underrun is
        # happening).
        self._write_cursor = 0
        self._play_cursor = 0

        # List of (play_cursor, MediaEvent), in sort order
        self._events = []

        # List of (cursor, timestamp), in sort order (cursor gives expiry
        # place of the timestamp)
        self._timestamps = []

        # This will be True if the last buffer has already been submitted.
        self.buffer_end_submitted = False

        self._buffers = []  # Current buffers in queue waiting to be played.

        self._xa2_source_voice = self._xa2_driver.get_source_voice(source, self)

        self._buffer_size = int(source.audio_format.sample_rate * 2)

    def on_driver_destroy(self):
        self.stop()
        self._xa2_source_voice = None

    def on_driver_reset(self):
        self._xa2_source_voice = self._xa2_driver.get_source_voice(self.source, self)

        # Queue up any buffers that are still in queue but weren't deleted. This does not pickup where the last sample
        # played, only where the last buffer was submitted. As such it's possible for audio to be replayed if buffer is
        # large enough.
        for cx2_buffer in self._buffers:
            self._xa2_source_voice.submit_buffer(cx2_buffer)

    def __del__(self):
        if self._xa2_source_voice:
            self._xa2_source_voice = None

    def delete(self):
        """Called from Player. Docs says to cleanup resources, but other drivers wait for GC to do it?"""
        if self._xa2_source_voice:
            self._deleted = True

    def play(self):
        assert _debug('XAudio2 play')

        if not self._playing:
            self._playing = True
            if not self._flushing:
                self._xa2_source_voice.play()

        assert _debug('return XAudio2 play')

    def stop(self):
        assert _debug('XAudio2 stop')

        if self._playing:
            self._playing = False
            self.buffer_end_submitted = False
            self._xa2_source_voice.stop()

        assert _debug('return XAudio2 stop')

    def clear(self):
        assert _debug('XAudio2 clear')
        super(XAudio2AudioPlayer, self).clear()
        self._play_cursor = 0
        self._write_cursor = 0
        self.buffer_end_submitted = False
        self._deleted = False

        if self._buffers:
            self._flushing = True

        self._xa2_source_voice.flush()
        self._buffers.clear()
        del self._events[:]
        del self._timestamps[:]

    def _restart(self, dt):
        """Prefill audio and attempt to replay audio."""
        if self._playing and self._xa2_source_voice:
            self.refill_source_player()
            self._xa2_source_voice.play()

    def refill_source_player(self):
        """Obtains audio data from the source, puts it into a buffer to submit to the voice.
        Unlike the other drivers this does not carve pieces of audio from the buffer and slowly
        consume it. This submits the buffer retrieved from the decoder in it's entirety.
        """

        buffers_queued = self._xa2_source_voice.buffers_queued

        # Free any buffers that have ended.
        while len(self._buffers) > buffers_queued:
            # Clean out any buffers that have played.
            buffer = self._buffers.pop(0)
            self._play_cursor += buffer.AudioBytes
            del buffer  # Does this remove AudioData within the buffer? Let GC remove or explicit remove?

        # We have to wait for all of the buffers we are flushing to end before we restart next buffer.
        # When voice reaches 0 buffers, it is available for re-use.
        if self._flushing:
            if buffers_queued == 0:
                self._flushing = False

                # This is required because the next call to play will come before all flushes are done.
                # Restart at next available opportunity.
                pyglet.clock.schedule_once(self._restart, 0)
            return

        if self._deleted:
            if buffers_queued == 0:
                self._deleted = False
                self._xa2_driver.return_voice(self._xa2_source_voice)
            return

        # Wait for the playback to hit 0 buffers before we eos.
        if self.buffer_end_submitted:
            if buffers_queued == 0:
                self._xa2_source_voice.stop()
                MediaEvent(0, "on_eos")._sync_dispatch_to_player(self.player)
        else:
            current_buffers = []
            while buffers_queued < self.max_buffer_count:
                audio_data = self.source.get_audio_data(self._buffer_size, 0.0)
                if audio_data:
                    assert _debug(
                        'Xaudio2: audio data - length: {}, duration: {}, buffer size: {}'.format(audio_data.length,
                                                                                                 audio_data.duration,
                                                                                                 self._buffer_size))

                    if audio_data.length == 0:  # Sometimes audio data has 0 length at the front?
                        continue

                    x2_buffer = self._xa2_driver.create_buffer(audio_data)

                    current_buffers.append(x2_buffer)

                    self._write_cursor += x2_buffer.AudioBytes  # We've pushed this many bytes into the source player.

                    self._add_audiodata_events(audio_data)
                    self._add_audiodata_timestamp(audio_data)

                    buffers_queued += 1
                else:
                    # End of audio data, set last packet as end.
                    self.buffer_end_submitted = True
                    break

            # We submit the buffers here, just in-case the end of stream was found.
            for cx2_buffer in current_buffers:
                self._xa2_source_voice.submit_buffer(cx2_buffer)

            # Store buffers temporarily, otherwise they get GC'd.
            self._buffers.extend(current_buffers)

        self._dispatch_pending_events()

    def _dispatch_new_event(self, event_name):
        MediaEvent(0, event_name)._sync_dispatch_to_player(self.player)

    def _add_audiodata_events(self, audio_data):
        for event in audio_data.events:
            event_cursor = self._write_cursor + event.timestamp * \
                           self.source.audio_format.bytes_per_second
            assert _debug('Adding event', event, 'at', event_cursor)
            self._events.append((event_cursor, event))

    def _add_audiodata_timestamp(self, audio_data):
        ts_cursor = self._write_cursor + audio_data.length
        self._timestamps.append(
            (ts_cursor, audio_data.timestamp + audio_data.duration))

    def _dispatch_pending_events(self):
        pending_events = []
        while self._events and self._events[0][0] <= self._play_cursor:
            _, event = self._events.pop(0)
            pending_events.append(event)

        assert _debug('Dispatching pending events: {}'.format(pending_events))
        assert _debug('Remaining events: {}'.format(self._events))

        for event in pending_events:
            event._sync_dispatch_to_player(self.player)

    def _cleanup_timestamps(self):
        while self._timestamps and self._timestamps[0][0] < self._play_cursor:
            del self._timestamps[0]

    def get_time(self):
        self.update_play_cursor()
        if self._timestamps:
            cursor, ts = self._timestamps[0]
            result = ts + (self._play_cursor - cursor) / float(self.source.audio_format.bytes_per_second)
        else:
            result = None

        return result

    def set_volume(self, volume):
        self._xa2_source_voice.volume = volume

    def set_position(self, position):
        if self._xa2_source_voice.is_emitter:
            self._xa2_source_voice.position = _convert_coordinates(position)

    def set_min_distance(self, min_distance):
        """Not a true min distance, but similar effect. Changes CurveDistanceScaler default is 1."""
        if self._xa2_source_voice.is_emitter:
            self._xa2_source_voice.distance_scaler = min_distance

    def set_max_distance(self, max_distance):
        """No such thing built into xaudio2"""
        return

    def set_pitch(self, pitch):
        self._xa2_source_voice.frequency = pitch

    def set_cone_orientation(self, cone_orientation):
        if self._xa2_source_voice.is_emitter:
            self._xa2_source_voice.cone_orientation = _convert_coordinates(cone_orientation)

    def set_cone_inner_angle(self, cone_inner_angle):
        if self._xa2_source_voice.is_emitter:
            self._cone_inner_angle = int(cone_inner_angle)
            self._set_cone_angles()

    def set_cone_outer_angle(self, cone_outer_angle):
        if self._xa2_source_voice.is_emitter:
            self._cone_outer_angle = int(cone_outer_angle)
            self._set_cone_angles()

    def _set_cone_angles(self):
        inner = min(self._cone_inner_angle, self._cone_outer_angle)
        outer = max(self._cone_inner_angle, self._cone_outer_angle)
        self._xa2_source_voice.set_cone_angles(math.radians(inner), math.radians(outer))

    def set_cone_outer_gain(self, cone_outer_gain):
        if self._xa2_source_voice.is_emitter:
            self._xa2_source_voice.cone_outside_volume = cone_outer_gain

    def prefill_audio(self):
        # Cannot refill during a flush. Schedule will handle it.
        if not self._flushing:
            self.refill_source_player()


class XAudio2Driver(AbstractAudioDriver):
    def __init__(self):
        self._xa2_driver = interface.XAudio2Driver()
        self._xa2_listener = self._xa2_driver.create_listener()

        assert self._xa2_driver is not None
        assert self._xa2_listener is not None

    def __del__(self):
        self.delete()

    def get_performance(self):
        assert self._xa2_driver is not None
        return self._xa2_driver.get_performance()

    def create_audio_player(self, source, player):
        assert self._xa2_driver is not None
        return XAudio2AudioPlayer(self, self._xa2_driver, source, player)

    def get_listener(self):
        assert self._xa2_driver is not None
        assert self._xa2_listener is not None
        return XAudio2Listener(self._xa2_listener, self._xa2_driver)

    def delete(self):
        self._xa2_listener = None


class XAudio2Listener(AbstractListener):
    def __init__(self, xa2_listener, xa2_driver):
        self._xa2_listener = xa2_listener
        self._xa2_driver = xa2_driver

    def _set_volume(self, volume):
        self._volume = volume
        self._xa2_driver.volume = volume

    def _set_position(self, position):
        self._position = position
        self._xa2_listener.position = _convert_coordinates(position)

    def _set_forward_orientation(self, orientation):
        self._forward_orientation = orientation
        self._set_orientation()

    def _set_up_orientation(self, orientation):
        self._up_orientation = orientation
        self._set_orientation()

    def _set_orientation(self):
        self._xa2_listener.orientation = _convert_coordinates(self._forward_orientation) + _convert_coordinates(
            self._up_orientation)
