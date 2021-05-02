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

import weakref

from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer
from pyglet.media.events import MediaEvent
from pyglet.media.exceptions import MediaException
from pyglet.media.drivers.listener import AbstractListener
from pyglet.debug import debug_print

from . import lib_pulseaudio as pa
from .interface import PulseAudioContext, PulseAudioContext, PulseAudioMainLoop, PulseAudioStream


_debug = debug_print('debug_media')


class PulseAudioDriver(AbstractAudioDriver):
    def __init__(self):
        self.mainloop = PulseAudioMainLoop()
        self.mainloop.start()
        self.lock = self.mainloop
        self.context = None

        self._players = weakref.WeakSet()
        self._listener = PulseAudioListener(self)

    def __del__(self):
        self.delete()

    def create_audio_player(self, source, player):
        assert self.context is not None
        player = PulseAudioPlayer(source, player, self)
        self._players.add(player)
        return player

    def connect(self, server=None):
        """Connect to pulseaudio server.

        :Parameters:
            `server` : str
                Server to connect to, or ``None`` for the default local
                server (which may be spawned as a daemon if no server is
                found).
        """
        # TODO disconnect from old
        assert not self.context, 'Already connected'

        self.context = self.mainloop.create_context()
        self.context.connect(server)

    def dump_debug_info(self):
        print('Client version: ', pa.pa_get_library_version())

        print('Server:         ', self.context.server)
        print('Protocol:       ', self.context.protocol_version)
        print('Server protocol:', self.context.server_protocol_version)
        print('Local context:  ', self.context.is_local and 'Yes' or 'No')

    def delete(self):
        """Completely shut down pulseaudio client."""
        if self.mainloop is not None:

            with self.mainloop:
                if self.context is not None:
                    self.context.delete()
                    self.context = None

        if self.mainloop is not None:
            self.mainloop.delete()
            self.mainloop = None
            self.lock = None

    def get_listener(self):
        return self._listener


class PulseAudioListener(AbstractListener):
    def __init__(self, driver):
        self.driver = weakref.proxy(driver)

    def _set_volume(self, volume):
        self._volume = volume
        for player in self.driver._players:
            player.set_volume(player._volume)

    def _set_position(self, position):
        self._position = position

    def _set_forward_orientation(self, orientation):
        self._forward_orientation = orientation

    def _set_up_orientation(self, orientation):
        self._up_orientation = orientation


class PulseAudioPlayer(AbstractAudioPlayer):
    _volume = 1.0

    def __init__(self, source, player, driver):
        super(PulseAudioPlayer, self).__init__(source, player)
        self.driver = weakref.ref(driver)

        self._events = []
        self._timestamps = []  # List of (ref_time, timestamp)
        self._write_index = 0  # Current write index (tracked manually)
        self._read_index_valid = False # True only if buffer has non-stale data

        self._clear_write = False
        self._buffered_audio_data = None
        self._playing = False

        self._current_audio_data = None

        self._time_sync_operation = None

        audio_format = source.audio_format
        assert audio_format

        with driver.mainloop:
            self.stream = driver.context.create_stream(audio_format)
            self.stream.push_handlers(self)
            self.stream.connect_playback()
            assert self.stream.is_ready

        assert _debug('PulseAudioPlayer: __init__ finished')

    def on_write_needed(self, nbytes, underflow):
        if underflow:
            self._handle_underflow()
        else:
            self._write_to_stream(nbytes)

        # Asynchronously update time
        if self._events:
            if self._time_sync_operation is not None and self._time_sync_operation.is_done:
                self._time_sync_operation.delete()
                self._time_sync_operation = None
            if self._time_sync_operation is None:
                assert _debug('PulseAudioPlayer: trigger timing info update')
                self._time_sync_operation = self.stream.update_timing_info(self._process_events)

    def _get_audio_data(self, nbytes=None):
        if self._current_audio_data is None and self.source is not None:
            # Always try to buffer at least 1 second of audio data
            min_bytes = 1 * self.source.audio_format.bytes_per_second
            if nbytes is None:
                nbytes = min_bytes
            else:
                nbytes = min(min_bytes, nbytes)
            assert _debug('PulseAudioPlayer: Try to get {} bytes of audio data'.format(nbytes))
            compensation_time = self.get_audio_time_diff()
            self._current_audio_data = self.source.get_audio_data(nbytes, compensation_time)
            self._schedule_events()
        if self._current_audio_data is None:
            assert _debug('PulseAudioPlayer: No audio data available')
        else:
            assert _debug('PulseAudioPlayer: Got {} bytes of audio data'.format(
                           self._current_audio_data.length))
        return self._current_audio_data

    def _has_audio_data(self):
        return self._get_audio_data() is not None

    def _consume_audio_data(self, nbytes):
        if self._current_audio_data is not None:
            if nbytes == self._current_audio_data.length:
                self._current_audio_data = None
            else:
                self._current_audio_data.consume(nbytes, self.source.audio_format)

    def _schedule_events(self):
        if self._current_audio_data is not None:
            for event in self._current_audio_data.events:
                event_index = self._write_index + event.timestamp * \
                    self.source.audio_format.bytes_per_second
                assert _debug('PulseAudioPlayer: Schedule event at index {}'.format(event_index))
                self._events.append((event_index, event))

    def _write_to_stream(self, nbytes=None):
        if nbytes is None:
            nbytes = self.stream.writable_size
        assert _debug('PulseAudioPlayer: Requested to write %d bytes to stream' % nbytes)

        seek_mode = pa.PA_SEEK_RELATIVE
        if self._clear_write:
            # When seeking, the stream.writable_size will be 0.
            # So we force at least 4096 bytes to overwrite the Buffer
            # starting at read index
            nbytes = max(4096, nbytes)
            seek_mode = pa.PA_SEEK_RELATIVE_ON_READ
            self._clear_write = False
            assert _debug('PulseAudioPlayer: Clear buffer')

        while self._has_audio_data() and nbytes > 0:
            audio_data = self._get_audio_data()

            write_length = min(nbytes, audio_data.length)
            consumption = self.stream.write(audio_data, write_length, seek_mode)

            seek_mode = pa.PA_SEEK_RELATIVE
            self._read_index_valid = True
            self._timestamps.append((self._write_index, audio_data.timestamp))
            self._write_index += consumption

            assert _debug('PulseAudioPlayer: Actually wrote {} bytes '
                          'to stream'.format(consumption))
            self._consume_audio_data(consumption)

            nbytes -= consumption

        if not self._has_audio_data():
            # In case the source group wasn't long enough to prebuffer stream
            # to PA's satisfaction, trigger immediate playback (has no effect
            # if stream is already playing).
            if self._playing:
                op = self.stream.trigger()
                op.delete()  # Explicit delete to prevent locking

    def _handle_underflow(self):
        assert _debug('Player: underflow')
        if self._has_audio_data():
            self._write_to_stream()
        else:
            self._add_event_at_write_index('on_eos')

    def _process_events(self):
        assert _debug('PulseAudioPlayer: Process events')
        if not self._events:
            assert _debug('PulseAudioPlayer: No events')
            return

        # Assume this is called after time sync
        timing_info = self.stream.get_timing_info()
        if not timing_info:
            assert _debug('PulseAudioPlayer: No timing info to process events')
            return

        read_index = timing_info.read_index
        assert _debug('PulseAudioPlayer: Dispatch events at index {}'.format(read_index))

        while self._events and self._events[0][0] <= read_index:
            _, event = self._events.pop(0)
            assert _debug('PulseAudioPlayer: Dispatch event', event)
            event._sync_dispatch_to_player(self.player)

    def _add_event_at_write_index(self, event_name):
        assert _debug('PulseAudioPlayer: Add event at index {}'.format(self._write_index))
        self._events.append((self._write_index, MediaEvent(0., event_name)))

    def delete(self):
        assert _debug('Delete PulseAudioPlayer')

        self.stream.pop_handlers()
        driver = self.driver()
        if driver is None:
            assert _debug('PulseAudioDriver has been garbage collected.')
            self.stream = None
            return

        if driver.mainloop is None:
            assert _debug('PulseAudioDriver already deleted. '
                      'PulseAudioPlayer could not clean up properly.')
            return

        if self._time_sync_operation is not None:
            with self._time_sync_operation:
                self._time_sync_operation.delete()
            self._time_sync_operation = None

        self.stream.delete()
        self.stream = None

    def clear(self):
        assert _debug('PulseAudioPlayer.clear')
        super(PulseAudioPlayer, self).clear()
        self._clear_write = True
        self._write_index = self._get_read_index()
        self._timestamps = []
        self._events = []

        with self.stream:
            self._read_index_valid = False
            self.stream.prebuf().wait()

    def play(self):
        assert _debug('PulseAudioPlayer.play')

        with self.stream:
            if self.stream.is_corked:
                self.stream.resume().wait().delete()
                assert _debug('PulseAudioPlayer: Resumed playback')
            if self.stream.underflow:
                self._write_to_stream()
            if not self._has_audio_data():
                self.stream.trigger().wait().delete()
                assert _debug('PulseAudioPlayer: Triggered stream for immediate playback')
            assert not self.stream.is_corked

        self._playing = True

    def stop(self):
        assert _debug('PulseAudioPlayer.stop')

        with self.stream:
            if not self.stream.is_corked:
                self.stream.pause().wait().delete()

        self._playing = False

    def _get_read_index(self):
        with self.stream:
            self.stream.update_timing_info().wait().delete()

        timing_info = self.stream.get_timing_info()
        if timing_info:
            read_index = timing_info.read_index
        else:
            read_index = 0

        assert _debug('_get_read_index ->', read_index)
        return read_index

    def _get_write_index(self):
        timing_info = self.stream.get_timing_info()
        if timing_info:
            write_index = timing_info.write_index
        else:
            write_index = 0

        assert _debug('_get_write_index ->', write_index)
        return write_index

    def _get_timing_info(self):
        with self.stream:
            self.stream.update_timing_info().wait().delete()

        timing_info = self.stream.get_timing_info()
        return timing_info

    def get_time(self):
        if not self._read_index_valid:
            assert _debug('get_time <_read_index_valid = False> -> 0')
            return 0

        t_info = self._get_timing_info()
        read_index = t_info.read_index
        transport_usec = t_info.transport_usec
        sink_usec = t_info.sink_usec

        write_index = 0
        timestamp = 0.0

        try:
            write_index, timestamp = self._timestamps[0]
            write_index, timestamp = self._timestamps[1]
            while read_index >= write_index:
                del self._timestamps[0]
                write_index, timestamp = self._timestamps[1]
        except IndexError:
            pass

        bytes_per_second = self.source.audio_format.bytes_per_second
        dt = (read_index - write_index) / float(bytes_per_second) * 1000000
        # We add 2x the transport time because we didn't take it into account
        # when we wrote the write index the first time. See _write_to_stream
        dt += t_info.transport_usec * 2
        dt -= t_info.sink_usec
        # We convert back to seconds
        dt /= 1000000
        time = timestamp + dt

        assert _debug('get_time ->', time)
        return time

    def set_volume(self, volume):
        self._volume = volume

        if self.stream:
            driver = self.driver()
            volume *= driver._listener._volume
            with driver.context:
                driver.context.set_input_volume(self.stream, volume).wait()

    def set_pitch(self, pitch):
        sample_rate = self.stream.audio_format.rate
        with self.stream:
            self.stream.update_sample_rate(int(pitch * sample_rate)).wait()

    def prefill_audio(self):
        self._write_to_stream(nbytes=None)
