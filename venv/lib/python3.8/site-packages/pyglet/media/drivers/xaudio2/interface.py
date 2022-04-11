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
import weakref
from collections import namedtuple, defaultdict

import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib

_debug = debug_print('debug_media')


class XAudio2Driver:
    # Specifies if positional audio should be used. Can be enabled later, but not disabled.
    allow_3d = True

    # Which processor to use. (#1 by default)
    processor = lib.XAUDIO2_DEFAULT_PROCESSOR

    # Which stream classification Windows uses on this driver.
    category = lib.AudioCategory_GameEffects

    # If the driver errors or disappears, it will attempt to restart the engine.
    restart_on_error = True

    # Max Frequency a voice can have. Setting this higher/lower will increase/decrease memory allocation.
    max_frequency_ratio = 2.0

    def __init__(self):
        """Creates an XAudio2 master voice and sets up 3D audio if specified. This attaches to the default audio
        device and will create a virtual audio endpoint that changes with the system. It will not recover if a
        critical error is encountered such as no more audio devices are present.
        """
        assert _debug('Constructing XAudio2Driver')
        self._listener = None
        self._xaudio2 = None
        self._dead = False

        self._emitting_voices = []  # Contains all of the emitting source voices.
        self._voice_pool = defaultdict(list)
        self._in_use = []  # All voices currently in use.

        self._players = []  # Only used for resetting/restoring xaudio2. Store players to callback.

        self._create_xa2()

        if self.restart_on_error:
            audio_devices = get_audio_device_manager()
            if audio_devices:
                assert _debug('Audio device instance found.')
                audio_devices.push_handlers(self)

                if audio_devices.get_default_output() is None:
                    raise ImportError("No default audio device found, can not create driver.")

                pyglet.clock.schedule_interval_soft(self._check_state, 0.5)

    def _check_state(self, dt):
        """Hack/workaround, you cannot shutdown/create XA2 within a COM callback, set a schedule to check state."""
        if self._dead is True:
            if self._xaudio2:
                self._shutdown_xaudio2()
        else:
            if not self._xaudio2:
                self._create_xa2()
                # Notify all active it's reset.
                for player in self._players:
                    player.dispatch_event('on_driver_reset')

                self._players.clear()

    def on_default_changed(self, device):
        """Callback derived from the Audio Devices to help us determine when the system no longer has output."""
        if device is None:
            assert _debug('Error: Default audio device was removed or went missing.')
            self._dead = True
        else:
            if self._dead:
                assert _debug('Warning: Default audio device added after going missing.')
                self._dead = False

    def _create_xa2(self, device_id=None):
        self._xaudio2 = lib.IXAudio2()

        try:
            lib.XAudio2Create(ctypes.byref(self._xaudio2), 0, self.processor)
        except OSError:
            raise ImportError("XAudio2 driver could not be initialized.")

        if _debug:
            # Debug messages are found in Windows Event Viewer, you must enable event logging:
            # Applications and Services -> Microsoft -> Windows -> Xaudio2 -> Debug Logging.
            # Right click -> Enable Logs
            debug = lib.XAUDIO2_DEBUG_CONFIGURATION()
            debug.LogThreadID = True
            debug.TraceMask = lib.XAUDIO2_LOG_ERRORS | lib.XAUDIO2_LOG_WARNINGS
            debug.BreakMask = lib.XAUDIO2_LOG_WARNINGS

            self._xaudio2.SetDebugConfiguration(ctypes.byref(debug), None)

        self._master_voice = lib.IXAudio2MasteringVoice()
        self._xaudio2.CreateMasteringVoice(byref(self._master_voice),
                                           lib.XAUDIO2_DEFAULT_CHANNELS,
                                           lib.XAUDIO2_DEFAULT_SAMPLERATE,
                                           0, device_id, None, self.category)

        if self.allow_3d:
            self.enable_3d()

    @property
    def active_voices(self):
        return self._in_use

    @property
    def pooled_voices(self):
        return [voice for voices in self._voice_pool.values() for voice in voices]

    @property
    def all_voices(self):
        """All pooled and active voices."""
        return self.active_voices + self.all_voices

    def clear_pool(self):
        """Destroy and then clear the pool of voices"""
        for voice in self.pooled_voices:
            voice.destroy()

        for voice_key in self._voice_pool:
            self._voice_pool[voice_key].clear()

    def clear_active(self):
        """Destroy and then clear all active voices"""
        for voice in self._in_use:
            voice.destroy()

        self._in_use.clear()

    def set_device(self, device):
        """Attach XA2 with a specific device rather than the virtual device."""
        self._shutdown_xaudio2()
        self._create_xa2(device.id)

        # Notify all active players it's reset..
        for player in self._players:
            player.dispatch_event('on_driver_reset')

        self._players.clear()

    def _shutdown_xaudio2(self):
        """Stops and destroys all active voices, then destroys XA2 instance."""
        for voice in self.active_voices:
            voice.player.on_driver_destroy()
            self._players.append(voice.player.player)

        self._delete_driver()

    def _delete_driver(self):
        if self._xaudio2:
            # Stop 3d
            if self.allow_3d:
                pyglet.clock.unschedule(self._calculate_3d_sources)

            # Destroy all pooled voices as master will change.
            self.clear_pool()
            self.clear_active()

            self._xaudio2.StopEngine()
            self._xaudio2.Release()
            self._xaudio2 = None

    def enable_3d(self):
        """Initializes the prerequisites for 3D positional audio and initializes with default DSP settings."""
        channel_mask = DWORD()
        self._master_voice.GetChannelMask(byref(channel_mask))

        self._x3d_handle = lib.X3DAUDIO_HANDLE()
        lib.X3DAudioInitialize(channel_mask.value, lib.X3DAUDIO_SPEED_OF_SOUND, self._x3d_handle)

        self._mvoice_details = lib.XAUDIO2_VOICE_DETAILS()
        self._master_voice.GetVoiceDetails(byref(self._mvoice_details))

        matrix = (FLOAT * self._mvoice_details.InputChannels)()
        self._dsp_settings = lib.X3DAUDIO_DSP_SETTINGS()
        self._dsp_settings.SrcChannelCount = 1
        self._dsp_settings.DstChannelCount = self._mvoice_details.InputChannels
        self._dsp_settings.pMatrixCoefficients = matrix

        pyglet.clock.schedule_interval_soft(self._calculate_3d_sources, 1 / 15.0)

    @property
    def volume(self):
        vol = c_float()
        self._master_voice.GetVolume(ctypes.byref(vol))
        return vol.value

    @volume.setter
    def volume(self, value):
        """Sets global volume of the master voice."""
        self._master_voice.SetVolume(value, 0)

    def _calculate_3d_sources(self, dt):
        """We calculate the 3d emitters and sources every 15 fps, committing everything after deferring all changes."""
        for source_voice in self._emitting_voices:
            self.apply3d(source_voice)

        self._xaudio2.CommitChanges(0)

    def _calculate3d(self, listener, emitter):
        lib.X3DAudioCalculate(
            self._x3d_handle,
            listener,
            emitter,
            lib.default_dsp_calculation,
            self._dsp_settings
        )

    def _apply3d(self, voice, commit):
        """Calculates the output channels based on the listener and emitter and default DSP settings.
           Commit determines if the settings are applied immediately (0) or committed at once through the xaudio driver.
        """
        voice.SetOutputMatrix(self._master_voice,
                              1,
                              self._mvoice_details.InputChannels,
                              self._dsp_settings.pMatrixCoefficients,
                              commit)

        voice.SetFrequencyRatio(self._dsp_settings.DopplerFactor, commit)

    def apply3d(self, source_voice, commit=1):
        self._calculate3d(self._listener.listener, source_voice._emitter)
        self._apply3d(source_voice._voice, commit)

    def __del__(self):
        try:
            self._delete_driver()
            pyglet.clock.unschedule(self._check_state)
        except AttributeError:
            # Usually gets unloaded by default on app exit, but be safe.
            pass

    def get_performance(self):
        """Retrieve some basic XAudio2 performance data such as memory usage and source counts."""
        pf = lib.XAUDIO2_PERFORMANCE_DATA()
        self._xaudio2.GetPerformanceData(ctypes.byref(pf))
        return pf

    def create_listener(self):
        assert self._listener is None, "You can only create one listener."
        self._listener = XAudio2Listener(self)
        return self._listener

    def get_source_voice(self, source, player):
        """ Get a source voice from the pool. Source voice creation can be slow to create/destroy. So pooling is
            recommended. We pool based on audio channels as channels must be the same as well as frequency.
            Source voice handles all of the audio playing and state for a single source."""
        voice_key = (source.audio_format.channels, source.audio_format.sample_size)
        if len(self._voice_pool[voice_key]) > 0:
            source_voice = self._voice_pool[voice_key].pop(0)
            source_voice.acquired(player)
        else:
            source_voice = self._get_voice(source, player)

        if source_voice.is_emitter:
            self._emitting_voices.append(source_voice)

        self._in_use.append(source_voice)
        return source_voice

    def _create_new_voice(self, source, player):
        """Has the driver create a new source voice for the source."""
        voice = lib.IXAudio2SourceVoice()

        wfx_format = self.create_wave_format(source.audio_format)

        callback = lib.XA2SourceCallback(player)
        self._xaudio2.CreateSourceVoice(ctypes.byref(voice),
                                        ctypes.byref(wfx_format),
                                        0,
                                        self.max_frequency_ratio,
                                        callback,
                                        None, None)
        return voice, callback

    def _get_voice(self, source, player):
        """Creates a new source voice and puts it into XA2SourceVoice high level wrap."""
        voice, callback = self._create_new_voice(source, player)
        return XA2SourceVoice(voice, callback, source.audio_format)

    def return_voice(self, voice):
        """Reset a voice and return it to the pool."""
        voice.reset()
        voice_key = (voice.audio_format.channels, voice.audio_format.sample_size)
        self._voice_pool[voice_key].append(voice)

        if voice.is_emitter:
            self._emitting_voices.remove(voice)

    @staticmethod
    def create_buffer(audio_data):
        """Creates a XAUDIO2_BUFFER to be used with a source voice.
            Audio data cannot be purged until the source voice has played it; doing so will cause glitches.
            Furthermore, if the data is not in a string buffer, such as pure bytes, it must be converted."""
        if type(audio_data.data) == bytes:
            data = (ctypes.c_char * audio_data.length)()
            ctypes.memmove(data, audio_data.data, audio_data.length)
        else:
            data = audio_data.data

        buff = lib.XAUDIO2_BUFFER()
        buff.AudioBytes = audio_data.length
        buff.pAudioData = data
        return buff

    @staticmethod
    def create_wave_format(audio_format):
        wfx = lib.WAVEFORMATEX()
        wfx.wFormatTag = lib.WAVE_FORMAT_PCM
        wfx.nChannels = audio_format.channels
        wfx.nSamplesPerSec = audio_format.sample_rate
        wfx.wBitsPerSample = audio_format.sample_size
        wfx.nBlockAlign = wfx.wBitsPerSample * wfx.nChannels // 8
        wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign
        return wfx


class XA2SourceVoice:

    def __init__(self, voice, callback, audio_format):
        self._voice_state = lib.XAUDIO2_VOICE_STATE()  # Used for buffer state, will be reused constantly.
        self._voice = voice
        self._callback = callback

        self.audio_format = audio_format
        # If it's a mono source, then we can make it an emitter.
        # In the future, non-mono source's can be supported as well.
        if audio_format is not None and audio_format.channels == 1:
            self._emitter = lib.X3DAUDIO_EMITTER()
            self._emitter.ChannelCount = audio_format.channels
            self._emitter.CurveDistanceScaler = 1.0

            # Commented are already set by the Player class.
            # Leaving for visibility on default values
            cone = lib.X3DAUDIO_CONE()
            # cone.InnerAngle = math.radians(360)
            # cone.OuterAngle = math.radians(360)
            cone.InnerVolume = 1.0
            # cone.OuterVolume = 1.0

            self._emitter.pCone = pointer(cone)
            self._emitter.pVolumeCurve = None
        else:
            self._emitter = None

    @property
    def player(self):
        """Returns the player class, stored within the callback."""
        return self._callback.xa2_player

    def delete(self):
        self._emitter = None
        self._voice.Stop(0, 0)
        self._voice.FlushSourceBuffers()
        self._voice = None
        self._callback.xa2_player = None

    def __del__(self):
        self.destroy()

    def destroy(self):
        """Completely destroy the voice."""
        self._emitter = None

        if self._voice is not None:
            try:
                self._voice.Stop(0, 0)
                self._voice.FlushSourceBuffers()
                self._voice.DestroyVoice()
            except TypeError:
                pass

            self._voice = None

        self._callback = None

    def acquired(self, player):
        """A voice has been reacquired, set the player for callback."""
        self._callback.xa2_player = player

    def reset(self):
        """When a voice is returned to the pool, reset position on emitter."""
        if self._emitter is not None:
            self.position = (0, 0, 0)

        self._voice.Stop(0, 0)
        self._voice.FlushSourceBuffers()
        self._callback.xa2_player = None

    @property
    def buffers_queued(self):
        """Get the amount of buffers in the current voice. Adding flag for no samples played is 3x faster."""
        self._voice.GetState(ctypes.byref(self._voice_state), lib.XAUDIO2_VOICE_NOSAMPLESPLAYED)
        return self._voice_state.BuffersQueued

    @property
    def volume(self):
        vol = c_float()
        self._voice.GetVolume(ctypes.byref(vol))
        return vol.value

    @volume.setter
    def volume(self, value):
        self._voice.SetVolume(value, 0)

    @property
    def is_emitter(self):
        return self._emitter is not None

    @property
    def position(self):
        if self.is_emitter:
            return self._emitter.Position.x, self._emitter.Position.y, self._emitter.Position.z
        else:
            return 0, 0, 0

    @position.setter
    def position(self, position):
        if self.is_emitter:
            x, y, z = position
            self._emitter.Position.x = x
            self._emitter.Position.y = y
            self._emitter.Position.z = z

    @property
    def min_distance(self):
        """Curve distance scaler that is used to scale normalized distance curves to user-defined world units,
        and/or to exaggerate their effect."""
        if self.is_emitter:
            return self._emitter.CurveDistanceScaler
        else:
            return 0

    @min_distance.setter
    def min_distance(self, value):
        if self.is_emitter:
            if self._emitter.CurveDistanceScaler != value:
                self._emitter.CurveDistanceScaler = min(value, lib.FLT_MAX)

    @property
    def frequency(self):
        """The actual frequency ratio. If voice is 3d enabled, will be overwritten next apply3d cycle."""
        value = c_float()
        self._voice.GetFrequencyRatio(byref(value))
        return value.value

    @frequency.setter
    def frequency(self, value):
        if self.frequency == value:
            return

        self._voice.SetFrequencyRatio(value, 0)

    @property
    def cone_orientation(self):
        """The orientation of the sound emitter."""
        if self.is_emitter:
            return self._emitter.OrientFront.x, self._emitter.OrientFront.y, self._emitter.OrientFront.z
        else:
            return 0, 0, 0

    @cone_orientation.setter
    def cone_orientation(self, value):
        if self.is_emitter:
            x, y, z = value
            self._emitter.OrientFront.x = x
            self._emitter.OrientFront.y = y
            self._emitter.OrientFront.z = z

    _ConeAngles = namedtuple('_ConeAngles', ['inside', 'outside'])

    @property
    def cone_angles(self):
        """The inside and outside angles of the sound projection cone."""
        if self.is_emitter:
            return self._ConeAngles(self._emitter.pCone.contents.InnerAngle, self._emitter.pCone.contents.OuterAngle)
        else:
            return self._ConeAngles(0, 0)

    def set_cone_angles(self, inside, outside):
        """The inside and outside angles of the sound projection cone."""
        if self.is_emitter:
            self._emitter.pCone.contents.InnerAngle = inside
            self._emitter.pCone.contents.OuterAngle = outside

    @property
    def cone_outside_volume(self):
        """The volume scaler of the sound beyond the outer cone."""
        if self.is_emitter:
            return self._emitter.pCone.contents.OuterVolume
        else:
            return 0

    @cone_outside_volume.setter
    def cone_outside_volume(self, value):
        if self.is_emitter:
            self._emitter.pCone.contents.OuterVolume = value

    @property
    def cone_inside_volume(self):
        """The volume scaler of the sound within the inner cone."""
        if self.is_emitter:
            return self._emitter.pCone.contents.InnerVolume
        else:
            return 0

    @cone_inside_volume.setter
    def cone_inside_volume(self, value):
        if self.is_emitter:
            self._emitter.pCone.contents.InnerVolume = value

    def flush(self):
        """Stop and removes all buffers already queued. OnBufferEnd is called for each."""
        self._voice.Stop(0, 0)
        self._voice.FlushSourceBuffers()

    def play(self):
        self._voice.Start(0, 0)

    def stop(self):
        self._voice.Stop(0, 0)

    def submit_buffer(self, x2_buffer):
        self._voice.SubmitSourceBuffer(ctypes.byref(x2_buffer), None)


class XAudio2Listener:
    def __init__(self, driver):
        self.xa2_driver = weakref.proxy(driver)
        self.listener = lib.X3DAUDIO_LISTENER()

        # Default listener orientations for DirectSound/XAudio2:
        # Front: (0, 0, 1), Up: (0, 1, 0)
        self.listener.OrientFront.x = 0
        self.listener.OrientFront.y = 0
        self.listener.OrientFront.z = 1

        self.listener.OrientTop.x = 0
        self.listener.OrientTop.y = 1
        self.listener.OrientTop.z = 0

    def __del__(self):
        self.delete()

    def delete(self):
        self.listener = None

    @property
    def position(self):
        return self.listener.Position.x, self.listener.Position.y, self.listener.Position.z

    @position.setter
    def position(self, value):
        x, y, z = value
        self.listener.Position.x = x
        self.listener.Position.y = y
        self.listener.Position.z = z

    @property
    def orientation(self):
        return self.listener.OrientFront.x, self.listener.OrientFront.y, self.listener.OrientFront.z, \
               self.listener.OrientTop.x, self.listener.OrientTop.y, self.listener.OrientTop.z

    @orientation.setter
    def orientation(self, orientation):
        front_x, front_y, front_z, top_x, top_y, top_z = orientation

        self.listener.OrientFront.x = front_x
        self.listener.OrientFront.y = front_y
        self.listener.OrientFront.z = front_z

        self.listener.OrientTop.x = top_x
        self.listener.OrientTop.y = top_y
        self.listener.OrientTop.z = top_z
