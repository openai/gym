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
"""High-level sound and video player."""

import threading
from collections import deque

import pyglet
from pyglet.media import buffered_logger as bl
from pyglet.media.drivers import get_audio_driver
from pyglet.media.codecs.base import Source, SourceGroup

_debug = pyglet.options['debug_media']


# class AudioClock(pyglet.clock.Clock):
#     """A dedicated background Clock for refilling audio buffers."""
#
#     def __init__(self, interval=0.1):
#         super().__init__()
#         self._interval = interval
#         self._thread = threading.Thread(target=self._tick_clock, daemon=True)
#         self._thread.start()
#
#     def _tick_clock(self):
#         while True:
#             self.tick()
#             self.sleep(self._interval * 1000000)
#
#
# clock = AudioClock()

clock = pyglet.clock.get_default()


class PlaybackTimer:
    """Playback Timer.

    This is a simple timer object which tracks the time elapsed. It can be
    paused and reset.
    """

    def __init__(self):
        """Initialize the timer with time 0."""
        self._time = 0.0
        self._systime = None

    def start(self):
        """Start the timer."""
        self._systime = clock.time()

    def pause(self):
        """Pause the timer."""
        self._time = self.get_time()
        self._systime = None

    def reset(self):
        """Reset the timer to 0."""
        self._time = 0.0
        if self._systime is not None:
            self._systime = clock.time()

    def get_time(self):
        """Get the elapsed time."""
        if self._systime is None:
            now = self._time
        else:
            now = clock.time() - self._systime + self._time
        return now

    def set_time(self, value):
        """
        Manually set the elapsed time.

        Args:
            value (float): the new elapsed time value
        """
        self.reset()
        self._time = value


class _PlayerProperty:
    """Descriptor for Player attributes to forward to the AudioPlayer.

    We want the Player to have attributes like volume, pitch, etc. These are
    actually implemented by the AudioPlayer. So this descriptor will forward
    an assignement to one of the attributes to the AudioPlayer. For example
    `player.volume = 0.5` will call `player._audio_player.set_volume(0.5)`.

    The Player class has default values at the class level which are retrieved
    if not found on the instance.
    """

    def __init__(self, attribute, doc=None):
        self.attribute = attribute
        self.__doc__ = doc or ''

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if '_' + self.attribute in obj.__dict__:
            return obj.__dict__['_' + self.attribute]
        return getattr(objtype, '_' + self.attribute)

    def __set__(self, obj, value):
        obj.__dict__['_' + self.attribute] = value
        if obj._audio_player:
            getattr(obj._audio_player, 'set_' + self.attribute)(value)


class Player(pyglet.event.EventDispatcher):
    """High-level sound and video player."""

    # Spacialisation attributes, preserved between audio players
    _volume = 1.0
    _min_distance = 1.0
    _max_distance = 100000000.

    _position = (0, 0, 0)
    _pitch = 1.0

    _cone_orientation = (0, 0, 1)
    _cone_inner_angle = 360.
    _cone_outer_angle = 360.
    _cone_outer_gain = 1.

    def __init__(self):
        """Initialize the Player with a MasterClock."""
        self._source = None
        self._playlists = deque()
        self._audio_player = None

        self._texture = None
        # Desired play state (not an indication of actual state).
        self._playing = False

        self._timer = PlaybackTimer()
        #: Loop the current source indefinitely or until
        #: :meth:`~Player.next_source` is called. Defaults to ``False``.
        #:
        #: :type: bool
        #:
        #: .. versionadded:: 1.4
        self.loop = False

        # self.pr = cProfile.Profile()

    def __del__(self):
        """Release the Player resources."""
        self.delete()

    def queue(self, source):
        """
        Queue the source on this player.

        If the player has no source, the player will start to play immediately
        or pause depending on its :attr:`.playing` attribute.

        Args:
            source (Source or Iterable[Source]): The source to queue.
        """
        if isinstance(source, (Source, SourceGroup)):
            source = _one_item_playlist(source)
        else:
            try:
                source = iter(source)
            except TypeError:
                raise TypeError("source must be either a Source or an iterable."
                                " Received type {0}".format(type(source)))
        self._playlists.append(source)

        if self.source is None:
            source = next(self._playlists[0])
            self._source = source.get_queue_source()

        self._set_playing(self._playing)

    def _set_playing(self, playing):
        # stopping = self._playing and not playing
        # starting = not self._playing and playing

        self._playing = playing
        source = self.source

        if playing and source:
            if source.audio_format:
                if self._audio_player is None:
                    self._create_audio_player()
                if self._audio_player:
                    # We succesfully created an audio player
                    self._audio_player.prefill_audio()

            if bl.logger is not None:
                bl.logger.init_wall_time()
                bl.logger.log("p.P._sp", 0.0)

            if source.video_format:
                if not self._texture:
                    self._create_texture()

            if self._audio_player:
                self._audio_player.play()
            if source.video_format:
                pyglet.clock.schedule_once(self.update_texture, 0)
            # For audio synchronization tests, the following will
            # add a delay to de-synchronize the audio.
            # Negative number means audio runs ahead.
            # self._mclock._systime += -0.3
            self._timer.start()
            if self._audio_player is None and source.video_format is None:
                pyglet.clock.schedule_once(lambda dt: self.dispatch_event("on_eos"), source.duration)

        else:
            if self._audio_player:
                self._audio_player.stop()

            pyglet.clock.unschedule(self.update_texture)
            self._timer.pause()

    @property
    def playing(self):
        """
        bool: Read-only. Determine if the player state is playing.

        The *playing* property is irrespective of whether or not there is
        actually a source to play. If *playing* is ``True`` and a source is
        queued, it will begin to play immediately. If *playing* is ``False``,
        it is implied that the player is paused. There is no other possible
        state.
        """
        return self._playing

    def play(self):
        """Begin playing the current source.

        This has no effect if the player is already playing.
        """
        self._set_playing(True)

    def pause(self):
        """Pause playback of the current source.

        This has no effect if the player is already paused.
        """
        self._set_playing(False)

    def delete(self):
        """Release the resources acquired by this player.

        The internal audio player and the texture will be deleted.
        """
        if self._audio_player:
            self._audio_player.delete()
            self._audio_player = None
        if self._texture:
            self._texture = None

    def next_source(self):
        """Move immediately to the next source in the current playlist.

        If the playlist is empty, discard it and check if another playlist
        is queued. There may be a gap in playback while the audio buffer
        is refilled.
        """
        was_playing = self._playing
        self.pause()
        self._timer.reset()

        if self.source:
            # Reset source to the beginning
            self.seek(0.0)
            self.source.is_player_source = False

        playlists = self._playlists
        if not playlists:
            return

        try:
            source = next(playlists[0])
        except StopIteration:
            self._playlists.popleft()
            if not self._playlists:
                source = None
            else:
                # Could someone queue an iterator which is empty??
                source = next(self._playlists[0])

        if source is None:
            self._source = None
            self.delete()
            self.dispatch_event('on_player_eos')
        else:
            old_audio_format = self.source.audio_format
            old_video_format = self.source.video_format
            self._source = source.get_queue_source()

            if old_audio_format == self.source.audio_format:
                self._audio_player.clear()
                self._audio_player.source = self.source
            else:
                self._audio_player.delete()
                self._audio_player = None
            if old_video_format != self.source.video_format:
                self._texture = None
                pyglet.clock.unschedule(self.update_texture)

            self._set_playing(was_playing)
            self.dispatch_event('on_player_next_source')

    def seek(self, time):
        """
        Seek for playback to the indicated timestamp on the current source.

        Timestamp is expressed in seconds. If the timestamp is outside the
        duration of the source, it will be clamped to the end.

        Args:
            time (float): The time where to seek in the source, clamped to the
                beginning and end of the source.
        """
        playing = self._playing
        if playing:
            self.pause()
        if not self.source:
            return

        if bl.logger is not None:
            bl.logger.log("p.P.sk", time)

        self._timer.set_time(time)
        self.source.seek(time)
        if self._audio_player:
            # XXX: According to docstring in AbstractAudioPlayer this cannot
            # be called when the player is not stopped
            self._audio_player.clear()
        if self.source.video_format:
            self.update_texture()
            pyglet.clock.unschedule(self.update_texture)
        self._set_playing(playing)

    def _create_audio_player(self):
        assert not self._audio_player
        assert self.source

        source = self.source
        audio_driver = get_audio_driver()
        if audio_driver is None:
            # Failed to find a valid audio driver
            return

        self._audio_player = audio_driver.create_audio_player(source, self)

        # Set the audio player attributes
        for attr in ('volume', 'min_distance', 'max_distance', 'position',
                     'pitch', 'cone_orientation', 'cone_inner_angle',
                     'cone_outer_angle', 'cone_outer_gain'):
            value = getattr(self, attr)
            setattr(self, attr, value)

    @property
    def source(self):
        """Source: Read-only. The current :class:`Source`, or ``None``."""
        return self._source

    @property
    def time(self):
        """
        float: Read-only. Current playback time of the current source.

        The playback time is a float expressed in seconds, with 0.0 being the
        beginning of the media. The playback time returned represents the
        player master clock time which is used to synchronize both the audio
        and the video.
        """
        return self._timer.get_time()

    def _create_texture(self):
        video_format = self.source.video_format
        self._texture = pyglet.image.Texture.create(
            video_format.width, video_format.height, rectangle=True)
        self._texture = self._texture.get_transform(flip_y=True)
        # After flipping the texture along the y axis, the anchor_y is set
        # to the top of the image. We want to keep it at the bottom.
        self._texture.anchor_y = 0
        return self._texture

    @property
    def texture(self):
        """
        :class:`pyglet.image.Texture`: Get the texture for the current video frame.

        You should call this method every time you display a frame of video,
        as multiple textures might be used. The return value will be None if
        there is no video in the current source.
        """
        return self._texture

    def get_texture(self):
        """
        Get the texture for the current video frame.

        You should call this method every time you display a frame of video,
        as multiple textures might be used. The return value will be None if
        there is no video in the current source.

        Returns:
            :class:`pyglet.image.Texture`

        .. deprecated:: 1.4
                Use :attr:`~texture` instead
        """
        return self.texture

    def seek_next_frame(self):
        """Step forwards one video frame in the current source."""
        time = self.source.get_next_video_timestamp()
        if time is None:
            return
        self.seek(time)

    def update_texture(self, dt=None):
        """Manually update the texture from the current source.

        This happens automatically, so you shouldn't need to call this method.

        Args:
            dt (float): The time elapsed since the last call to
                ``update_texture``.
        """
        # self.pr.disable()
        # if dt > 0.05:
        #     print("update_texture dt:", dt)
        #     import pstats
        #     ps = pstats.Stats(self.pr).sort_stats("cumulative")
        #     ps.print_stats()
        source = self.source
        time = self.time
        if bl.logger is not None:
            bl.logger.log(
                "p.P.ut.1.0", dt, time,
                self._audio_player.get_time() if self._audio_player else 0,
                bl.logger.rebased_wall_time()
            )

        frame_rate = source.video_format.frame_rate
        frame_duration = 1 / frame_rate
        ts = source.get_next_video_timestamp()
        # Allow up to frame_duration difference
        while ts is not None and ts + frame_duration < time:
            source.get_next_video_frame()  # Discard frame
            if bl.logger is not None:
                bl.logger.log("p.P.ut.1.5", ts)
            ts = source.get_next_video_timestamp()

        if bl.logger is not None:
            bl.logger.log("p.P.ut.1.6", ts)

        if ts is None:
            # No more video frames to show. End of video stream.
            if bl.logger is not None:
                bl.logger.log("p.P.ut.1.7", frame_duration)

            pyglet.clock.schedule_once(self._video_finished, 0)
            return

        image = source.get_next_video_frame()
        if image is not None:
            if self._texture is None:
                self._create_texture()
            self._texture.blit_into(image, 0, 0, 0)
        elif bl.logger is not None:
            bl.logger.log("p.P.ut.1.8")

        ts = source.get_next_video_timestamp()
        if ts is None:
            delay = frame_duration
        else:
            delay = ts - time

        delay = max(0.0, delay)
        if bl.logger is not None:
            bl.logger.log("p.P.ut.1.9", delay, ts)
        pyglet.clock.schedule_once(self.update_texture, delay)
        # self.pr.enable()

    def _video_finished(self, dt):
        if self._audio_player is None:
            self.dispatch_event("on_eos")

    volume = _PlayerProperty('volume', doc="""
    The volume level of sound playback.

    The nominal level is 1.0, and 0.0 is silence.

    The volume level is affected by the distance from the listener (if
    positioned).
    """)
    min_distance = _PlayerProperty('min_distance', doc="""
    The distance beyond which the sound volume drops by half, and within
    which no attenuation is applied.

    The minimum distance controls how quickly a sound is attenuated as it
    moves away from the listener. The gain is clamped at the nominal value
    within the min distance. By default the value is 1.0.

    The unit defaults to meters, but can be modified with the listener
    properties. """)
    max_distance = _PlayerProperty('max_distance', doc="""
    The distance at which no further attenuation is applied.

    When the distance from the listener to the player is greater than this
    value, attenuation is calculated as if the distance were value. By
    default the maximum distance is infinity.

    The unit defaults to meters, but can be modified with the listener
    properties.
    """)
    position = _PlayerProperty('position', doc="""
    The position of the sound in 3D space.

    The position is given as a tuple of floats (x, y, z). The unit
    defaults to meters, but can be modified with the listener properties.
    """)
    pitch = _PlayerProperty('pitch', doc="""
    The pitch shift to apply to the sound.

    The nominal pitch is 1.0. A pitch of 2.0 will sound one octave higher,
    and play twice as fast. A pitch of 0.5 will sound one octave lower, and
    play twice as slow. A pitch of 0.0 is not permitted.
    """)
    cone_orientation = _PlayerProperty('cone_orientation', doc="""
    The direction of the sound in 3D space.

    The direction is specified as a tuple of floats (x, y, z), and has no
    unit. The default direction is (0, 0, -1). Directional effects are only
    noticeable if the other cone properties are changed from their default
    values.
    """)
    cone_inner_angle = _PlayerProperty('cone_inner_angle', doc="""
    The interior angle of the inner cone.

    The angle is given in degrees, and defaults to 360. When the listener
    is positioned within the volume defined by the inner cone, the sound is
    played at normal gain (see :attr:`volume`).
    """)
    cone_outer_angle = _PlayerProperty('cone_outer_angle', doc="""
    The interior angle of the outer cone.

    The angle is given in degrees, and defaults to 360. When the listener
    is positioned within the volume defined by the outer cone, but outside
    the volume defined by the inner cone, the gain applied is a smooth
    interpolation between :attr:`volume` and :attr:`cone_outer_gain`.
    """)
    cone_outer_gain = _PlayerProperty('cone_outer_gain', doc="""
    The gain applied outside the cone.

    When the listener is positioned outside the volume defined by the outer
    cone, this gain is applied instead of :attr:`volume`.
    """)

    # Events

    def on_player_eos(self):
        """The player ran out of sources. The playlist is empty.

        :event:
        """
        if _debug:
            print('Player.on_player_eos')

    def on_eos(self):
        """The current source ran out of data.

        The default behaviour is to advance to the next source in the
        playlist if the :attr:`.loop` attribute is set to ``False``.
        If :attr:`.loop` attribute is set to ``True``, the current source
        will start to play again until :meth:`next_source` is called or
        :attr:`.loop` is set to ``False``.

        :event:
        """
        if _debug:
            print('Player.on_eos')
        if bl.logger is not None:
            bl.logger.log("p.P.oe")
            bl.logger.close()

        if self.loop:
            was_playing = self._playing
            self.pause()
            self._timer.reset()

            if self.source:
                # Reset source to the beginning
                self.seek(0.0)
            self._audio_player.clear()
            self._set_playing(was_playing)

        else:
            self.next_source()

    def on_player_next_source(self):
        """The player starts to play the next queued source in the playlist.

        This is a useful event for adjusting the window size to the new
        source :class:`VideoFormat` for example.

        :event:
        """
        pass


Player.register_event_type('on_eos')
Player.register_event_type('on_player_eos')
Player.register_event_type('on_player_next_source')


def _one_item_playlist(source):
    yield source


class PlayerGroup:
    """Group of players that can be played and paused simultaneously.

    Create a player group for the given list of players.

    All players in the group must currently not belong to any other group.

    Args:
        players (List[Player]): List of :class:`.Player` s in this group.
    """

    def __init__(self, players):
        """Initialize the PlayerGroup with the players."""
        self.players = list(players)

    def play(self):
        """Begin playing all players in the group simultaneously."""
        audio_players = [p._audio_player
                         for p in self.players if p._audio_player]
        if audio_players:
            audio_players[0]._play_group(audio_players)
        for player in self.players:
            player.play()

    def pause(self):
        """Pause all players in the group simultaneously."""
        audio_players = [p._audio_player
                         for p in self.players if p._audio_player]
        if audio_players:
            audio_players[0]._stop_group(audio_players)
        for player in self.players:
            player.pause()
