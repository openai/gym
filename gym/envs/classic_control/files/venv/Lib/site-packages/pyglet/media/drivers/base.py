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

import math
import weakref

from abc import ABCMeta, abstractmethod

from pyglet.util import with_metaclass
from pyglet.media.player import clock


class AbstractAudioPlayer(with_metaclass(ABCMeta, object)):
    """Base class for driver audio players.
    """

    # Audio synchronization constants
    AUDIO_DIFF_AVG_NB = 20
    # no audio correction is done if too big error
    AV_NOSYNC_THRESHOLD = 10.0

    def __init__(self, source, player):
        """Create a new audio player.

        :Parameters:
            `source` : `Source`
                Source to play from.
            `player` : `Player`
                Player to receive EOS and video frame sync events.

        """
        # We only keep weakref to the player and its source to avoid
        # circular references. It's the player who owns the source and
        # the audio_player
        self.source = source
        self.player = weakref.proxy(player)

        self.audio_clock = clock

        # Audio synchronization
        self.audio_diff_avg_count = 0
        self.audio_diff_cum = 0.0
        self.audio_diff_avg_coef = math.exp(math.log10(0.01) / self.AUDIO_DIFF_AVG_NB)
        self.audio_diff_threshold = 0.1  # Experimental. ffplay computes it differently

    @abstractmethod
    def play(self):
        """Begin playback."""

    @abstractmethod
    def stop(self):
        """Stop (pause) playback."""

    @abstractmethod
    def delete(self):
        """Stop playing and clean up all resources used by player."""

    def _play_group(self, audio_players):
        """Begin simultaneous playback on a list of audio players."""
        # This should be overridden by subclasses for better synchrony.
        for player in audio_players:
            player.play()

    def _stop_group(self, audio_players):
        """Stop simultaneous playback on a list of audio players."""
        # This should be overridden by subclasses for better synchrony.
        for player in audio_players:
            player.stop()

    @abstractmethod
    def clear(self):
        """Clear all buffered data and prepare for replacement data.

        The player should be stopped before calling this method.
        """
        self.audio_diff_avg_count = 0
        self.audio_diff_cum = 0.0

    @abstractmethod
    def get_time(self):
        """Return approximation of current playback time within current source.

        Returns ``None`` if the audio player does not know what the playback
        time is (for example, before any valid audio data has been read).

        :rtype: float
        :return: current play cursor time, in seconds.
        """
        # TODO determine which source within group

    @abstractmethod
    def prefill_audio(self):
        """Prefill the audio buffer with audio data.

        This method is called before the audio player starts in order to 
        reduce the time it takes to fill the whole audio buffer.
        """

    def get_audio_time_diff(self):
        """Queries the time difference between the audio time and the `Player`
        master clock.

        The time difference returned is calculated using a weighted average on
        previous audio time differences. The algorithms will need at least 20
        measurements before returning a weighted average.

        :rtype: float
        :return: weighted average difference between audio time and master
            clock from `Player`
        """
        audio_time = self.get_time() or 0
        p_time = self.player.time
        diff = audio_time - p_time
        if abs(diff) < self.AV_NOSYNC_THRESHOLD:
            self.audio_diff_cum = diff + self.audio_diff_cum * self.audio_diff_avg_coef
            if self.audio_diff_avg_count < self.AUDIO_DIFF_AVG_NB:
                self.audio_diff_avg_count += 1
            else:
                avg_diff = self.audio_diff_cum * (1 - self.audio_diff_avg_coef)
                if abs(avg_diff) > self.audio_diff_threshold:
                    return avg_diff
        else:
            self.audio_diff_avg_count = 0
            self.audio_diff_cum = 0.0
        return 0.0

    def set_volume(self, volume):
        """See `Player.volume`."""
        pass

    def set_position(self, position):
        """See :py:attr:`~pyglet.media.Player.position`."""
        pass

    def set_min_distance(self, min_distance):
        """See `Player.min_distance`."""
        pass

    def set_max_distance(self, max_distance):
        """See `Player.max_distance`."""
        pass

    def set_pitch(self, pitch):
        """See :py:attr:`~pyglet.media.Player.pitch`."""
        pass

    def set_cone_orientation(self, cone_orientation):
        """See `Player.cone_orientation`."""
        pass

    def set_cone_inner_angle(self, cone_inner_angle):
        """See `Player.cone_inner_angle`."""
        pass

    def set_cone_outer_angle(self, cone_outer_angle):
        """See `Player.cone_outer_angle`."""
        pass

    def set_cone_outer_gain(self, cone_outer_gain):
        """See `Player.cone_outer_gain`."""
        pass

    @property
    def source(self):
        """Source to play from."""
        return self._source

    @source.setter
    def source(self, value):
        self._source = weakref.proxy(value)


class AbstractAudioDriver(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def create_audio_player(self, source, player):
        pass

    @abstractmethod
    def get_listener(self):
        pass

    @abstractmethod
    def delete(self):
        pass
