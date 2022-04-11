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

from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer
from pyglet.media.drivers.listener import AbstractListener


class SilentAudioPlayer(AbstractAudioPlayer):

    def delete(self):
        pass

    def play(self):
        pass

    def stop(self):
        pass

    def clear(self):
        pass

    def get_write_size(self):
        pass

    def write(self, audio_data, length):
        pass

    def get_time(self):
        return 0

    def set_volume(self, volume):
        pass

    def set_position(self, position):
        pass

    def set_min_distance(self, min_distance):
        pass

    def set_max_distance(self, max_distance):
        pass

    def set_pitch(self, pitch):
        pass

    def set_cone_orientation(self, cone_orientation):
        pass

    def set_cone_inner_angle(self, cone_inner_angle):
        pass

    def set_cone_outer_angle(self, cone_outer_angle):
        pass

    def set_cone_outer_gain(self, cone_outer_gain):
        pass

    def prefill_audio(self):
        pass


class SilentDriver(AbstractAudioDriver):

    def create_audio_player(self, source, player):
        return SilentAudioPlayer(source, player)

    def get_listener(self):
        return SilentListener()

    def delete(self):
        pass


class SilentListener(AbstractListener):

    def _set_volume(self, volume):
        pass

    def _set_position(self, position):
        pass

    def _set_forward_orientation(self, orientation):
        pass

    def _set_up_orientation(self, orientation):
        pass

    def _set_orientation(self):
        pass
