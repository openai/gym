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
"""Drivers for playing back media."""

import atexit

import pyglet

_debug = pyglet.options['debug_media']


def get_audio_driver():
    """Get the preferred audio driver for the current platform.

    Currently pyglet supports DirectSound, PulseAudio and OpenAL drivers. If
    the platform supports more than one of those audio drivers, the
    application can give its preference with :data:`pyglet.options` ``audio``
    keyword. See the Programming guide, section
    :doc:`/programming_guide/media`.

    Returns:
        AbstractAudioDriver : The concrete implementation of the preferred 
        audio driver for this platform.
    """
    global _audio_driver

    if _audio_driver:
        return _audio_driver

    _audio_driver = None

    for driver_name in pyglet.options['audio']:
        try:
            if driver_name == 'pulse':
                from . import pulse
                _audio_driver = pulse.create_audio_driver()
                break
            elif driver_name == 'xaudio2':
                from pyglet.libs.win32.constants import WINDOWS_8_OR_GREATER
                if WINDOWS_8_OR_GREATER:
                    from . import xaudio2
                    _audio_driver = xaudio2.create_audio_driver()
                    break
            elif driver_name == 'directsound':
                from . import directsound
                _audio_driver = directsound.create_audio_driver()
                break
            elif driver_name == 'openal':
                from . import openal
                _audio_driver = openal.create_audio_driver()
                break
            elif driver_name == 'silent':
                from . import silent
                _audio_driver = silent.create_audio_driver()
                break
        except Exception:
            if _debug:
                print('Error importing driver %s:' % driver_name)
                import traceback
                traceback.print_exc()
    else:
        from . import silent
        _audio_driver = silent.create_audio_driver()

    return _audio_driver


def _delete_audio_driver():
    # First cleanup any remaining spontaneous Player
    from .. import Source
    for p in Source._players:
        # Remove the reference to _on_player_eos which had a closure on the player
        p.on_player_eos = None
        del p

    del Source._players
    global _audio_driver
    _audio_driver = None


_audio_driver = None
atexit.register(_delete_audio_driver)
