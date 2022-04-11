import atexit

import pyglet


def get_audio_device_manager():
    global _audio_device_manager

    if _audio_device_manager:
        return _audio_device_manager

    _audio_device_manager = None

    if pyglet.compat_platform == 'win32':
        from pyglet.media.devices.win32 import Win32AudioDeviceManager
        _audio_device_manager = Win32AudioDeviceManager()

    return _audio_device_manager


def _delete_manager():
    """Deletes existing manager. If audio device manager is stored anywhere.
    Required to remove handlers before exit, as it can cause problems with the event system's weakrefs."""
    global _audio_device_manager
    _audio_device_manager = None


global _audio_device_manager
_audio_device_manager = None

atexit.register(_delete_manager)
