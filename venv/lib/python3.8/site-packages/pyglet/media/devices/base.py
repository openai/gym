from abc import ABCMeta, abstractmethod

from pyglet import event
from pyglet.util import with_metaclass


class DeviceState:
    ACTIVE = "active"
    DISABLED = "disabled"
    MISSING = "missing"
    UNPLUGGED = "unplugged"


class DeviceFlow:
    OUTPUT = "output"
    INPUT = "input"
    INPUT_OUTPUT = "input/output"


class AudioDevice:
    """Base class for a platform independent audio device.
       _platform_state and _platform_flow is used to make device state numbers."""
    _platform_state = {}  # Must be defined by the parent.
    _platform_flow = {}  # Must be defined by the parent.

    def __init__(self, dev_id, name, description, flow, state):
        self.id = dev_id
        self.flow = flow
        self.state = state
        self.name = name
        self.description = description

    def __repr__(self):
        return "{}(name={}, state={}, flow={})".format(
            self.__class__.__name__, self.name, self._platform_state[self.state], self._platform_flow[self.flow])


class AbstractAudioDeviceManager(with_metaclass(ABCMeta, event.EventDispatcher, object)):

    def __del__(self):
        """Required to remove handlers before exit, as it can cause problems with the event system's weakrefs."""
        self.remove_handlers(self)

    @abstractmethod
    def get_default_output(self):
        """Returns a default active output device or None if none available."""
        pass

    @abstractmethod
    def get_default_input(self):
        """Returns a default active input device or None if none available."""
        pass

    @abstractmethod
    def get_output_devices(self):
        """Returns a list of all active output devices."""
        pass

    @abstractmethod
    def get_input_devices(self):
        """Returns a list of all active input devices."""
        pass

    @abstractmethod
    def get_all_devices(self):
        """Returns a list of all audio devices, no matter what state they are in."""
        pass

    def on_device_state_changed(self, device, old_state, new_state):
        """Event, occurs when the state of a device changes, provides old state and new state."""
        pass

    def on_device_added(self, device):
        """Event, occurs when a new device is added to the system."""
        pass

    def on_device_removed(self, device):
        """Event, occurs when an existing device is removed from the system."""
        pass

    def on_default_changed(self, device):
        """Event, occurs when the default audio device changes."""
        pass


AbstractAudioDeviceManager.register_event_type('on_device_state_changed')
AbstractAudioDeviceManager.register_event_type('on_device_added')
AbstractAudioDeviceManager.register_event_type('on_device_removed')
AbstractAudioDeviceManager.register_event_type('on_default_changed')
