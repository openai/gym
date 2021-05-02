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

from pyglet.input.base import Tablet, TabletCanvas
from pyglet.input.base import TabletCursor, DeviceOpenException
from pyglet.input.x11_xinput import XInputWindowEventDispatcher
from pyglet.input.x11_xinput import get_devices, DeviceResponder


try:
    from pyglet.libs.x11 import xinput as xi
    _have_xinput = True
except:
    _have_xinput = False


class XInputTablet(Tablet):
    name = 'XInput Tablet'

    def __init__(self, cursors):
        self.cursors = cursors

    def open(self, window):
        return XInputTabletCanvas(window, self.cursors)


class XInputTabletCanvas(DeviceResponder, TabletCanvas):
    def __init__(self, window, cursors):
        super(XInputTabletCanvas, self).__init__(window)
        self.cursors = cursors

        dispatcher = XInputWindowEventDispatcher.get_dispatcher(window)

        self.display = window.display
        self._open_devices = []
        self._cursor_map = {}
        for cursor in cursors:
            device = cursor.device
            device_id = device._device_id
            self._cursor_map[device_id] = cursor

            cursor.max_pressure = device.axes[2].max

            if self.display._display != device.display._display:
                raise DeviceOpenException('Window and device displays differ')

            open_device = xi.XOpenDevice(device.display._display, device_id)
            if not open_device:
                # Ignore this cursor; fail if no cursors added
                continue
            self._open_devices.append(open_device)

            dispatcher.open_device(device_id, open_device, self)

    def close(self):
        for device in self._open_devices:
            xi.XCloseDevice(self.display._display, device)

    def _motion(self, e):
        cursor = self._cursor_map.get(e.deviceid)
        x = e.x
        y = self.window.height - e.y
        pressure = e.axis_data[2] / float(cursor.max_pressure)
        self.dispatch_event('on_motion', cursor, x, y, pressure, 0.0, 0.0)

    def _proximity_in(self, e):
        cursor = self._cursor_map.get(e.deviceid)
        self.dispatch_event('on_enter', cursor)

    def _proximity_out(self, e):
        cursor = self._cursor_map.get(e.deviceid)
        self.dispatch_event('on_leave', cursor)


class XInputTabletCursor(TabletCursor):
    def __init__(self, device):
        super(XInputTabletCursor, self).__init__(device.name)
        self.device = device


def get_tablets(display=None):
    # Each cursor appears as a separate xinput device; find devices that look
    # like Wacom tablet cursors and amalgamate them into a single tablet. 
    valid_names = ('stylus', 'cursor', 'eraser', 'wacom', 'pen', 'pad')
    cursors = []
    devices = get_devices(display)
    for device in devices:
        dev_name = device.name.lower().split()
        if any(n in dev_name for n in valid_names) and len(device.axes) >= 3:
            cursors.append(XInputTabletCursor(device))

    if cursors:
        return [XInputTablet(cursors)]
    return []
