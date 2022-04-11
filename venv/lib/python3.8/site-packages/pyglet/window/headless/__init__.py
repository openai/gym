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

from pyglet.window import BaseWindow, _PlatformEventHandler, _ViewEventHandler
from pyglet.window import WindowException, NoSuchDisplayException, MouseCursorException
from pyglet.window import MouseCursor, DefaultMouseCursor, ImageMouseCursor


from pyglet.libs.egl import egl


from pyglet.canvas.headless import HeadlessCanvas

# from pyglet.window import key
# from pyglet.window import mouse
from pyglet.event import EventDispatcher

# Platform event data is single item, so use platform event handler directly.
HeadlessEventHandler = _PlatformEventHandler
ViewEventHandler = _ViewEventHandler


class HeadlessWindow(BaseWindow):
    _egl_display_connection = None
    _egl_surface = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _recreate(self, changes):
        pass

    def flip(self):
        if self.context:
            self.context.flip()

    def switch_to(self):
        if self.context:
            self.context.set_current()

    def set_caption(self, caption):
        pass

    def set_minimum_size(self, width, height):
        pass

    def set_maximum_size(self, width, height):
        pass

    def set_size(self, width, height):
        pass

    def get_size(self):
        return self._width, self._height

    def set_location(self, x, y):
        pass

    def get_location(self):
        pass

    def activate(self):
        pass

    def set_visible(self, visible=True):
        pass

    def minimize(self):
        pass

    def maximize(self):
        pass

    def set_vsync(self, vsync):
        pass

    def set_mouse_platform_visible(self, platform_visible=None):
        pass

    def set_exclusive_mouse(self, exclusive=True):
        pass

    def set_exclusive_keyboard(self, exclusive=True):
        pass

    def get_system_mouse_cursor(self, name):
        pass

    def dispatch_events(self):
        while self._event_queue:
            EventDispatcher.dispatch_event(self, *self._event_queue.pop(0))

    def dispatch_pending_events(self):
        pass

    def _create(self):
        self._egl_display_connection = self.display._display_connection

        if not self._egl_surface:
            pbuffer_attribs = (egl.EGL_WIDTH, self._width, egl.EGL_HEIGHT, self._height, egl.EGL_NONE)
            pbuffer_attrib_array = (egl.EGLint * len(pbuffer_attribs))(*pbuffer_attribs)
            self._egl_surface = egl.eglCreatePbufferSurface(self._egl_display_connection,
                                                            self.config._egl_config,
                                                            pbuffer_attrib_array)

            self.canvas = HeadlessCanvas(self.display, self._egl_surface)

            self.context.attach(self.canvas)

            self.dispatch_event('on_resize', self._width, self._height)


__all__ = ["HeadlessWindow"]
