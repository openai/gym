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

"""Application-wide functionality.

Applications
------------

Most applications need only call :func:`run` after creating one or more 
windows to begin processing events.  For example, a simple application 
consisting of one window is::

    import pyglet

    win = pyglet.window.Window()
    pyglet.app.run()


Events
======

To handle events on the main event loop, instantiate it manually.  The
following example exits the application as soon as any window is closed (the
default policy is to wait until all windows are closed)::

    event_loop = pyglet.app.EventLoop()

    @event_loop.event
    def on_window_close(window):
        event_loop.exit()

.. versionadded:: 1.1
"""

import sys
import weakref

from pyglet.app.base import EventLoop
from pyglet import compat_platform

_is_pyglet_doc_run = hasattr(sys, "is_pyglet_doc_run") and sys.is_pyglet_doc_run


if _is_pyglet_doc_run:
    from pyglet.app.base import PlatformEventLoop
else:
    if compat_platform == 'darwin':
        from pyglet.app.cocoa import CocoaEventLoop as PlatformEventLoop
    elif compat_platform in ('win32', 'cygwin'):
        from pyglet.app.win32 import Win32EventLoop as PlatformEventLoop
    else:
        from pyglet.app.xlib import XlibEventLoop as PlatformEventLoop


class AppException(Exception):
    pass


windows = weakref.WeakSet()
"""Set of all open windows (including invisible windows).  Instances of
:class:`pyglet.window.Window` are automatically added to this set upon 
construction. The set uses weak references, so windows are removed from 
the set when they are no longer referenced or are closed explicitly.
"""


def run():
    """Begin processing events, scheduled functions and window updates.

    This is a convenience function, equivalent to::

        pyglet.app.event_loop.run()

    """
    event_loop.run()


def exit():
    """Exit the application event loop.

    Causes the application event loop to finish, if an event loop is currently
    running.  The application may not necessarily exit (for example, there may
    be additional code following the `run` invocation).

    This is a convenience function, equivalent to::

        event_loop.exit()

    """
    event_loop.exit()


#: The global event loop.  Applications can replace this
#: with their own subclass of :class:`EventLoop` before calling 
#: :meth:`EventLoop.run`.
event_loop = EventLoop()

#: The platform-dependent event loop.
#: Applications must not subclass or replace this :class:`PlatformEventLoop`
#: object.
platform_event_loop = PlatformEventLoop()
