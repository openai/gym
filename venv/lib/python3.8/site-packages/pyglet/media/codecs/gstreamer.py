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

"""Multi-format decoder using Gstreamer.
"""
import queue
import atexit
import weakref
import tempfile

from threading import Event, Thread

from ..exceptions import MediaDecodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


class GStreamerDecodeException(MediaDecodeException):
    pass


class _GLibMainLoopThread(Thread):
    """A background Thread for a GLib MainLoop"""
    def __init__(self):
        super().__init__(daemon=True)
        self.mainloop = GLib.MainLoop.new(None, False)
        self.start()

    def run(self):
        self.mainloop.run()


class _MessageHandler:
    """Message Handler class for GStreamer Sources.
    
    This separate class holds a weak reference to the
    Source, preventing garbage collection issues. 
    
    """
    def __init__(self, source):
        self.source = weakref.proxy(source)

    def message(self, bus, message):
        """The main message callback"""
        if message.type == Gst.MessageType.EOS:

            self.source.queue.put(self.source.sentinal)
            if not self.source.caps:
                raise GStreamerDecodeException("Appears to be an unsupported file")

        elif message.type == Gst.MessageType.ERROR:
            raise GStreamerDecodeException(message.parse_error())

    def notify_caps(self, pad, *args):
        """notify::caps callback"""
        self.source.caps = True
        info = pad.get_current_caps().get_structure(0)

        self.source._duration = pad.get_peer().query_duration(Gst.Format.TIME).duration / Gst.SECOND
        channels = info.get_int('channels')[1]
        sample_rate = info.get_int('rate')[1]
        sample_size = int("".join(filter(str.isdigit, info.get_string('format'))))

        self.source.audio_format = AudioFormat(channels=channels, sample_size=sample_size, sample_rate=sample_rate)

        # Allow GStreamerSource.__init__ to complete:
        self.source.is_ready.set()

    def pad_added(self, element, pad):
        """pad-added callback"""
        name = pad.query_caps(None).to_string()
        if name.startswith('audio/x-raw'):
            nextpad = self.source.converter.get_static_pad('sink')
            if not nextpad.is_linked():
                self.source.pads = True
                pad.link(nextpad)

    def no_more_pads(self, element):
        """Finished Adding pads"""
        if not self.source.pads:
            raise GStreamerDecodeException('No Streams Found')

    def new_sample(self, sink):
        """new-sample callback"""
        # Pull the sample, and get it's buffer:
        buffer = sink.emit('pull-sample').get_buffer()
        # Extract a copy of the memory in the buffer:
        mem = buffer.extract_dup(0, buffer.get_size())
        self.source.queue.put(mem)
        return Gst.FlowReturn.OK

    @staticmethod
    def unknown_type(uridecodebin, decodebin, caps):
        """unknown-type callback for unreadable files"""
        streaminfo = caps.to_string()
        if not streaminfo.startswith('audio/'):
            return
        raise GStreamerDecodeException(streaminfo)


class GStreamerSource(StreamingSource):

    source_instances = weakref.WeakSet()
    sentinal = object()

    def __init__(self, filename, file=None):
        self._pipeline = Gst.Pipeline()

        msg_handler = _MessageHandler(self)

        if file:
            file.seek(0)
            self._file = tempfile.NamedTemporaryFile(buffering=False)
            self._file.write(file.read())
            filename = self._file.name

        # Create the major parts of the pipeline:
        self.filesrc = Gst.ElementFactory.make("filesrc", None)
        self.decoder = Gst.ElementFactory.make("decodebin", None)
        self.converter = Gst.ElementFactory.make("audioconvert", None)
        self.appsink = Gst.ElementFactory.make("appsink", None)
        if not all((self.filesrc, self.decoder, self.converter, self.appsink)):
            raise GStreamerDecodeException("Could not initialize GStreamer.")

        # Set callbacks for EOS and error messages:
        self._pipeline.bus.add_signal_watch()
        self._pipeline.bus.connect("message", msg_handler.message)

        # Set the file path to load:
        self.filesrc.set_property("location", filename)

        # Set decoder callback handlers:
        self.decoder.connect("pad-added", msg_handler.pad_added)
        self.decoder.connect("no-more-pads", msg_handler.no_more_pads)
        self.decoder.connect("unknown-type", msg_handler.unknown_type)

        # Set the sink's capabilities and behavior:
        self.appsink.set_property('caps', Gst.Caps.from_string('audio/x-raw,format=S16LE,layout=interleaved'))
        self.appsink.set_property('drop', False)
        self.appsink.set_property('sync', False)
        self.appsink.set_property('max-buffers', 0)     # unlimited
        self.appsink.set_property('emit-signals', True)
        # The callback to receive decoded data:
        self.appsink.connect("new-sample", msg_handler.new_sample)

        # Add all components to the pipeline:
        self._pipeline.add(self.filesrc)
        self._pipeline.add(self.decoder)
        self._pipeline.add(self.converter)
        self._pipeline.add(self.appsink)
        # Link together necessary components:
        self.filesrc.link(self.decoder)
        self.decoder.link(self.converter)
        self.converter.link(self.appsink)

        # Callback to notify once the sink is ready:
        self.caps_handler = self.appsink.get_static_pad("sink").connect("notify::caps", msg_handler.notify_caps)

        # Set by callbacks:
        self.pads = False
        self.caps = False
        self._pipeline.set_state(Gst.State.PLAYING)
        self.queue = queue.Queue(5)
        self._finished = Event()
        # Wait until the is_ready event is set by a callback:
        self.is_ready = Event()
        if not self.is_ready.wait(timeout=1):
            raise GStreamerDecodeException('Initialization Error')

        GStreamerSource.source_instances.add(self)

    def __del__(self):
        self.delete()

    def delete(self):
        if hasattr(self, '_file'):
            self._file.close()

        try:
            while not self.queue.empty():
                self.queue.get_nowait()
            sink = self.appsink.get_static_pad("sink")
            if sink.handler_is_connected(self.caps_handler):
                sink.disconnect(self.caps_handler)
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline.bus.remove_signal_watch()
            self.filesrc.set_property("location", None)
        except (ImportError, AttributeError):
            pass

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        if self._finished.is_set():
            return None

        data = bytes()
        while len(data) < num_bytes:
            packet = self.queue.get()
            if packet == self.sentinal:
                self._finished.set()
                break
            data += packet

        if not data:
            return None

        timestamp = self._pipeline.query_position(Gst.Format.TIME).cur / Gst.SECOND
        duration = self.audio_format.bytes_per_second / len(data)

        return AudioData(data, len(data), timestamp, duration, [])

    def seek(self, timestamp):
        # First clear any data in the queue:
        while not self.queue.empty():
            self.queue.get_nowait()

        self._pipeline.seek_simple(Gst.Format.TIME,
                                   Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                                   timestamp * Gst.SECOND)
        self._finished.clear()


def _cleanup():
    # At exist, ensure any remaining Source instances are cleaned up.
    # If this is not done, GStreamer may hang due to dangling callbacks.
    for src in GStreamerSource.source_instances:
        src.delete()


atexit.register(_cleanup)


#########################################
#   Decoder class:
#########################################

class GStreamerDecoder(MediaDecoder):

    def __init__(self):
        Gst.init(None)
        self._glib_loop = _GLibMainLoopThread()

    def get_file_extensions(self):
        return '.mp3', '.flac', '.ogg', '.m4a'

    def decode(self, file, filename, streaming=True):

        if not any(filename.endswith(ext) for ext in self.get_file_extensions()):
            # Refuse to decode anything not specifically listed in the supported
            # file extensions list. This decoder does not yet support video, but
            # it would still decode it and return only the Audio track. This is
            # not desired, since the other decoders will not get a turn. Instead
            # we bail out and let pyglet pass it to the next codec (FFmpeg).
            raise GStreamerDecodeException('Unsupported format.')

        if streaming:
            return GStreamerSource(filename, file)
        else:
            return StaticSource(GStreamerSource(filename, file))


def get_decoders():
    return [GStreamerDecoder()]


def get_encoders():
    return []
