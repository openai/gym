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
"""Wrapper for include/libavformat/avformat.h
"""
from ctypes import c_int, c_uint16, c_int32, c_int64, c_uint32, c_uint64
from ctypes import c_uint8, c_uint, c_double, c_float, c_ubyte, c_size_t, c_char, c_char_p
from ctypes import c_void_p, addressof, byref, cast, POINTER, CFUNCTYPE, Structure, Union
from ctypes import create_string_buffer, memmove

import pyglet
import pyglet.lib
from . import libavcodec
from . import libavutil

avformat = pyglet.lib.load_library(
    'avformat',
    win32='avformat-58',
    darwin='avformat.58'
)

AVSEEK_FLAG_BACKWARD = 1  # ///< seek backward
AVSEEK_FLAG_BYTE = 2  # ///< seeking based on position in bytes
AVSEEK_FLAG_ANY = 4  # ///< seek to any frame, even non-keyframes
AVSEEK_FLAG_FRAME = 8  # ///< seeking based on frame number

MAX_REORDER_DELAY = 16


class AVPacketList(Structure): pass


class AVInputFormat(Structure):
    _fields_ = [
        ('name', c_char_p)
    ]


class AVOutputFormat(Structure):
    pass


class AVIOContext(Structure):
    pass


class AVIndexEntry(Structure):
    pass


class AVStreamInfo(Structure):
    _fields_ = [
        ('last_dts', c_int64),
        ('duration_gcd', c_int64),
        ('duration_count', c_int),
        ('rfps_duration_sum', c_int64),
        ('duration_error', POINTER(c_double * 2 * (30 * 12 + 30 + 3 + 6))),
        ('codec_info_duration', c_int64),
        ('codec_info_duration_fields', c_int64),
        ('frame_delay_evidence', c_int),
        ('found_decoder', c_int),
        ('last_duration', c_int64),
        ('fps_first_dts', c_int64),
        ('fps_first_dts_idx', c_int),
        ('fps_last_dts', c_int64),
        ('fps_last_dts_idx', c_int),
    ]


class AVProbeData(Structure):
    _fields_ = [
        ('filename', c_char_p),
        ('buf', POINTER(c_ubyte)),
        ('buf_size', c_int),
        ('mime_type', c_char_p)
    ]


class FFFrac(Structure):
    pass


class AVStreamInternal(Structure):
    pass


class AVFrac(Structure):
    _fields_ = [
        ('val', c_int64),
        ('num', c_int64),
        ('den', c_int64),
    ]


AVCodecContext = libavcodec.AVCodecContext
AVPacketSideData = libavcodec.AVPacketSideData
AVPacket = libavcodec.AVPacket
AVCodecParserContext = libavcodec.AVCodecParserContext
AVCodecParameters = libavcodec.AVCodecParameters
AVRational = libavutil.AVRational
AVDictionary = libavutil.AVDictionary
AVFrame = libavutil.AVFrame


class AVStream(Structure):
    _fields_ = [
        ('index', c_int),
        ('id', c_int),
        ('codec', POINTER(AVCodecContext)),
        ('priv_data', c_void_p),
        ('time_base', AVRational),
        ('start_time', c_int64),
        ('duration', c_int64),
        ('nb_frames', c_int64),
        ('disposition', c_int),
        ('discard', c_int),
        ('sample_aspect_ratio', AVRational),
        ('metadata', POINTER(AVDictionary)),
        ('avg_frame_rate', AVRational),
        ('attached_pic', AVPacket),
        ('side_data', POINTER(AVPacketSideData)),
        ('nb_side_data', c_int),
        ('event_flags', c_int),
        ('r_frame_rate', AVRational),
        ('recommended_encoder_configuration', c_char_p),
        ('codecpar', POINTER(AVCodecParameters)),
        ('info', POINTER(AVStreamInfo)),
        ('pts_wrap_bits', c_int),
        ('first_dts', c_int64),
        ('cur_dts', c_int64),
        ('last_IP_pts', c_int64),
        ('last_IP_duration', c_int),
        ('probe_packets', c_int),
        ('codec_info_nb_frames', c_int),
        ('need_parsing', c_int),
        ('parser', POINTER(AVCodecParserContext)),
        ('last_in_packet_buffer', POINTER(AVPacketList)),
        ('probe_data', AVProbeData),
        ('pts_buffer', c_int64 * (MAX_REORDER_DELAY + 1)),
        ('index_entries', POINTER(AVIndexEntry)),
        ('nb_index_entries', c_int),
        ('index_entries_allocated_size', c_uint),
        ('stream_identifier', c_int),
        ('interleaver_chunk_size', c_int64),
        ('interleaver_chunk_duration', c_int64),
        ('request_probe', c_int),
        ('skip_to_keyframe', c_int),
        ('skip_samples', c_int),
        ('start_skip_samples', c_int64),
        ('first_discard_sample', c_int64),
        ('last_discard_sample', c_int64),
        ('nb_decoded_frames', c_int),
        ('mux_ts_offset', c_int64),
        ('pts_wrap_reference', c_int64),
        ('pts_wrap_behavior', c_int),
        ('update_initial_durations_done', c_int),
        ('pts_reorder_error', c_int64 * (MAX_REORDER_DELAY + 1)),
        ('pts_reorder_error_count', c_uint8 * (MAX_REORDER_DELAY + 1)),
        ('last_dts_for_order_check', c_int64),
        ('dts_ordered', c_uint8),
        ('dts_misordered', c_uint8),
        ('inject_global_side_data', c_int),
        ('display_aspect_ratio', AVRational),
        ('internal', POINTER(AVStreamInternal))
    ]


class AVProgram(Structure):
    pass


class AVChapter(Structure):
    pass


class AVFormatInternal(Structure):
    pass


class AVIOInterruptCB(Structure):
    _fields_ = [
        ('callback', CFUNCTYPE(c_int, c_void_p)),
        ('opaque', c_void_p)
    ]


AVClass = libavutil.AVClass
AVCodec = libavcodec.AVCodec


class AVFormatContext(Structure):
    pass


AVFormatContext._fields_ = [
    ('av_class', POINTER(AVClass)),
    ('iformat', POINTER(AVInputFormat)),
    ('oformat', POINTER(AVOutputFormat)),
    ('priv_data', c_void_p),
    ('pb', POINTER(AVIOContext)),
    ('ctx_flags', c_int),
    ('nb_streams', c_uint),
    ('streams', POINTER(POINTER(AVStream))),
    ('filename', c_char * 1024),  # Deprecated
    ('url', c_char_p),
    ('start_time', c_int64),
    ('duration', c_int64),
    ('bit_rate', c_int64),
    ('packet_size', c_uint),
    ('max_delay', c_int),
    ('flags', c_int),
    ('probesize', c_int64),
    ('max_analyze_duration', c_int64),
    ('key', POINTER(c_uint8)),
    ('keylen', c_int),
    ('nb_programs', c_uint),
    ('programs', POINTER(POINTER(AVProgram))),
    ('video_codec_id', c_int),
    ('audio_codec_id', c_int),
    ('subtitle_codec_id', c_int),
    ('max_index_size', c_uint),
    ('max_picture_buffer', c_uint),
    ('nb_chapters', c_uint),
    ('chapters', POINTER(POINTER(AVChapter))),
    ('metadata', POINTER(AVDictionary)),
    ('start_time_realtime', c_int64),
    ('fps_probe_size', c_int),
    ('error_recognition', c_int),
    ('interrupt_callback', AVIOInterruptCB),
    ('debug', c_int),
    ('max_interleave_delta', c_int64),
    ('strict_std_compliance', c_int),
    ('event_flags', c_int),
    ('max_ts_probe', c_int),
    ('avoid_negative_ts', c_int),
    ('ts_id', c_int),
    ('audio_preload', c_int),
    ('max_chunk_duration', c_int),
    ('max_chunk_size', c_int),
    ('use_wallclock_as_timestamps', c_int),
    ('avio_flags', c_int),
    ('duration_estimation_method', c_uint),
    ('skip_initial_bytes', c_int64),
    ('correct_ts_overflow', c_uint),
    ('seek2any', c_int),
    ('flush_packets', c_int),
    ('probe_score', c_int),
    ('format_probesize', c_int),
    ('codec_whitelist', c_char_p),
    ('format_whitelist', c_char_p),
    ('internal', POINTER(AVFormatInternal)),
    ('io_repositioned', c_int),
    ('video_codec', POINTER(AVCodec)),
    ('audio_codec', POINTER(AVCodec)),
    ('subtitle_codec', POINTER(AVCodec)),
    ('data_codec', POINTER(AVCodec)),
    ('metadata_header_padding', c_int),
    ('opaque', c_void_p),
    ('control_message_cb', CFUNCTYPE(c_int,
                                     POINTER(AVFormatContext), c_int, c_void_p,
                                     c_size_t)),
    ('output_ts_offset', c_int64),
    ('dump_separator', POINTER(c_uint8)),
    ('data_codec_id', c_int),
    # ! one more in here?
    ('protocol_whitelist', c_char_p),
    ('io_open', CFUNCTYPE(c_int,
                          POINTER(AVFormatContext),
                          POINTER(POINTER(AVIOContext)),
                          c_char_p, c_int,
                          POINTER(POINTER(AVDictionary)))),
    ('io_close', CFUNCTYPE(None,
                           POINTER(AVFormatContext), POINTER(AVIOContext))),
    ('protocol_blacklist', c_char_p),
    ('max_streams', c_int)
]

avformat.av_register_all.restype = None
avformat.av_find_input_format.restype = c_int
avformat.av_find_input_format.argtypes = [c_int]
avformat.avformat_open_input.restype = c_int
avformat.avformat_open_input.argtypes = [
    POINTER(POINTER(AVFormatContext)),
    c_char_p,
    POINTER(AVInputFormat),
    POINTER(POINTER(AVDictionary))]
avformat.avformat_find_stream_info.restype = c_int
avformat.avformat_find_stream_info.argtypes = [
    POINTER(AVFormatContext),
    POINTER(POINTER(AVDictionary))]
avformat.avformat_close_input.restype = None
avformat.avformat_close_input.argtypes = [
    POINTER(POINTER(AVFormatContext))]
avformat.av_read_frame.restype = c_int
avformat.av_read_frame.argtypes = [POINTER(AVFormatContext),
                                   POINTER(AVPacket)]
avformat.av_seek_frame.restype = c_int
avformat.av_seek_frame.argtypes = [POINTER(AVFormatContext),
                                   c_int, c_int64, c_int]
avformat.avformat_seek_file.restype = c_int
avformat.avformat_seek_file.argtypes = [POINTER(AVFormatContext),
                                        c_int, c_int64, c_int64, c_int64, c_int]
avformat.av_guess_frame_rate.restype = AVRational
avformat.av_guess_frame_rate.argtypes = [POINTER(AVFormatContext),
                                         POINTER(AVStream), POINTER(AVFrame)]

__all__ = [
    'avformat',
    'AVSEEK_FLAG_BACKWARD',
    'AVSEEK_FLAG_BYTE',
    'AVSEEK_FLAG_ANY',
    'AVSEEK_FLAG_FRAME',
    'AVFormatContext'
]
