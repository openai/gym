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
"""Wrapper for include/libavutil/avutil.h
"""
from ctypes import c_int, c_uint16, c_int32, c_int64, c_uint32, c_uint64
from ctypes import c_uint8, c_int8, c_uint, c_double, c_float, c_ubyte, c_size_t, c_char
from ctypes import c_char_p, c_void_p, addressof, byref, cast, POINTER, CFUNCTYPE, Structure
from ctypes import Union, create_string_buffer, memmove

import pyglet
import pyglet.lib

avutil = pyglet.lib.load_library(
    'avutil',
    win32='avutil-56',
    darwin='avutil.56'
)

AVMEDIA_TYPE_UNKNOWN = -1
AVMEDIA_TYPE_VIDEO = 0
AVMEDIA_TYPE_AUDIO = 1
AVMEDIA_TYPE_DATA = 2
AVMEDIA_TYPE_SUBTITLE = 3
AVMEDIA_TYPE_ATTACHMENT = 4
AVMEDIA_TYPE_NB = 5

AV_SAMPLE_FMT_U8 = 0
AV_SAMPLE_FMT_S16 = 1
AV_SAMPLE_FMT_S32 = 2
AV_SAMPLE_FMT_FLT = 3
AV_SAMPLE_FORMAT_DOUBLE = 4
AV_SAMPLE_FMT_U8P = 5
AV_SAMPLE_FMT_S16P = 6
AV_SAMPLE_FMT_S32P = 7
AV_SAMPLE_FMT_FLTP = 8
AV_SAMPLE_FMT_DBLP = 9
AV_SAMPLE_FMT_S64 = 10
AV_SAMPLE_FMT_S64P = 11

AV_NUM_DATA_POINTERS = 8

AV_PIX_FMT_RGB24 = 2
AV_PIX_FMT_ARGB = 25
AV_PIX_FMT_RGBA = 26


class AVBuffer(Structure):
    _fields_ = [
        ('data', POINTER(c_uint8)),
        ('size', c_int),
        # .. more
    ]


class AVBufferRef(Structure):
    _fields_ = [
        ('buffer', POINTER(AVBuffer)),
        ('data', POINTER(c_uint8)),
        ('size', c_int)
    ]


class AVDictionaryEntry(Structure):
    _fields_ = [
        ('key', c_char_p),
        ('value', c_char_p)
    ]


class AVDictionary(Structure):
    _fields_ = [
        ('count', c_int),
        ('elems', POINTER(AVDictionaryEntry))
    ]


class AVClass(Structure):
    pass


class AVRational(Structure):
    _fields_ = [
        ('num', c_int),
        ('den', c_int)
    ]


class AVFrameSideData(Structure):
    pass


class AVFrame(Structure):
    _fields_ = [
        ('data', POINTER(c_uint8) * AV_NUM_DATA_POINTERS),
        ('linesize', c_int * AV_NUM_DATA_POINTERS),
        ('extended_data', POINTER(POINTER(c_uint8))),
        ('width', c_int),
        ('height', c_int),
        ('nb_samples', c_int),
        ('format', c_int),
        ('key_frame', c_int),
        ('pict_type', c_int),
        ('sample_aspect_ratio', AVRational),
        ('pts', c_int64),
        ('pkt_pts', c_int64),  # Deprecated
        ('pkt_dts', c_int64),
        ('coded_picture_number', c_int),
        ('display_picture_number', c_int),
        ('quality', c_int),
        ('opaque', c_void_p),
        ('error', c_uint64 * AV_NUM_DATA_POINTERS),  # Deprecated
        ('repeat_pict', c_int),
        ('interlaced_frame', c_int),
        ('top_field_first', c_int),
        ('palette_has_changed', c_int),
        ('reordered_opaque', c_int64),
        ('sample_rate', c_int),
        ('channel_layout', c_uint64),
        ('buf', POINTER(AVBufferRef) * AV_NUM_DATA_POINTERS),
        ('extended_buf', POINTER(POINTER(AVBufferRef))),
        ('nb_extended_buf', c_int),
        ('side_data', POINTER(POINTER(AVFrameSideData))),
        ('nb_side_data', c_int),
        ('flags', c_int),
        ('color_range', c_int),
        ('color_primaries', c_int),
        ('color_trc', c_int),
        ('colorspace', c_int),
        ('chroma_location', c_int),
        ('best_effort_timestamp', c_int64),
        ('pkt_pos', c_int64),
        ('pkt_duration', c_int64),
        # !
        ('metadata', POINTER(AVDictionary)),
        ('decode_error_flags', c_int),
        ('channels', c_int),
        ('pkt_size', c_int),
        ('qscale_table', POINTER(c_int8)),  # Deprecated
        ('qstride', c_int),  # Deprecated
        ('qscale_type', c_int),  # Deprecated
        ('qp_table_buf', POINTER(AVBufferRef)),  # Deprecated
        ('hw_frames_ctx', POINTER(AVBufferRef)),
        ('opaque_ref', POINTER(AVBufferRef)),
        ('crop_top', c_size_t),  # video frames only
        ('crop_bottom', c_size_t),  # video frames only
        ('crop_left', c_size_t),  # video frames only
        ('crop_right', c_size_t),  # video frames only
        ('private_ref', POINTER(AVBufferRef)),
    ]


AV_NOPTS_VALUE = -0x8000000000000000
AV_TIME_BASE = 1000000
AV_TIME_BASE_Q = AVRational(1, AV_TIME_BASE)

avutil.av_version_info.restype = c_char_p
avutil.av_dict_get.restype = POINTER(AVDictionaryEntry)
avutil.av_dict_get.argtypes = [POINTER(AVDictionary),
                               c_char_p, POINTER(AVDictionaryEntry),
                               c_int]
avutil.av_rescale_q.restype = c_int64
avutil.av_rescale_q.argtypes = [c_int64, AVRational, AVRational]
avutil.av_samples_get_buffer_size.restype = c_int
avutil.av_samples_get_buffer_size.argtypes = [POINTER(c_int),
                                              c_int, c_int, c_int]
avutil.av_frame_alloc.restype = POINTER(AVFrame)
avutil.av_frame_free.argtypes = [POINTER(POINTER(AVFrame))]
avutil.av_get_default_channel_layout.restype = c_int64
avutil.av_get_default_channel_layout.argtypes = [c_int]
avutil.av_get_bytes_per_sample.restype = c_int
avutil.av_get_bytes_per_sample.argtypes = [c_int]
avutil.av_strerror.restype = c_int
avutil.av_strerror.argtypes = [c_int, c_char_p, c_size_t]
avutil.av_frame_get_best_effort_timestamp.restype = c_int64
avutil.av_frame_get_best_effort_timestamp.argtypes = [POINTER(AVFrame)]
avutil.av_image_fill_arrays.restype = c_int
avutil.av_image_fill_arrays.argtypes = [POINTER(c_uint8) * 4, c_int * 4,
                                        POINTER(c_uint8), c_int, c_int, c_int, c_int]
avutil.av_dict_set.restype = c_int
avutil.av_dict_set.argtypes = [POINTER(POINTER(AVDictionary)),
                               c_char_p, c_char_p, c_int]
avutil.av_dict_free.argtypes = [POINTER(POINTER(AVDictionary))]
avutil.av_log_set_level.restype = c_int
avutil.av_log_set_level.argtypes = [c_uint]

__all__ = [
    'avutil',
    'AVMEDIA_TYPE_UNKNOWN',
    'AVMEDIA_TYPE_VIDEO',
    'AVMEDIA_TYPE_AUDIO',
    'AVMEDIA_TYPE_DATA',
    'AVMEDIA_TYPE_SUBTITLE',
    'AVMEDIA_TYPE_ATTACHMENT',
    'AVMEDIA_TYPE_NB',
    'AV_SAMPLE_FMT_U8',
    'AV_SAMPLE_FMT_S16',
    'AV_SAMPLE_FMT_S32',
    'AV_SAMPLE_FMT_FLT',
    'AV_SAMPLE_FORMAT_DOUBLE',
    'AV_SAMPLE_FMT_U8P',
    'AV_SAMPLE_FMT_S16P',
    'AV_SAMPLE_FMT_S32P',
    'AV_SAMPLE_FMT_FLTP',
    'AV_SAMPLE_FMT_DBLP',
    'AV_SAMPLE_FMT_S64',
    'AV_SAMPLE_FMT_S64P',
    'AV_NUM_DATA_POINTERS',
    'AV_PIX_FMT_RGB24',
    'AV_PIX_FMT_ARGB',
    'AV_PIX_FMT_RGBA',
    'AV_NOPTS_VALUE',
    'AV_TIME_BASE',
    'AV_TIME_BASE_Q',
    'AVFrame',
    'AVRational',
    'AVDictionary',
]
