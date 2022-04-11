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

"""Wrapper for include/libavcodec/avcodec.h
"""

from ctypes import c_int, c_uint16, c_int32, c_int64, c_uint32, c_uint64
from ctypes import c_uint8, c_uint, c_double, c_float, c_ubyte, c_size_t, c_char, c_char_p
from ctypes import c_void_p, addressof, byref, cast, POINTER, CFUNCTYPE, Structure, Union
from ctypes import create_string_buffer, memmove

import pyglet
import pyglet.lib
from . import libavutil

avcodec = pyglet.lib.load_library(
    'avcodec',
    win32='avcodec-58',
    darwin='avcodec.58'
)

FF_INPUT_BUFFER_PADDING_SIZE = 32


class AVPacketSideData(Structure):
    _fields_ = [
        ('data', POINTER(c_uint8)),
        ('size', c_int),
        ('type', c_int)
    ]


AVBufferRef = libavutil.AVBufferRef


class AVPacket(Structure):
    _fields_ = [
        ('buf', POINTER(AVBufferRef)),
        ('pts', c_int64),
        ('dts', c_int64),
        ('data', POINTER(c_uint8)),
        ('size', c_int),
        ('stream_index', c_int),
        ('flags', c_int),
        ('side_data', POINTER(AVPacketSideData)),
        ('side_data_elems', c_int),
        ('duration', c_int64),
        ('pos', c_int64),
        ('convergence_duration', c_int64)  # Deprecated
    ]


class AVCodecParserContext(Structure):
    pass


AVRational = libavutil.AVRational


class AVCodecParameters(Structure):
    _fields_ = [
        ('codec_type', c_int),
        ('codec_id', c_int),
        ('codec_tag', c_uint32),
        ('extradata', POINTER(c_uint8)),
        ('extradata_size', c_int),
        ('format', c_int),
        ('bit_rate', c_int64),
        ('bits_per_coded_sample', c_int),
        ('bits_per_raw_sample', c_int),
        ('profile', c_int),
        ('level', c_int),
        ('width', c_int),
        ('height', c_int),
        ('sample_aspect_ratio', AVRational),
        ('field_order', c_int),
        ('color_range', c_int),
        ('color_primaries', c_int),
        ('color_trc', c_int),
        ('color_space', c_int),
        ('chroma_location', c_int),
        ('video_delay', c_int),
        ('channel_layout', c_uint64),
        ('channels', c_int),
        ('sample_rate', c_int),
        ('block_align', c_int),
        ('frame_size', c_int),
        ('initial_padding', c_int),
        ('trailing_padding', c_int),
        ('seek_preroll', c_int),
    ]


class AVProfile(Structure):
    _fields_ = [
        ('profile', c_int),
        ('name', c_char_p),
    ]


class AVCodecDescriptor(Structure):
    _fields_ = [
        ('id', c_int),
        ('type', c_int),
        ('name', c_char_p),
        ('long_name', c_char_p),
        ('props', c_int),
        ('mime_types', c_char_p),
        ('profiles', POINTER(AVProfile))
    ]


class AVCodecInternal(Structure):
    pass


class AVCodec(Structure):
    _fields_ = [
        ('name', c_char_p),
        ('long_name', c_char_p),
        ('type', c_int),
        ('id', c_int),
        ('capabilities', c_int),
        ('supported_framerates', POINTER(AVRational)),
        ('pix_fmts', POINTER(c_int)),
        ('supported_samplerates', POINTER(c_int)),
        ('sample_fmts', POINTER(c_int)),
        ('channel_layouts', POINTER(c_uint64)),
        ('max_lowres', c_uint8),
        # And more...
    ]


class AVCodecContext(Structure):
    pass


class RcOverride(Structure):
    pass


class AVHWAccel(Structure):
    pass


AVClass = libavutil.AVClass
AVFrame = libavutil.AVFrame
AV_NUM_DATA_POINTERS = libavutil.AV_NUM_DATA_POINTERS
AVCodecContext._fields_ = [
    ('av_class', POINTER(AVClass)),
    ('log_level_offset', c_int),
    ('codec_type', c_int),
    ('codec', POINTER(AVCodec)),
    ('codec_id', c_int),
    ('codec_tag', c_uint),
    ('priv_data', c_void_p),
    ('internal', POINTER(AVCodecInternal)),
    ('opaque', c_void_p),
    ('bit_rate', c_int64),
    ('bit_rate_tolerance', c_int),
    ('global_quality', c_int),
    ('compression_level', c_int),
    ('flags', c_int),
    ('flags2', c_int),
    ('extradata', POINTER(c_uint8)),
    ('extradata_size', c_int),
    ('time_base', AVRational),
    ('ticks_per_frame', c_int),
    ('delay', c_int),
    ('width', c_int),
    ('height', c_int),
    ('coded_width', c_int),
    ('coded_height', c_int),
    ('gop_size', c_int),
    ('pix_fmt', c_int),
    ('draw_horiz_band', CFUNCTYPE(None,
                                  POINTER(AVCodecContext), POINTER(AVFrame),
                                  c_int * 8, c_int, c_int, c_int)),
    ('get_format', CFUNCTYPE(c_int, POINTER(AVCodecContext), POINTER(c_int))),
    ('max_b_frames', c_int),
    ('b_quant_factor', c_float),
    ('b_frame_strategy', c_int),  # Deprecated
    ('b_quant_offset', c_float),
    ('has_b_frames', c_int),
    ('mpeg_quant', c_int),  # Deprecated
    ('i_quant_factor', c_float),
    ('i_quant_offset', c_float),
    ('lumi_masking', c_float),
    ('temporal_cplx_masking', c_float),
    ('spatial_cplx_masking', c_float),
    ('p_masking', c_float),
    ('dark_masking', c_float),
    ('slice_count', c_int),
    ('prediction_method', c_int),  # Deprecated
    ('slice_offset', POINTER(c_int)),
    ('sample_aspect_ratio', AVRational),
    ('me_cmp', c_int),
    ('me_sub_cmp', c_int),
    ('mb_cmp', c_int),
    ('ildct_cmp', c_int),
    ('dia_size', c_int),
    ('last_predictor_count', c_int),
    ('pre_me', c_int),  # Deprecated
    ('me_pre_cmp', c_int),
    ('pre_dia_size', c_int),
    ('me_subpel_quality', c_int),
    ('me_range', c_int),
    ('slice_flags', c_int),
    ('mb_decision', c_int),
    ('intra_matrix', POINTER(c_uint16)),
    ('inter_matrix', POINTER(c_uint16)),
    ('scenechange_threshold', c_int),  # Deprecated
    ('noise_reduction', c_int),  # Deprecated
    ('intra_dc_precision', c_int),
    ('skip_top', c_int),
    ('skip_bottom', c_int),
    ('mb_lmin', c_int),
    ('mb_lmax', c_int),
    ('me_penalty_compensation', c_int),  # Deprecated
    ('bidir_refine', c_int),
    ('brd_scale', c_int),  # Deprecated
    ('keyint_min', c_int),
    ('refs', c_int),
    ('chromaoffset', c_int),  # Deprecated
    ('mv0_threshold', c_int),
    ('b_sensitivity', c_int),  # Deprecated
    ('color_primaries', c_int),
    ('color_trc', c_int),
    ('colorspace', c_int),
    ('color_range', c_int),
    ('chroma_sample_location', c_int),
    ('slices', c_int),
    ('field_order', c_int),
    ('sample_rate', c_int),
    ('channels', c_int),
    ('sample_fmt', c_int),
    ('frame_size', c_int),
    ('frame_number', c_int),
    ('block_align', c_int),
    ('cutoff', c_int),
    ('channel_layout', c_uint64),
    ('request_channel_layout', c_uint64),
    ('audio_service_type', c_int),
    ('request_sample_fmt', c_int),
    ('get_buffer2', CFUNCTYPE(c_int, POINTER(AVCodecContext), POINTER(AVFrame), c_int)),
    ('refcounted_frames', c_int),  # Deprecated
    ('qcompress', c_float),
    ('qblur', c_float),
    ('qmin', c_int),
    ('qmax', c_int),
    ('max_qdiff', c_int),
    ('rc_buffer_size', c_int),
    ('rc_override_count', c_int),
    ('rc_override', POINTER(RcOverride)),
    ('rc_max_rate', c_int64),
    ('rc_min_rate', c_int64),
    ('rc_max_available_vbv_use', c_float),
    ('rc_min_vbv_overflow_use', c_float),
    ('rc_initial_buffer_occupancy', c_int),
    ('coder_type', c_int),  # Deprecated
    ('context_model', c_int),  # Deprecated
    ('frame_skip_threshold', c_int),  # Deprecated
    ('frame_skip_factor', c_int),  # Deprecated
    ('frame_skip_exp', c_int),  # Deprecated
    ('frame_skip_cmp', c_int),  # Deprecated
    ('trellis', c_int),
    ('min_prediction_order', c_int),  # Deprecated
    ('max_prediction_order', c_int),  # Deprecated
    ('timecode_frame_start', c_int64),  # Deprecated
    ('rtp_callback', CFUNCTYPE(None,  # Deprecated
                               POINTER(AVCodecContext), c_void_p, c_int, c_int)),
    ('rtp_payload_size', c_int),  # Deprecated
    ('mv_bits', c_int),  # Deprecated
    ('header_bits', c_int),  # Deprecated
    ('i_tex_bits', c_int),  # Deprecated
    ('p_tex_bits', c_int),  # Deprecated
    ('i_count', c_int),  # Deprecated
    ('p_count', c_int),  # Deprecated
    ('skip_count', c_int),  # Deprecated
    ('misc_bits', c_int),  # Deprecated
    ('frame_bits', c_int),  # Deprecated
    ('stats_out', c_char_p),
    ('stats_in', c_char_p),
    ('workaround_bugs', c_int),
    ('strict_std_compliance', c_int),
    ('error_concealment', c_int),
    ('debug', c_int),
    ('err_recognition', c_int),
    ('reordered_opaque', c_int64),
    ('hwaccel', POINTER(AVHWAccel)),
    ('hwaccel_context', c_void_p),
    ('error', c_uint64 * AV_NUM_DATA_POINTERS),
    ('dct_algo', c_int),
    ('idct_algo', c_int),
    ('bits_per_coded_sample', c_int),
    ('bits_per_raw_sample', c_int),
    ('lowres', c_int),
    ('coded_frame', POINTER(AVFrame)),  # Deprecated
    ('thread_count', c_int),
    ('thread_type', c_int),
    ('active_thread_type', c_int),
    ('thread_safe_callbacks', c_int),
    ('execute', CFUNCTYPE(c_int,
                          POINTER(AVCodecContext),
                          CFUNCTYPE(c_int, POINTER(AVCodecContext), c_void_p),
                          c_void_p, c_int, c_int, c_int)),
    ('execute2', CFUNCTYPE(c_int,
                           POINTER(AVCodecContext),
                           CFUNCTYPE(c_int, POINTER(AVCodecContext), c_void_p, c_int, c_int),
                           c_void_p, c_int, c_int)),
    ('nsse_weight', c_int),
    ('profile', c_int),
    ('level', c_int),
    ('skip_loop_filter', c_int),
    ('skip_idct', c_int),
    ('skip_frame', c_int),
    ('subtitle_header', POINTER(c_uint8)),
    ('subtitle_header_size', c_int),
    ('vbv_delay', c_uint64),  # Deprecated
    ('side_data_only_packets', c_int),  # Deprecated
    ('initial_padding', c_int),
    ('framerate', AVRational),
    # !
    ('sw_pix_fmt', c_int),
    ('pkt_timebase', AVRational),
    ('codec_dexcriptor', AVCodecDescriptor),
    ('pts_correction_num_faulty_pts', c_int64),
    ('pts_correction_num_faulty_dts', c_int64),
    ('pts_correction_last_pts', c_int64),
    ('pts_correction_last_dts', c_int64),
    ('sub_charenc', c_char_p),
    ('sub_charenc_mode', c_int),
    ('skip_alpha', c_int),
    ('seek_preroll', c_int),
    ('debug_mv', c_int),
    ('chroma_intra_matrix', POINTER(c_uint16)),
    ('dump_separator', POINTER(c_uint8)),
    ('codec_whitelist', c_char_p),
    ('properties', c_uint),
    ('coded_side_data', POINTER(AVPacketSideData)),
    ('nb_coded_side_data', c_int),
    ('hw_frames_ctx', POINTER(AVBufferRef)),
    ('sub_text_format', c_int),
    ('trailing_padding', c_int),
    ('max_pixels', c_int64),
    ('hw_device_ctx', POINTER(AVBufferRef)),
    ('hwaccel_flags', c_int),
    ('apply_cropping', c_int),
    ('extra_hw_frames', c_int)

]

AV_CODEC_ID_VP8 = 139
AV_CODEC_ID_VP9 = 167

avcodec.av_packet_unref.argtypes = [POINTER(AVPacket)]
avcodec.av_packet_free.argtypes = [POINTER(POINTER(AVPacket))]
avcodec.av_packet_clone.restype = POINTER(AVPacket)
avcodec.av_packet_clone.argtypes = [POINTER(AVPacket)]
avcodec.av_packet_move_ref.argtypes = [POINTER(AVPacket), POINTER(AVPacket)]
avcodec.avcodec_find_decoder.restype = POINTER(AVCodec)
avcodec.avcodec_find_decoder.argtypes = [c_int]
AVDictionary = libavutil.AVDictionary
avcodec.avcodec_open2.restype = c_int
avcodec.avcodec_open2.argtypes = [POINTER(AVCodecContext),
                                  POINTER(AVCodec),
                                  POINTER(POINTER(AVDictionary))]
avcodec.avcodec_free_context.argtypes = [POINTER(POINTER(AVCodecContext))]
avcodec.av_packet_alloc.restype = POINTER(AVPacket)
avcodec.av_init_packet.argtypes = [POINTER(AVPacket)]
avcodec.avcodec_decode_audio4.restype = c_int
avcodec.avcodec_decode_audio4.argtypes = [POINTER(AVCodecContext),
                                          POINTER(AVFrame), POINTER(c_int),
                                          POINTER(AVPacket)]
avcodec.avcodec_decode_video2.restype = c_int
avcodec.avcodec_decode_video2.argtypes = [POINTER(AVCodecContext),
                                          POINTER(AVFrame), POINTER(c_int),
                                          POINTER(AVPacket)]
avcodec.avcodec_flush_buffers.argtypes = [POINTER(AVCodecContext)]
avcodec.avcodec_alloc_context3.restype = POINTER(AVCodecContext)
avcodec.avcodec_alloc_context3.argtypes = [POINTER(AVCodec)]
avcodec.avcodec_free_context.argtypes = [POINTER(POINTER(AVCodecContext))]
avcodec.avcodec_parameters_to_context.restype = c_int
avcodec.avcodec_parameters_to_context.argtypes = [POINTER(AVCodecContext),
                                                  POINTER(AVCodecParameters)]
avcodec.avcodec_get_name.restype = c_char_p
avcodec.avcodec_get_name.argtypes = [c_int]
avcodec.avcodec_find_decoder_by_name.restype = POINTER(AVCodec)
avcodec.avcodec_find_decoder_by_name.argtypes = [c_char_p]

__all__ = [
    'avcodec',
    'FF_INPUT_BUFFER_PADDING_SIZE',
    'AVPacket',
    'AVCodecContext',
    'AV_CODEC_ID_VP8',
    'AV_CODEC_ID_VP9',
]
