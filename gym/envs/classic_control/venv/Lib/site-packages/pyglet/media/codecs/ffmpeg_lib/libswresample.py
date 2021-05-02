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
"""Wrapper for include/libswresample/swresample.h
"""
from ctypes import c_int, c_uint16, c_int32, c_int64, c_uint32, c_uint64
from ctypes import c_uint8, c_uint, c_double, c_float, c_ubyte, c_size_t, c_char, c_char_p
from ctypes import c_void_p, addressof, byref, cast, POINTER, CFUNCTYPE, Structure, Union
from ctypes import create_string_buffer, memmove

import pyglet
import pyglet.lib

swresample = pyglet.lib.load_library(
    'swresample',
    win32='swresample-3',
    darwin='swresample.3'
)

SWR_CH_MAX = 32


class SwrContext(Structure):
    pass


swresample.swr_alloc_set_opts.restype = POINTER(SwrContext)
swresample.swr_alloc_set_opts.argtypes = [POINTER(SwrContext),
                                          c_int64, c_int, c_int, c_int64,
                                          c_int, c_int, c_int, c_void_p]
swresample.swr_init.restype = c_int
swresample.swr_init.argtypes = [POINTER(SwrContext)]
swresample.swr_free.argtypes = [POINTER(POINTER(SwrContext))]
swresample.swr_convert.restype = c_int
swresample.swr_convert.argtypes = [POINTER(SwrContext),
                                   POINTER(c_uint8) * SWR_CH_MAX,
                                   c_int,
                                   POINTER(POINTER(c_uint8)),
                                   c_int]
swresample.swr_set_compensation.restype = c_int
swresample.swr_set_compensation.argtypes = [POINTER(SwrContext),
                                            c_int, c_int]
swresample.swr_get_out_samples.restype = c_int
swresample.swr_get_out_samples.argtypes = [POINTER(SwrContext), c_int]

__all__ = [
    'swresample',
    'SWR_CH_MAX',
    'SwrContext'
]
