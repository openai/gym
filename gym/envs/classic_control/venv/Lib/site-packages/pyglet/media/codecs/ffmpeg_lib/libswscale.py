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
"""Wrapper for include/libswscale/swscale.h
"""
from ctypes import c_int, c_uint16, c_int32, c_int64, c_uint32, c_uint64
from ctypes import c_uint8, c_uint, c_double, c_float, c_ubyte, c_size_t, c_char, c_char_p
from ctypes import c_void_p, addressof, byref, cast, POINTER, CFUNCTYPE, Structure, Union
from ctypes import create_string_buffer, memmove

import pyglet
import pyglet.lib

swscale = pyglet.lib.load_library(
    'swscale',
    win32='swscale-5',
    darwin='swscale.5'
)

SWS_FAST_BILINEAR = 1


class SwsContext(Structure):
    pass


class SwsFilter(Structure):
    pass


swscale.sws_getCachedContext.restype = POINTER(SwsContext)
swscale.sws_getCachedContext.argtypes = [POINTER(SwsContext),
                                         c_int, c_int, c_int, c_int,
                                         c_int, c_int, c_int,
                                         POINTER(SwsFilter), POINTER(SwsFilter),
                                         POINTER(c_double)]
swscale.sws_freeContext.argtypes = [POINTER(SwsContext)]
swscale.sws_scale.restype = c_int
swscale.sws_scale.argtypes = [POINTER(SwsContext),
                              POINTER(POINTER(c_uint8)),
                              POINTER(c_int),
                              c_int, c_int,
                              POINTER(POINTER(c_uint8)),
                              POINTER(c_int)]

__all__ = [
    'swscale',
    'SWS_FAST_BILINEAR',
    'SwsContext',
    'SwsFilter'
]
