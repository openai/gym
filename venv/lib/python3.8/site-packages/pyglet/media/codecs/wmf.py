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
import os
import platform
import warnings

from pyglet import com, image
from pyglet.util import debug_print
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.media import Source, MediaDecodeException
from pyglet.media.codecs import AudioFormat, AudioData, VideoFormat, MediaDecoder, StaticSource

_debug = debug_print('debug_media')

try:
    mfreadwrite = 'mfreadwrite'
    mfplat = 'mfplat'

    # System32 and SysWOW64 folders are opposite perception in Windows x64.
    # System32 = x64 dll's | SysWOW64 = x86 dlls
    # By default ctypes only seems to look in system32 regardless of Python architecture, which has x64 dlls.
    if platform.architecture()[0] == '32bit':
        if platform.machine().endswith('64'):  # Machine is 64 bit, Python is 32 bit.
            mfreadwrite = os.path.join(os.environ['WINDIR'], 'SysWOW64', 'mfreadwrite.dll')
            mfplat = os.path.join(os.environ['WINDIR'], 'SysWOW64', 'mfplat.dll')

    mfreadwrite_lib = ctypes.windll.LoadLibrary(mfreadwrite)
    mfplat_lib = ctypes.windll.LoadLibrary(mfplat)
except OSError:
    # Doesn't exist? Should stop import of library.
    raise ImportError('Could not load WMF library.')

MF_SOURCE_READERF_ERROR = 0x00000001
MF_SOURCE_READERF_ENDOFSTREAM = 0x00000002
MF_SOURCE_READERF_NEWSTREAM = 0x00000004
MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED = 0x00000010
MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED = 0x00000020
MF_SOURCE_READERF_STREAMTICK = 0x00000100

# Audio attributes
MF_LOW_LATENCY = com.GUID(0x9c27891a, 0xed7a, 0x40e1, 0x88, 0xe8, 0xb2, 0x27, 0x27, 0xa0, 0x24, 0xee)

# Audio information
MF_MT_ALL_SAMPLES_INDEPENDENT = com.GUID(0xc9173739, 0x5e56, 0x461c, 0xb7, 0x13, 0x46, 0xfb, 0x99, 0x5c, 0xb9, 0x5f)
MF_MT_FIXED_SIZE_SAMPLES = com.GUID(0xb8ebefaf, 0xb718, 0x4e04, 0xb0, 0xa9, 0x11, 0x67, 0x75, 0xe3, 0x32, 0x1b)
MF_MT_SAMPLE_SIZE = com.GUID(0xdad3ab78, 0x1990, 0x408b, 0xbc, 0xe2, 0xeb, 0xa6, 0x73, 0xda, 0xcc, 0x10)
MF_MT_COMPRESSED = com.GUID(0x3afd0cee, 0x18f2, 0x4ba5, 0xa1, 0x10, 0x8b, 0xea, 0x50, 0x2e, 0x1f, 0x92)
MF_MT_WRAPPED_TYPE = com.GUID(0x4d3f7b23, 0xd02f, 0x4e6c, 0x9b, 0xee, 0xe4, 0xbf, 0x2c, 0x6c, 0x69, 0x5d)
MF_MT_AUDIO_NUM_CHANNELS = com.GUID(0x37e48bf5, 0x645e, 0x4c5b, 0x89, 0xde, 0xad, 0xa9, 0xe2, 0x9b, 0x69, 0x6a)
MF_MT_AUDIO_SAMPLES_PER_SECOND = com.GUID(0x5faeeae7, 0x0290, 0x4c31, 0x9e, 0x8a, 0xc5, 0x34, 0xf6, 0x8d, 0x9d, 0xba)
MF_MT_AUDIO_FLOAT_SAMPLES_PER_SECOND = com.GUID(0xfb3b724a, 0xcfb5, 0x4319, 0xae, 0xfe, 0x6e, 0x42, 0xb2, 0x40, 0x61, 0x32)
MF_MT_AUDIO_AVG_BYTES_PER_SECOND = com.GUID(0x1aab75c8, 0xcfef, 0x451c, 0xab, 0x95, 0xac, 0x03, 0x4b, 0x8e, 0x17, 0x31)
MF_MT_AUDIO_BLOCK_ALIGNMENT = com.GUID(0x322de230, 0x9eeb, 0x43bd, 0xab, 0x7a, 0xff, 0x41, 0x22, 0x51, 0x54, 0x1d)
MF_MT_AUDIO_BITS_PER_SAMPLE = com.GUID(0xf2deb57f, 0x40fa, 0x4764, 0xaa, 0x33, 0xed, 0x4f, 0x2d, 0x1f, 0xf6, 0x69)
MF_MT_AUDIO_VALID_BITS_PER_SAMPLE = com.GUID(0xd9bf8d6a, 0x9530, 0x4b7c, 0x9d, 0xdf, 0xff, 0x6f, 0xd5, 0x8b, 0xbd, 0x06)
MF_MT_AUDIO_SAMPLES_PER_BLOCK = com.GUID(0xaab15aac, 0xe13a, 0x4995, 0x92, 0x22, 0x50, 0x1e, 0xa1, 0x5c, 0x68, 0x77)
MF_MT_AUDIO_CHANNEL_MASK = com.GUID(0x55fb5765, 0x644a, 0x4caf, 0x84, 0x79, 0x93, 0x89, 0x83, 0xbb, 0x15, 0x88)
MF_PD_DURATION = com.GUID(0x6c990d33, 0xbb8e, 0x477a, 0x85, 0x98, 0xd, 0x5d, 0x96, 0xfc, 0xd8, 0x8a)


# Media types categories
MF_MT_MAJOR_TYPE = com.GUID(0x48eba18e, 0xf8c9, 0x4687, 0xbf, 0x11, 0x0a, 0x74, 0xc9, 0xf9, 0x6a, 0x8f)
MF_MT_SUBTYPE = com.GUID(0xf7e34c9a, 0x42e8, 0x4714, 0xb7, 0x4b, 0xcb, 0x29, 0xd7, 0x2c, 0x35, 0xe5)

# Major types
MFMediaType_Audio = com.GUID(0x73647561, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71)
MFMediaType_Video = com.GUID(0x73646976, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71)
MFMediaType_Protected = com.GUID(0x7b4b6fe6, 0x9d04, 0x4494, 0xbe, 0x14, 0x7e, 0x0b, 0xd0, 0x76, 0xc8, 0xe4)
MFMediaType_Image = com.GUID(0x72178C23, 0xE45B, 0x11D5, 0xBC, 0x2A, 0x00, 0xB0, 0xD0, 0xF3, 0xF4, 0xAB)
MFMediaType_HTML = com.GUID(0x72178C24, 0xE45B, 0x11D5, 0xBC, 0x2A, 0x00, 0xB0, 0xD0, 0xF3, 0xF4, 0xAB)
MFMediaType_Subtitle = com.GUID(0xa6d13581, 0xed50, 0x4e65, 0xae, 0x08, 0x26, 0x06, 0x55, 0x76, 0xaa, 0xcc)

# Video subtypes, attributes, and enums (Uncompressed)
D3DFMT_X8R8G8B8 = 22
D3DFMT_P8 = 41
D3DFMT_A8R8G8B8 = 21
MFVideoFormat_RGB32 = com.GUID(D3DFMT_X8R8G8B8, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71)
MFVideoFormat_RGB8 = com.GUID(D3DFMT_P8, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71)
MFVideoFormat_ARGB32 = com.GUID(D3DFMT_A8R8G8B8, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71)

MFVideoInterlace_Progressive = 2
MF_MT_INTERLACE_MODE = com.GUID(0xe2724bb8, 0xe676, 0x4806, 0xb4, 0xb2, 0xa8, 0xd6, 0xef, 0xb4, 0x4c, 0xcd)
MF_MT_FRAME_SIZE = com.GUID(0x1652c33d, 0xd6b2, 0x4012, 0xb8, 0x34, 0x72, 0x03, 0x08, 0x49, 0xa3, 0x7d)
MF_MT_FRAME_RATE = com.GUID(0xc459a2e8, 0x3d2c, 0x4e44, 0xb1, 0x32, 0xfe, 0xe5, 0x15, 0x6c, 0x7b, 0xb0)
MF_MT_PIXEL_ASPECT_RATIO = com.GUID(0xc6376a1e, 0x8d0a, 0x4027, 0xbe, 0x45, 0x6d, 0x9a, 0x0a, 0xd3, 0x9b, 0xb6)
MF_MT_DRM_FLAGS = com.GUID(0x8772f323, 0x355a, 0x4cc7, 0xbb, 0x78, 0x6d, 0x61, 0xa0, 0x48, 0xae, 0x82)
MF_MT_DEFAULT_STRIDE = com.GUID(0x644b4e48, 0x1e02, 0x4516, 0xb0, 0xeb, 0xc0, 0x1c, 0xa9, 0xd4, 0x9a, 0xc6)

# Audio Subtypes (Uncompressed)
WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3
MFAudioFormat_PCM = com.GUID(WAVE_FORMAT_PCM, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71)
MFAudioFormat_Float = com.GUID(WAVE_FORMAT_IEEE_FLOAT, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71)

# Image subtypes.
MFImageFormat_RGB32 = com.GUID(0x00000016, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71)
MFImageFormat_JPEG = com.GUID(0x19e4a5aa, 0x5662, 0x4fc5, 0xa0, 0xc0, 0x17, 0x58, 0x02, 0x8e, 0x10, 0x57)

# Video attributes
# Enables hardware decoding
MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS = com.GUID(0xa634a91c, 0x822b, 0x41b9, 0xa4, 0x94, 0x4d, 0xe4, 0x64, 0x36, 0x12,
                                                   0xb0)
# Enable video decoding
MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING = com.GUID(0xfb394f3d, 0xccf1, 0x42ee, 0xbb, 0xb3, 0xf9, 0xb8, 0x45, 0xd5,
                                                    0x68, 0x1d)
MF_SOURCE_READER_D3D_MANAGER = com.GUID(0xec822da2, 0xe1e9, 0x4b29, 0xa0, 0xd8, 0x56, 0x3c, 0x71, 0x9f, 0x52, 0x69)
MF_MEDIA_ENGINE_DXGI_MANAGER = com.GUID(0x065702da, 0x1094, 0x486d, 0x86, 0x17, 0xee, 0x7c, 0xc4, 0xee, 0x46, 0x48)
MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING = com.GUID(0xf81da2c, 0xb537, 0x4672, 0xa8, 0xb2, 0xa6, 0x81, 0xb1,
                                                             0x73, 0x7, 0xa3)

# Some common errors
MF_E_INVALIDSTREAMNUMBER = -1072875853  # 0xC00D36B3
MF_E_UNSUPPORTED_BYTESTREAM_TYPE = -1072875836  # 0xC00D36C4
MF_E_NO_MORE_TYPES = 0xC00D36B9
MF_E_TOPO_CODEC_NOT_FOUND = -1072868846  # 0xC00D5212


VT_I8 = 20  # Only enum we care about: https://docs.microsoft.com/en-us/windows/win32/api/wtypes/ne-wtypes-varenum


def timestamp_from_wmf(timestamp):  # 100-nanoseconds
    return float(timestamp) / 10000000


def timestamp_to_wmf(timestamp):  # 100-nanoseconds
    return int(timestamp * 10000000)


class IMFAttributes(com.pIUnknown):
    _methods_ = [
        ('GetItem',
         com.STDMETHOD()),
        ('GetItemType',
         com.STDMETHOD()),
        ('CompareItem',
         com.STDMETHOD()),
        ('Compare',
         com.STDMETHOD()),
        ('GetUINT32',
         com.STDMETHOD(com.REFIID, POINTER(c_uint32))),
        ('GetUINT64',
         com.STDMETHOD(com.REFIID, POINTER(c_uint64))),
        ('GetDouble',
         com.STDMETHOD()),
        ('GetGUID',
         com.STDMETHOD(com.REFIID, POINTER(com.GUID))),
        ('GetStringLength',
         com.STDMETHOD()),
        ('GetString',
         com.STDMETHOD()),
        ('GetAllocatedString',
         com.STDMETHOD()),
        ('GetBlobSize',
         com.STDMETHOD()),
        ('GetBlob',
         com.STDMETHOD()),
        ('GetAllocatedBlob',
         com.STDMETHOD()),
        ('GetUnknown',
         com.STDMETHOD()),
        ('SetItem',
         com.STDMETHOD()),
        ('DeleteItem',
         com.STDMETHOD()),
        ('DeleteAllItems',
         com.STDMETHOD()),
        ('SetUINT32',
         com.STDMETHOD(com.REFIID, c_uint32)),
        ('SetUINT64',
         com.STDMETHOD()),
        ('SetDouble',
         com.STDMETHOD()),
        ('SetGUID',
         com.STDMETHOD(com.REFIID, com.REFIID)),
        ('SetString',
         com.STDMETHOD()),
        ('SetBlob',
         com.STDMETHOD()),
        ('SetUnknown',
         com.STDMETHOD(com.REFIID, com.pIUnknown)),
        ('LockStore',
         com.STDMETHOD()),
        ('UnlockStore',
         com.STDMETHOD()),
        ('GetCount',
         com.STDMETHOD()),
        ('GetItemByIndex',
         com.STDMETHOD()),
        ('CopyAllItems',
         com.STDMETHOD(c_void_p)),  # IMFAttributes
    ]


class IMFMediaBuffer(com.pIUnknown):
    _methods_ = [
        ('Lock',
         com.STDMETHOD(POINTER(POINTER(BYTE)), POINTER(DWORD), POINTER(DWORD))),
        ('Unlock',
         com.STDMETHOD()),
        ('GetCurrentLength',
         com.STDMETHOD(POINTER(DWORD))),
        ('SetCurrentLength',
         com.STDMETHOD(DWORD)),
        ('GetMaxLength',
         com.STDMETHOD(POINTER(DWORD)))
    ]


class IMFSample(IMFAttributes, com.pIUnknown):
    _methods_ = [
        ('GetSampleFlags',
         com.STDMETHOD()),
        ('SetSampleFlags',
         com.STDMETHOD()),
        ('GetSampleTime',
         com.STDMETHOD()),
        ('SetSampleTime',
         com.STDMETHOD()),
        ('GetSampleDuration',
         com.STDMETHOD(POINTER(c_ulonglong))),
        ('SetSampleDuration',
         com.STDMETHOD(DWORD, IMFMediaBuffer)),
        ('GetBufferCount',
         com.STDMETHOD(POINTER(DWORD))),
        ('GetBufferByIndex',
         com.STDMETHOD(DWORD, IMFMediaBuffer)),
        ('ConvertToContiguousBuffer',
         com.STDMETHOD(POINTER(IMFMediaBuffer))),  # out
        ('AddBuffer',
         com.STDMETHOD(POINTER(DWORD))),
        ('RemoveBufferByIndex',
         com.STDMETHOD()),
        ('RemoveAllBuffers',
         com.STDMETHOD()),
        ('GetTotalLength',
         com.STDMETHOD(POINTER(DWORD))),
        ('CopyToBuffer',
         com.STDMETHOD()),
    ]


class IMFMediaType(IMFAttributes, com.pIUnknown):
    _methods_ = [
        ('GetMajorType',
         com.STDMETHOD()),
        ('IsCompressedFormat',
         com.STDMETHOD()),
        ('IsEqual',
         com.STDMETHOD()),
        ('GetRepresentation',
         com.STDMETHOD()),
        ('FreeRepresentation',
         com.STDMETHOD()),
    ]


class IMFByteStream(com.pIUnknown):
    _methods_ = [
        ('GetCapabilities',
         com.STDMETHOD()),
        ('GetLength',
         com.STDMETHOD()),
        ('SetLength',
         com.STDMETHOD()),
        ('GetCurrentPosition',
         com.STDMETHOD()),
        ('SetCurrentPosition',
         com.STDMETHOD(c_ulonglong)),
        ('IsEndOfStream',
         com.STDMETHOD()),
        ('Read',
         com.STDMETHOD()),
        ('BeginRead',
         com.STDMETHOD()),
        ('EndRead',
         com.STDMETHOD()),
        ('Write',
         com.STDMETHOD(POINTER(BYTE), ULONG, POINTER(ULONG))),
        ('BeginWrite',
         com.STDMETHOD()),
        ('EndWrite',
         com.STDMETHOD()),
        ('Seek',
         com.STDMETHOD()),
        ('Flush',
         com.STDMETHOD()),
        ('Close',
         com.STDMETHOD()),
    ]


class IMFSourceReader(com.pIUnknown):
    _methods_ = [
        ('GetStreamSelection',
         com.STDMETHOD(DWORD, POINTER(BOOL))),  # in, out
        ('SetStreamSelection',
         com.STDMETHOD(DWORD, BOOL)),
        ('GetNativeMediaType',
         com.STDMETHOD(DWORD, DWORD, POINTER(IMFMediaType))),
        ('GetCurrentMediaType',
         com.STDMETHOD(DWORD, POINTER(IMFMediaType))),
        ('SetCurrentMediaType',
         com.STDMETHOD(DWORD, POINTER(DWORD), IMFMediaType)),
        ('SetCurrentPosition',
         com.STDMETHOD(com.REFIID, POINTER(PROPVARIANT))),
        ('ReadSample',
         com.STDMETHOD(DWORD, DWORD, POINTER(DWORD), POINTER(DWORD), POINTER(c_longlong), POINTER(IMFSample))),
        ('Flush',
         com.STDMETHOD(DWORD)),  # in
        ('GetServiceForStream',
         com.STDMETHOD()),
        ('GetPresentationAttribute',
         com.STDMETHOD(DWORD, com.REFIID, POINTER(PROPVARIANT))),
    ]


class WAVEFORMATEX(ctypes.Structure):
    _fields_ = [
        ('wFormatTag', WORD),
        ('nChannels', WORD),
        ('nSamplesPerSec', DWORD),
        ('nAvgBytesPerSec', DWORD),
        ('nBlockAlign', WORD),
        ('wBitsPerSample', WORD),
        ('cbSize', WORD),
    ]

    def __repr__(self):
        return 'WAVEFORMATEX(wFormatTag={}, nChannels={}, nSamplesPerSec={}, nAvgBytesPersec={}' \
               ', nBlockAlign={}, wBitsPerSample={}, cbSize={})'.format(
            self.wFormatTag, self.nChannels, self.nSamplesPerSec,
            self.nAvgBytesPerSec, self.nBlockAlign, self.wBitsPerSample,
            self.cbSize)


# Stream constants
MF_SOURCE_READER_ALL_STREAMS = 0xfffffffe
MF_SOURCE_READER_ANY_STREAM = 4294967294  # 0xfffffffe
MF_SOURCE_READER_FIRST_AUDIO_STREAM = 4294967293  # 0xfffffffd
MF_SOURCE_READER_FIRST_VIDEO_STREAM = 0xfffffffc
MF_SOURCE_READER_MEDIASOURCE = 0xffffffff

# Version calculation
if WINDOWS_7_OR_GREATER:
    MF_SDK_VERSION = 0x0002
else:
    MF_SDK_VERSION = 0x0001

MF_API_VERSION = 0x0070  # Only used in Vista.

MF_VERSION = (MF_SDK_VERSION << 16 | MF_API_VERSION)

MFStartup = mfplat_lib.MFStartup
MFStartup.restype = HRESULT
MFStartup.argtypes = [LONG, DWORD]

MFShutdown = mfplat_lib.MFShutdown
MFShutdown.restype = HRESULT
MFShutdown.argtypes = []

MFCreateAttributes = mfplat_lib.MFCreateAttributes
MFCreateAttributes.restype = HRESULT
MFCreateAttributes.argtypes = [POINTER(IMFAttributes), c_uint32]  # attributes, cInitialSize

MFCreateSourceReaderFromURL = mfreadwrite_lib.MFCreateSourceReaderFromURL
MFCreateSourceReaderFromURL.restype = HRESULT
MFCreateSourceReaderFromURL.argtypes = [LPCWSTR, IMFAttributes, POINTER(IMFSourceReader)]

MFCreateSourceReaderFromByteStream = mfreadwrite_lib.MFCreateSourceReaderFromByteStream
MFCreateSourceReaderFromByteStream.restype = HRESULT
MFCreateSourceReaderFromByteStream.argtypes = [IMFByteStream, IMFAttributes, POINTER(IMFSourceReader)]

if WINDOWS_7_OR_GREATER:
    MFCreateMFByteStreamOnStream = mfplat_lib.MFCreateMFByteStreamOnStream
    MFCreateMFByteStreamOnStream.restype = HRESULT
    MFCreateMFByteStreamOnStream.argtypes = [c_void_p, POINTER(IMFByteStream)]

MFCreateTempFile = mfplat_lib.MFCreateTempFile
MFCreateTempFile.restype = HRESULT
MFCreateTempFile.argtypes = [UINT, UINT, UINT, POINTER(IMFByteStream)]

MFCreateMediaType = mfplat_lib.MFCreateMediaType
MFCreateMediaType.restype = HRESULT
MFCreateMediaType.argtypes = [POINTER(IMFMediaType)]

MFCreateWaveFormatExFromMFMediaType = mfplat_lib.MFCreateWaveFormatExFromMFMediaType
MFCreateWaveFormatExFromMFMediaType.restype = HRESULT
MFCreateWaveFormatExFromMFMediaType.argtypes = [IMFMediaType, POINTER(POINTER(WAVEFORMATEX)), POINTER(c_uint32), c_uint32]


class WMFSource(Source):
    low_latency = True  # Quicker latency but possible quality loss.

    decode_audio = True
    decode_video = True

    def __init__(self, filename, file=None):
        assert any([self.decode_audio, self.decode_video]), "Source must decode audio, video, or both, not none."
        self._current_audio_sample = None
        self._current_audio_buffer = None
        self._current_video_sample = None
        self._current_video_buffer = None
        self._timestamp = 0
        self._attributes = None
        self._stream_obj = None
        self._imf_bytestream = None
        self._wfx = None
        self._stride = None

        self.set_config_attributes()

        # Create SourceReader
        self._source_reader = IMFSourceReader()

        # If it's a file, we need to load it as a stream.
        if file is not None:
            data = file.read()

            self._imf_bytestream = IMFByteStream()

            data_len = len(data)

            if WINDOWS_7_OR_GREATER:
                # Stole code from GDIPlus for older IStream support.
                hglob = kernel32.GlobalAlloc(GMEM_MOVEABLE, data_len)
                ptr = kernel32.GlobalLock(hglob)
                ctypes.memmove(ptr, data, data_len)
                kernel32.GlobalUnlock(hglob)

                # Create IStream
                self._stream_obj = com.pIUnknown()
                ole32.CreateStreamOnHGlobal(hglob, True, ctypes.byref(self._stream_obj))

                # MFCreateMFByteStreamOnStreamEx for future async operations exists, however Windows 8+ only. Requires new interface
                # (Also unsure how/if new Windows async functions and callbacks work with ctypes.)
                MFCreateMFByteStreamOnStream(self._stream_obj, ctypes.byref(self._imf_bytestream))
            else:
                # Vista does not support MFCreateMFByteStreamOnStream.
                # HACK: Create file in Windows temp folder to write our byte data to.
                # (Will be automatically deleted when IMFByteStream is Released.)
                MFCreateTempFile(MF_ACCESSMODE_READWRITE,
                                 MF_OPENMODE_DELETE_IF_EXIST,
                                 MF_FILEFLAGS_NONE,
                                 ctypes.byref(self._imf_bytestream))

                wrote_length = ULONG()
                data_ptr = cast(data, POINTER(BYTE))
                self._imf_bytestream.Write(data_ptr, data_len, ctypes.byref(wrote_length))
                self._imf_bytestream.SetCurrentPosition(0)

                if wrote_length.value != data_len:
                    raise MediaDecodeException("Could not write all of the data to the bytestream file.")

            try:
                MFCreateSourceReaderFromByteStream(self._imf_bytestream, self._attributes, ctypes.byref(self._source_reader))
            except OSError as err:
                raise MediaDecodeException(err) from None
        else:
            # We can just load from filename if no file object specified..
            try:
                MFCreateSourceReaderFromURL(filename, self._attributes, ctypes.byref(self._source_reader))
            except OSError as err:
                raise MediaDecodeException(err) from None

        if self.decode_audio:
            self._load_audio()

        if self.decode_video:
            self._load_video()

        assert self.audio_format or self.video_format, "Source was decoded, but no video or audio streams were found."

        # Get duration of the media file after everything has been ok to decode.
        try:
            prop = PROPVARIANT()
            self._source_reader.GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE,
                                                         ctypes.byref(MF_PD_DURATION),
                                                         ctypes.byref(prop))

            self._duration = timestamp_from_wmf(prop.llVal)
            ole32.PropVariantClear(ctypes.byref(prop))
        except OSError:
            warnings.warn("Could not determine duration of media file: '{}'.".format(filename))

    def _load_audio(self, stream=MF_SOURCE_READER_FIRST_AUDIO_STREAM):
        """ Prepares the audio stream for playback by detecting if it's compressed and attempting to decompress to PCM.
            Default: Only get the first available audio stream.
        """
        # Will be an audio file.
        self._audio_stream_index = stream

        # Get what the native/real media type is (audio only)
        imfmedia = IMFMediaType()

        try:
            self._source_reader.GetNativeMediaType(self._audio_stream_index, 0, ctypes.byref(imfmedia))
        except OSError as err:
            if err.winerror == MF_E_INVALIDSTREAMNUMBER:
                assert _debug('WMFAudioDecoder: No audio stream found.')
            return

        # Get Major media type (Audio, Video, etc)
        # TODO: Make GUID take no arguments for a null version:
        guid_audio_type = com.GUID(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        imfmedia.GetGUID(MF_MT_MAJOR_TYPE, ctypes.byref(guid_audio_type))

        if guid_audio_type == MFMediaType_Audio:
            assert _debug('WMFAudioDecoder: Found Audio Stream.')

            # Deselect any other streams if we don't need them. (Small speedup)
            if not self.decode_video:
                self._source_reader.SetStreamSelection(MF_SOURCE_READER_ANY_STREAM, False)

            # Select first audio stream.
            self._source_reader.SetStreamSelection(MF_SOURCE_READER_FIRST_AUDIO_STREAM, True)

            # Check sub media type, AKA what kind of codec
            guid_compressed = com.GUID(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            imfmedia.GetGUID(MF_MT_SUBTYPE, ctypes.byref(guid_compressed))

            if guid_compressed == MFAudioFormat_PCM or guid_compressed == MFAudioFormat_Float:
                assert _debug('WMFAudioDecoder: Found Uncompressed Audio:', guid_compressed)
            else:
                assert _debug('WMFAudioDecoder: Found Compressed Audio:', guid_compressed)
                # If audio is compressed, attempt to decompress it by forcing source reader to use PCM
                mf_mediatype = IMFMediaType()

                MFCreateMediaType(ctypes.byref(mf_mediatype))
                mf_mediatype.SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio)
                mf_mediatype.SetGUID(MF_MT_SUBTYPE, MFAudioFormat_PCM)

                try:
                    self._source_reader.SetCurrentMediaType(self._audio_stream_index, None, mf_mediatype)
                except OSError as err:  # Can't decode codec.
                    raise MediaDecodeException(err) from None

            # Current media type should now be properly decoded at this point.
            decoded_media_type = IMFMediaType()  # Maybe reusing older IMFMediaType will work?
            self._source_reader.GetCurrentMediaType(self._audio_stream_index, ctypes.byref(decoded_media_type))

            wfx_length = ctypes.c_uint32()
            wfx = POINTER(WAVEFORMATEX)()

            MFCreateWaveFormatExFromMFMediaType(decoded_media_type,
                                                ctypes.byref(wfx),
                                                ctypes.byref(wfx_length),
                                                0)

            self._wfx = wfx.contents
            self.audio_format = AudioFormat(channels=self._wfx.nChannels,
                                            sample_size=self._wfx.wBitsPerSample,
                                            sample_rate=self._wfx.nSamplesPerSec)
        else:
            assert _debug('WMFAudioDecoder: Audio stream not found')

    def get_format(self):
        """Returns the WAVEFORMATEX data which has more information thah audio_format"""
        return self._wfx

    def _load_video(self, stream=MF_SOURCE_READER_FIRST_VIDEO_STREAM):
        self._video_stream_index = stream

        # Get what the native/real media type is (video only)
        imfmedia = IMFMediaType()

        try:
            self._source_reader.GetCurrentMediaType(self._video_stream_index, ctypes.byref(imfmedia))
        except OSError as err:
            if err.winerror == MF_E_INVALIDSTREAMNUMBER:
                assert _debug('WMFVideoDecoder: No video stream found.')
            return

        assert _debug('WMFVideoDecoder: Found Video Stream')

        # All video is basically compressed, try to decompress.
        uncompressed_mt = IMFMediaType()
        MFCreateMediaType(ctypes.byref(uncompressed_mt))

        imfmedia.CopyAllItems(uncompressed_mt)

        imfmedia.Release()

        uncompressed_mt.SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32)
        uncompressed_mt.SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive)
        uncompressed_mt.SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, 1)

        try:
            self._source_reader.SetCurrentMediaType(self._video_stream_index, None, uncompressed_mt)
        except OSError as err:  # Can't decode codec.
            raise MediaDecodeException(err) from None

        height, width = self._get_attribute_size(uncompressed_mt, MF_MT_FRAME_SIZE)

        self.video_format = VideoFormat(width=width, height=height)
        assert _debug('WMFVideoDecoder: Frame width: {} height: {}'.format(width, height))

        # Frame rate
        den, num = self._get_attribute_size(uncompressed_mt, MF_MT_FRAME_RATE)
        self.video_format.frame_rate = num / den
        assert _debug('WMFVideoDecoder: Frame Rate: {} / {} = {}'.format(num, den, self.video_format.frame_rate))

        # Sometimes it can return negative? Variable bit rate? Needs further tests and examples.
        if self.video_format.frame_rate < 0:
            self.video_format.frame_rate = 30000 / 1001
            assert _debug('WARNING: Negative frame rate, attempting to use default, but may experience issues.')

        # Pixel ratio
        den, num = self._get_attribute_size(uncompressed_mt, MF_MT_PIXEL_ASPECT_RATIO)
        self.video_format.sample_aspect = num / den
        assert _debug('WMFVideoDecoder: Pixel Ratio: {} / {} = {}'.format(num, den, self.video_format.sample_aspect))

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        flags = DWORD()
        timestamp = ctypes.c_longlong()
        audio_data_length = DWORD()

        # If we have an audio sample already in use and we call this again, release the memory of buffer and sample.
        # Can only release after the data is played or else glitches and pops can be heard.
        if self._current_audio_sample:
            self._current_audio_buffer.Release()
            self._current_audio_sample.Release()

        self._current_audio_sample = IMFSample()
        self._current_audio_buffer = IMFMediaBuffer()

        while True:
            self._source_reader.ReadSample(self._audio_stream_index, 0, None, ctypes.byref(flags),
                                           ctypes.byref(timestamp), ctypes.byref(self._current_audio_sample))

            if flags.value & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED:
                assert _debug('WMFAudioDecoder: Data is no longer valid.')
                break

            if flags.value & MF_SOURCE_READERF_ENDOFSTREAM:
                assert _debug('WMFAudioDecoder: End of data from stream source.')
                break

            if not self._current_audio_sample:
                assert _debug('WMFAudioDecoder: No sample.')
                continue

            # Convert to single buffer as a sample could potentially(rarely) have multiple buffers.
            self._current_audio_sample.ConvertToContiguousBuffer(ctypes.byref(self._current_audio_buffer))

            audio_data_ptr = POINTER(BYTE)()

            self._current_audio_buffer.Lock(ctypes.byref(audio_data_ptr), None, ctypes.byref(audio_data_length))
            self._current_audio_buffer.Unlock()

            audio_data = create_string_buffer(audio_data_length.value)
            memmove(audio_data, audio_data_ptr, audio_data_length.value)

            return AudioData(audio_data,
                             audio_data_length.value,
                             timestamp_from_wmf(timestamp.value),
                             audio_data_length.value / self.audio_format.sample_rate,
                             [])

        return None

    def get_next_video_frame(self, skip_empty_frame=True):
        video_data_length = DWORD()
        flags = DWORD()
        timestamp = ctypes.c_longlong()

        if self._current_video_sample:
            self._current_video_buffer.Release()
            self._current_video_sample.Release()

        self._current_video_sample = IMFSample()
        self._current_video_buffer = IMFMediaBuffer()

        while True:
            self._source_reader.ReadSample(self._video_stream_index, 0, None, ctypes.byref(flags),
                                           ctypes.byref(timestamp), ctypes.byref(self._current_video_sample))

            if flags.value & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED:
                assert _debug('WMFVideoDecoder: Data is no longer valid.')

                # Get Major media type (Audio, Video, etc)
                new = IMFMediaType()
                self._source_reader.GetCurrentMediaType(self._video_stream_index, ctypes.byref(new))

                # Sometimes this happens once. I think this only
                # changes if the stride is added/changed before playback?
                stride = ctypes.c_uint32()
                new.GetUINT32(MF_MT_DEFAULT_STRIDE, ctypes.byref(stride))

                self._stride = stride.value

            if flags.value & MF_SOURCE_READERF_ENDOFSTREAM:
                self._timestamp = None
                assert _debug('WMFVideoDecoder: End of data from stream source.')
                break

            if not self._current_video_sample:
                assert _debug('WMFVideoDecoder: No sample.')
                continue

            self._current_video_buffer = IMFMediaBuffer()

            # Convert to single buffer as a sample could potentially have multiple buffers.
            self._current_video_sample.ConvertToContiguousBuffer(ctypes.byref(self._current_video_buffer))

            video_data = POINTER(BYTE)()

            self._current_video_buffer.Lock(ctypes.byref(video_data), None, ctypes.byref(video_data_length))

            width = self.video_format.width
            height = self.video_format.height

            # buffer = ctypes.create_string_buffer(size)
            self._timestamp = timestamp_from_wmf(timestamp.value)

            self._current_video_buffer.Unlock()

            # This is made with the assumption that the video frame will be blitted into the player texture immediately
            # after, and then cleared next frame attempt.
            return image.ImageData(width, height, 'BGRA', video_data, self._stride)

        return None

    def get_next_video_timestamp(self):
        return self._timestamp

    def seek(self, timestamp):
        timestamp = min(timestamp, self._duration) if self._duration else timestamp

        prop = PROPVARIANT()
        prop.vt = VT_I8
        prop.llVal = timestamp_to_wmf(timestamp)

        pos_com = com.GUID(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        try:
            self._source_reader.SetCurrentPosition(pos_com, prop)
        except OSError as err:
            warnings.warn(str(err))

        ole32.PropVariantClear(ctypes.byref(prop))

    @staticmethod
    def _get_attribute_size(attributes, guidKey):
        """ Convert int64 attributes to int32"""  # HI32/LOW32

        size = ctypes.c_uint64()
        attributes.GetUINT64(guidKey, size)
        lParam = size.value

        x = ctypes.c_int32(lParam).value
        y = ctypes.c_int32(lParam >> 32).value
        return x, y

    def set_config_attributes(self):
        """ Here we set user specified attributes, by default we try to set low latency mode. (Win7+)"""
        if self.low_latency or self.decode_video:
            self._attributes = IMFAttributes()

            MFCreateAttributes(ctypes.byref(self._attributes), 3)

        if self.low_latency and WINDOWS_7_OR_GREATER:
            self._attributes.SetUINT32(ctypes.byref(MF_LOW_LATENCY), 1)

            assert _debug('WMFAudioDecoder: Setting configuration attributes.')

        # If it's a video we need to enable the streams to be accessed.
        if self.decode_video:
            self._attributes.SetUINT32(ctypes.byref(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS), 1)
            self._attributes.SetUINT32(ctypes.byref(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING), 1)

            assert _debug('WMFVideoDecoder: Setting configuration attributes.')

    def __del__(self):
        if self._stream_obj:
            self._stream_obj.Release()

        if self._imf_bytestream:
            self._imf_bytestream.Release()

        if self._current_audio_sample:
            self._current_audio_buffer.Release()
            self._current_audio_sample.Release()

        if self._current_video_sample:
            self._current_video_buffer.Release()
            self._current_video_sample.Release()


#########################################
#   Decoder class:
#########################################

class WMFDecoder(MediaDecoder):
    def __init__(self):

        self.ole32 = None
        self.MFShutdown = None

        try:
            # Coinitialize supposed to be called for COMs?
            ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
        except OSError as err:
            warnings.warn(str(err))

        try:
            MFStartup(MF_VERSION, 0)
        except OSError as err:
            raise ImportError('WMF could not startup:', err.strerror)

        self.extensions = self._build_decoder_extensions()

        self.ole32 = ole32
        self.MFShutdown = MFShutdown

        assert _debug('Windows Media Foundation: Initialized.')

    @staticmethod
    def _build_decoder_extensions():
        """Extension support varies depending on OS version."""
        extensions = []
        if WINDOWS_VISTA_OR_GREATER:
            extensions.extend(['.asf', '.wma', '.wmv',
                               '.mp3',
                               '.sami', '.smi',
                               ])

        if WINDOWS_7_OR_GREATER:
            extensions.extend(['.3g2', '.3gp', '.3gp2', '.3gp',
                               '.aac', '.adts',
                               '.avi',
                               '.m4a', '.m4v', '.mov', '.mp4',
                               # '.wav'  # Can do wav, but we have a WAVE decoder.
                               ])

        if WINDOWS_10_ANNIVERSARY_UPDATE_OR_GREATER:
            extensions.extend(['.mkv', '.flac', '.ogg'])

        return extensions

    def get_file_extensions(self):
        return self.extensions

    def decode(self, file, filename, streaming=True):
        if streaming:
            return WMFSource(filename, file)
        else:
            return StaticSource(WMFSource(filename, file))

    def __del__(self):
        if self.MFShutdown is not None:
            self.MFShutdown()
        if self.ole32 is not None:
            self.ole32.CoUninitialize()


def get_decoders():
    return [WMFDecoder()]


def get_encoders():
    return []
