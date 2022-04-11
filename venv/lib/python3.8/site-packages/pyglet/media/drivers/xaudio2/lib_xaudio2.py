from pyglet.media.events import MediaEvent

import pyglet
import ctypes
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet import com
import platform
import os
from pyglet.util import debug_print

_debug = debug_print('debug_media')


def load_xaudio2(dll_name):
    """This will attempt to load a version of XAudio2. Versions supported: 2.9, 2.8.
       While Windows 8 ships with 2.8 and Windows 10 ships with version 2.9, it is possible to install 2.9 on 8/8.1.
    """
    xaudio2 = dll_name
    # System32 and SysWOW64 folders are opposite perception in Windows x64.
    # System32 = x64 dll's | SysWOW64 = x86 dlls
    # By default ctypes only seems to look in system32 regardless of Python architecture, which has x64 dlls.
    if platform.architecture()[0] == '32bit':
        if platform.machine().endswith('64'):  # Machine is 64 bit, Python is 32 bit.
            xaudio2 = os.path.join(os.environ['WINDIR'], 'SysWOW64', '{}.dll'.format(xaudio2))

    xaudio2_lib = ctypes.windll.LoadLibrary(xaudio2)

    # Somehow x3d uses different calling structure than the rest of the DLL; Only affects 32 bit? Microsoft...
    x3d_lib = ctypes.cdll.LoadLibrary(xaudio2)
    return xaudio2_lib, x3d_lib


try:
    xaudio2_lib, x3d_lib = load_xaudio2("xaudio2_9")
except OSError:
    _debug("Could not load XAudio2.9 library")
    try:
        xaudio2_lib, x3d_lib = load_xaudio2("xaudio2_8")
    except OSError:
        _debug("Could not load XAudio2.8 library")
        raise ImportError('Could not locate a supported XAudio2 library.')


UINT32 = c_uint32
FLOAT32 = c_float


class XAUDIO2_DEBUG_CONFIGURATION(ctypes.Structure):
    _fields_ = [
        ('TraceMask', UINT32),
        ('BreakMask', UINT32),
        ('LogThreadID', BOOL),
        ('LogFileline', BOOL),
        ('LogFunctionName', BOOL),
        ('LogTiming', BOOL),
    ]


class XAUDIO2_PERFORMANCE_DATA(ctypes.Structure):
    _fields_ = [
        ('AudioCyclesSinceLastQuery', c_uint64),
        ('TotalCyclesSinceLastQuery', c_uint64),
        ('MinimumCyclesPerQuantum', UINT32),
        ('MaximumCyclesPerQuantum', UINT32),
        ('MemoryUsageInBytes', UINT32),
        ('CurrentLatencyInSamples', UINT32),
        ('GlitchesSinceEngineStarted', UINT32),
        ('ActiveSourceVoiceCount', UINT32),
        ('TotalSourceVoiceCount', UINT32),
        ('ActiveSubmixVoiceCount', UINT32),
        ('ActiveResamplerCount', UINT32),
        ('ActiveMatrixMixCount', UINT32),
        ('ActiveXmaSourceVoices', UINT32),
        ('ActiveXmaStreams', UINT32),
    ]

    def __repr__(self):
        return "XAUDIO2PerformanceData(active_voices={}, total_voices={}, glitches={}, latency={} samples, memory_usage={} bytes)".format(self.ActiveSourceVoiceCount, self.TotalSourceVoiceCount, self.GlitchesSinceEngineStarted, self.CurrentLatencyInSamples, self.MemoryUsageInBytes)


class XAUDIO2_VOICE_SENDS(ctypes.Structure):
    _fields_ = [
        ('SendCount', UINT32),
        ('pSends', c_void_p),
    ]


class XAUDIO2_BUFFER(ctypes.Structure):
    _fields_ = [
        ('Flags', UINT32),
        ('AudioBytes', UINT32),
        ('pAudioData', POINTER(c_char)),
        ('PlayBegin', UINT32),
        ('PlayLength', UINT32),
        ('LoopBegin', UINT32),
        ('LoopLength', UINT32),
        ('LoopCount', UINT32),
        ('pContext', c_void_p),
    ]

class XAUDIO2_VOICE_STATE(ctypes.Structure):
    _fields_ = [
        ('pCurrentBufferContext', c_void_p),
        ('BuffersQueued', UINT32),
        ('SamplesPlayed', UINT32)
    ]

    def __repr__(self):
        return "XAUDIO2_VOICE_STATE(BuffersQueued={0}, SamplesPlayed={1})".format(self.BuffersQueued, self.SamplesPlayed)

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

XAUDIO2_USE_DEFAULT_PROCESSOR = 0x00000000  # Win 10+

if WINDOWS_10_ANNIVERSARY_UPDATE_OR_GREATER:
    XAUDIO2_DEFAULT_PROCESSOR = XAUDIO2_USE_DEFAULT_PROCESSOR
else:
    XAUDIO2_DEFAULT_PROCESSOR = 0x00000001  # Windows 8/8.1


XAUDIO2_LOG_ERRORS = 0x0001   # For handled errors with serious effects.
XAUDIO2_LOG_WARNINGS = 0x0002   # For handled errors that may be recoverable.
XAUDIO2_LOG_INFO = 0x0004   # Informational chit-chat (e.g. state changes).
XAUDIO2_LOG_DETAIL = 0x0008   # More detailed chit-chat.
XAUDIO2_LOG_API_CALLS = 0x0010   # Public API function entries and exits.
XAUDIO2_LOG_FUNC_CALLS = 0x0020   # Internal function entries and exits.
XAUDIO2_LOG_TIMING = 0x0040   # Delays detected and other timing data.
XAUDIO2_LOG_LOCKS = 0x0080   # Usage of critical sections and mutexes.
XAUDIO2_LOG_MEMORY = 0x0100   # Memory heap usage information.
XAUDIO2_LOG_STREAMING = 0x1000   # Audio streaming information.


# Some XAUDIO2 global settings, most not used, but useful information
XAUDIO2_MAX_BUFFER_BYTES = 0x80000000   # Maximum bytes allowed in a source buffer
XAUDIO2_MAX_QUEUED_BUFFERS = 64         # Maximum buffers allowed in a voice queue
XAUDIO2_MAX_BUFFERS_SYSTEM = 2          # Maximum buffers allowed for system threads (Xbox 360 only)
XAUDIO2_MAX_AUDIO_CHANNELS = 64         # Maximum channels in an audio stream
XAUDIO2_MIN_SAMPLE_RATE = 1000          # Minimum audio sample rate supported
XAUDIO2_MAX_SAMPLE_RATE = 200000        # Maximum audio sample rate supported
XAUDIO2_MAX_VOLUME_LEVEL = 16777216.0   # Maximum acceptable volume level (2^24)
XAUDIO2_MIN_FREQ_RATIO =  (1/1024.0)    # Minimum SetFrequencyRatio argument
XAUDIO2_MAX_FREQ_RATIO = 1024.0         # Maximum MaxFrequencyRatio argument
XAUDIO2_DEFAULT_FREQ_RATIO  = 2.0       # Default MaxFrequencyRatio argument
XAUDIO2_MAX_FILTER_ONEOVERQ = 1.5       # Maximum XAUDIO2_FILTER_PARAMETERS.OneOverQ
XAUDIO2_MAX_FILTER_FREQUENCY = 1.0      # Maximum XAUDIO2_FILTER_PARAMETERS.Frequency
XAUDIO2_MAX_LOOP_COUNT = 254            # Maximum non-infinite XAUDIO2_BUFFER.LoopCount
XAUDIO2_MAX_INSTANCES = 8               # Maximum simultaneous XAudio2 objects on Xbox 360


XAUDIO2_FILTER_TYPE = UINT
LowPassFilter = 0  # Attenuates frequencies above the cutoff frequency (state-variable filter).
BandPassFilter = 1   # Attenuates frequencies outside a given range      (state-variable filter).
HighPassFilter = 2   # Attenuates frequencies below the cutoff frequency (state-variable filter).
NotchFilter = 3   # Attenuates frequencies inside a given range       (state-variable filter).
LowPassOnePoleFilter = 4   # Attenuates frequencies above the cutoff frequency (one-pole filter, XAUDIO2_FILTER_PARAMETERS.OneOverQ has no effect)
HighPassOnePoleFilter = 5   # Attenuates frequencies below the cutoff frequency (one-pole filter, XAUDIO2_FILTER_PARAMETERS.OneOverQ has no effect)

XAUDIO2_NO_LOOP_REGION = 0  # Used in XAUDIO2_BUFFER.LoopCount
XAUDIO2_LOOP_INFINITE = 255  # Used in XAUDIO2_BUFFER.LoopCount
XAUDIO2_DEFAULT_CHANNELS = 0  # Used in CreateMasteringVoice
XAUDIO2_DEFAULT_SAMPLERATE = 0  # Used in CreateMasteringVoice

WAVE_FORMAT_PCM = 1

XAUDIO2_DEBUG_ENGINE =            0x0001    # Used in XAudio2Create
XAUDIO2_VOICE_NOPITCH =           0x0002    # Used in IXAudio2::CreateSourceVoice
XAUDIO2_VOICE_NOSRC =             0x0004    # Used in IXAudio2::CreateSourceVoice
XAUDIO2_VOICE_USEFILTER =         0x0008    # Used in IXAudio2::CreateSource/SubmixVoice
XAUDIO2_PLAY_TAILS =              0x0020    # Used in IXAudio2SourceVoice::Stop
XAUDIO2_END_OF_STREAM =           0x0040    # Used in XAUDIO2_BUFFER.Flags
XAUDIO2_SEND_USEFILTER =          0x0080    # Used in XAUDIO2_SEND_DESCRIPTOR.Flags
XAUDIO2_VOICE_NOSAMPLESPLAYED =   0x0100    # Used in IXAudio2SourceVoice::GetState
XAUDIO2_STOP_ENGINE_WHEN_IDLE  =  0x2000    # Used in XAudio2Create to force the engine to Stop when no source voices are Started, and Start when a voice is Started
XAUDIO2_1024_QUANTUM =            0x8000    # Used in XAudio2Create to specify nondefault processing quantum of 21.33 ms (1024 samples at 48KHz)
XAUDIO2_NO_VIRTUAL_AUDIO_CLIENT = 0x10000   # Used in CreateMasteringVoice to create a virtual audio client


class IXAudio2VoiceCallback(com.Interface):
    _methods_ = [
        ('OnVoiceProcessingPassStart',
         com.STDMETHOD(UINT32)),
        ('OnVoiceProcessingPassEnd',
         com.STDMETHOD()),
        ('onStreamEnd',
         com.STDMETHOD()),
        ('onBufferStart',
         com.STDMETHOD(ctypes.c_void_p)),
        ('OnBufferEnd',
         com.STDMETHOD(ctypes.c_void_p)),
        ('OnLoopEnd',
         com.STDMETHOD(ctypes.c_void_p)),
    ]


class XA2SourceCallback(com.COMObject):
    """Callback class used to trigger when buffers or streams end..
           WARNING: Whenever a callback is running, XAudio2 cannot generate audio.
           Make sure these functions run as fast as possible and do not block/delay more than a few milliseconds.
           MS Recommendation:
           At a minimum, callback functions must not do the following:
                - Access the hard disk or other permanent storage
                - Make expensive or blocking API calls
                - Synchronize with other parts of client code
                - Require significant CPU usage
    """
    _interfaces_ = [IXAudio2VoiceCallback]

    def __init__(self, xa2_player):
        self.xa2_player = xa2_player

    def OnVoiceProcessingPassStart(self, bytesRequired):
        pass

    def OnVoiceProcessingPassEnd(self):
        pass

    def onStreamEnd(self):
        pass

    def onBufferStart(self, pBufferContext):
        pass

    def OnBufferEnd(self, pBufferContext):
        """At the end of playing one buffer, attempt to refill again.
        Even if the player is out of sources, it needs to be called to purge all buffers.
        """
        if self.xa2_player:
            self.xa2_player.refill_source_player()

    def OnLoopEnd(self, this, pBufferContext):
        pass

    def onVoiceError(self, this, pBufferContext, hresult):
        raise Exception("Error occurred during audio playback.", hresult)


class XAUDIO2_EFFECT_DESCRIPTOR(Structure):
    _fields_ = [
        ('pEffect', com.pIUnknown),
        ('InitialState', c_bool),
        ('OutputChannels', UINT32)
    ]


class XAUDIO2_EFFECT_CHAIN(ctypes.Structure):
    _fields_ = [
        ('EffectCount', UINT32),
        ('pEffectDescriptors', POINTER(XAUDIO2_EFFECT_DESCRIPTOR)),
    ]


class XAUDIO2_FILTER_PARAMETERS(Structure):
    _fields_ = [
        ('Type', XAUDIO2_FILTER_TYPE),
        ('Frequency', FLOAT),
        ('OneOverQ', FLOAT)
    ]


class XAUDIO2_VOICE_DETAILS(Structure):
    _fields_ = [
        ('CreationFlags', UINT32),
        ('ActiveFlags', UINT32),
        ('InputChannels', UINT32),
        ('InputSampleRate', UINT32)
    ]


class IXAudio2Voice(com.pInterface):
    _methods_ = [
        ('GetVoiceDetails',
         com.STDMETHOD(POINTER(XAUDIO2_VOICE_DETAILS))),
        ('SetOutputVoices',
         com.STDMETHOD()),
        ('SetEffectChain',
         com.STDMETHOD(POINTER(XAUDIO2_EFFECT_CHAIN))),
        ('EnableEffect',
         com.STDMETHOD()),
        ('DisableEffect',
         com.STDMETHOD()),
        ('GetEffectState',
         com.STDMETHOD()),
        ('SetEffectParameters',
         com.STDMETHOD()),
        ('GetEffectParameters',
         com.STDMETHOD()),
        ('SetFilterParameters',
         com.STDMETHOD(POINTER(XAUDIO2_FILTER_PARAMETERS), UINT32)),
        ('GetFilterParameters',
         com.STDMETHOD()),
        ('SetOutputFilterParameters',
         com.STDMETHOD()),
        ('GetOutputFilterParameters',
         com.STDMETHOD()),
        ('SetVolume',
         com.STDMETHOD(ctypes.c_float, UINT32)),
        ('GetVolume',
         com.STDMETHOD(POINTER(c_float))),
        ('SetChannelVolumes',
         com.STDMETHOD()),
        ('GetChannelVolumes',
         com.STDMETHOD()),
        ('SetOutputMatrix',
         com.STDMETHOD(c_void_p, UINT32, UINT32, POINTER(FLOAT), UINT32)),
        ('GetOutputMatrix',
         com.STDMETHOD()),
        ('DestroyVoice',
         com.STDMETHOD())
    ]


class IXAudio2SubmixVoice(IXAudio2Voice):
    pass


class IXAudio2SourceVoice(IXAudio2Voice):
    _methods_ = [
        ('Start',
         com.STDMETHOD(UINT32, UINT32)),
        ('Stop',
         com.STDMETHOD(UINT32, UINT32)),
        ('SubmitSourceBuffer',
         com.STDMETHOD(POINTER(XAUDIO2_BUFFER), c_void_p)),
        ('FlushSourceBuffers',
         com.STDMETHOD()),
        ('Discontinuity',
         com.STDMETHOD()),
        ('ExitLoop',
         com.STDMETHOD()),
        ('GetState',
         com.STDMETHOD(POINTER(XAUDIO2_VOICE_STATE), UINT32)),
        ('SetFrequencyRatio',
         com.STDMETHOD(FLOAT, UINT32)),
        ('GetFrequencyRatio',
         com.STDMETHOD(POINTER(c_float))),
        ('SetSourceSampleRate',
         com.STDMETHOD()),
    ]


class IXAudio2MasteringVoice(IXAudio2Voice):
    _methods_ = [
        ('GetChannelMask',
         com.STDMETHOD(POINTER(DWORD)))
    ]


class IXAudio2EngineCallback(com.Interface):
    _methods_ = [
        ('OnProcessingPassStart',
         com.METHOD(ctypes.c_void_p)),
        ('OnProcessingPassEnd',
         com.METHOD(ctypes.c_void_p)),
        ('OnCriticalError',
         com.METHOD(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)),
    ]


class XA2EngineCallback(com.COMObject):
    _interfaces_ = [IXAudio2EngineCallback]

    def OnProcessingPassStart(self):
        pass

    def OnProcessingPassEnd(self):
        pass

    def OnCriticalError(self, this, hresult):
        raise Exception("Critical Error:", hresult)



# -------------- 3D Audio Positioning----------
class X3DAUDIO_DISTANCE_CURVE_POINT(ctypes.Structure):
    _fields_ = [
        ('Distance', FLOAT32),
        ('DSPSetting', FLOAT32)
    ]


class X3DAUDIO_DISTANCE_CURVE(ctypes.Structure):
    _fields_ = [
        ('pPoints', POINTER(X3DAUDIO_DISTANCE_CURVE_POINT)),
        ('PointCount', UINT32)
    ]


class X3DAUDIO_VECTOR(ctypes.Structure):
    _fields_ = [
        ('x', c_float),
        ('y', c_float),
        ('z', c_float),
    ]



"""Cone:
   Specifies directionality for a listener or single-channel emitter by
   modifying DSP behaviour with respect to its front orientation.
   This is modeled using two sound cones: an inner cone and an outer cone.
   On/within the inner cone, DSP settings are scaled by the inner values.
   On/beyond the outer cone, DSP settings are scaled by the outer values.
   If on both the cones, DSP settings are scaled by the inner values only.
   Between the two cones, the scaler is linearly interpolated between the
   inner and outer values.  Set both cone angles to 0 or X3DAUDIO_2PI for
   omnidirectionality using only the outer or inner values respectively."""
class X3DAUDIO_CONE(Structure):
    _fields_ = [
        ('InnerAngle', FLOAT32),  # inner cone angle in radians, must be within [0.0f, X3DAUDIO_2PI]
        ('OuterAngle', FLOAT32),  # outer cone angle in radians, must be within [InnerAngle, X3DAUDIO_2PI]
        ('InnerVolume', FLOAT32),  # volume level scaler on/within inner cone, used only for matrix calculations, must be within [0.0f, 2.0f] when used
        ('OuterVolume', FLOAT32),  #  volume level scaler on/beyond outer cone, used only for matrix calculations, must be within [0.0f, 2.0f] when used
        ('InnerLPF', FLOAT32),  # LPF (both direct and reverb paths) coefficient subtrahend on/within inner cone, used only for LPF (both direct and reverb paths) calculations, must be within [0.0f, 1.0f] when used
        ('OuterLPF', FLOAT32),  # LPF (both direct and reverb paths) coefficient subtrahend on/beyond outer cone, used only for LPF (both direct and reverb paths) calculations, must be within [0.0f, 1.0f] when used
        ('InnerReverb', FLOAT32),  # reverb send level scaler on/within inner cone, used only for reverb calculations, must be within [0.0f, 2.0f] when used
        ('OuterReverb', FLOAT32)  # reverb send level scaler on/beyond outer cone, used only for reverb calculations, must be within [0.0f, 2.0f] when used
    ]


class X3DAUDIO_LISTENER(Structure):
    _fields_ = [
        ('OrientFront', X3DAUDIO_VECTOR),  # orientation of front direction, used only for matrix and delay calculations or listeners with cones for matrix, LPF (both direct and reverb paths), and reverb calculations, must be normalized when used
        ('OrientTop', X3DAUDIO_VECTOR),  # orientation of top direction, used only for matrix and delay calculations, must be orthonormal with OrientFront when used
        ('Position', X3DAUDIO_VECTOR),  # position in user-defined world units, does not affect Velocity
        ('Velocity', X3DAUDIO_VECTOR),  # velocity vector in user-defined world units/second, used only for doppler calculations, does not affect Position
        ('pCone', POINTER(X3DAUDIO_CONE))  # sound cone, used only for matrix, LPF (both direct and reverb paths), and reverb calculations, NULL specifies omnidirectionality
    ]


class X3DAUDIO_EMITTER(Structure):
    _fields_ = [
        ('pCone', POINTER(X3DAUDIO_CONE)),
        ('OrientFront', X3DAUDIO_VECTOR),
        ('OrientTop', X3DAUDIO_VECTOR),
        ('Position', X3DAUDIO_VECTOR),
        ('Velocity', X3DAUDIO_VECTOR),
        ('InnerRadius', FLOAT32),
        ('InnerRadiusAngle', FLOAT32),
        ('ChannelCount', UINT32),
        ('ChannelRadius', FLOAT32),
        ('pChannelAzimuths', POINTER(FLOAT32)),
        ('pVolumeCurve', POINTER(X3DAUDIO_DISTANCE_CURVE)),
        ('pLFECurve', POINTER(X3DAUDIO_DISTANCE_CURVE)),
        ('pLPFDirectCurve', POINTER(X3DAUDIO_DISTANCE_CURVE)),
        ('pLPFReverbCurve', POINTER(X3DAUDIO_DISTANCE_CURVE)),
        ('pReverbCurve', POINTER(X3DAUDIO_DISTANCE_CURVE)),
        ('CurveDistanceScaler', FLOAT32),
        ('DopplerScaler', FLOAT32)
    ]


class X3DAUDIO_DSP_SETTINGS(Structure):
    _fields_ = [
        ('pMatrixCoefficients', POINTER(FLOAT)),  # float array
        ('pDelayTimes', POINTER(FLOAT32)),
        ('SrcChannelCount', UINT32),
        ('DstChannelCount', UINT32),
        ('LPFDirectCoefficient', FLOAT32),
        ('LPFReverbCoefficient', FLOAT32),
        ('ReverbLevel', FLOAT32),
        ('DopplerFactor', FLOAT32),
        ('EmitterToListenerAngle', FLOAT32),
        ('EmitterToListenerDistance', FLOAT32),
        ('EmitterVelocityComponent', FLOAT32),
        ('ListenerVelocityComponent', FLOAT32)
    ]

# Other constants that may or may not be used in X3D.

SPEAKER_FRONT_LEFT             = 0x00000001
SPEAKER_FRONT_RIGHT            = 0x00000002
SPEAKER_FRONT_CENTER           = 0x00000004
SPEAKER_LOW_FREQUENCY          = 0x00000008
SPEAKER_BACK_LEFT              = 0x00000010
SPEAKER_BACK_RIGHT             = 0x00000020
SPEAKER_FRONT_LEFT_OF_CENTER   = 0x00000040
SPEAKER_FRONT_RIGHT_OF_CENTER  = 0x00000080
SPEAKER_BACK_CENTER            = 0x00000100
SPEAKER_SIDE_LEFT              = 0x00000200
SPEAKER_SIDE_RIGHT             = 0x00000400
SPEAKER_TOP_CENTER             = 0x00000800
SPEAKER_TOP_FRONT_LEFT         = 0x00001000
SPEAKER_TOP_FRONT_CENTER       = 0x00002000
SPEAKER_TOP_FRONT_RIGHT        = 0x00004000
SPEAKER_TOP_BACK_LEFT          = 0x00008000
SPEAKER_TOP_BACK_CENTER        = 0x00010000
SPEAKER_TOP_BACK_RIGHT         = 0x00020000
SPEAKER_RESERVED               = 0x7FFC0000  # bit mask locations reserved for future use
SPEAKER_ALL                    = 0x80000000  # used to specify that any possible permutation of speaker configurations

SPEAKER_MONO = SPEAKER_FRONT_CENTER
SPEAKER_STEREO = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT)
SPEAKER_2POINT1 = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_LOW_FREQUENCY)
SPEAKER_SURROUND = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_BACK_CENTER)
SPEAKER_QUAD = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT)
SPEAKER_4POINT1 = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT)
SPEAKER_5POINT1 = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT)
SPEAKER_7POINT1 = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT | SPEAKER_FRONT_LEFT_OF_CENTER | SPEAKER_FRONT_RIGHT_OF_CENTER)
SPEAKER_5POINT1_SURROUND = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_SIDE_LEFT  | SPEAKER_SIDE_RIGHT)
SPEAKER_7POINT1_SURROUND = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT | SPEAKER_SIDE_LEFT  | SPEAKER_SIDE_RIGHT)


DBL_DECIMAL_DIG = 17                      # # of decimal digits of rounding precision
DBL_DIG = 15                      # # of decimal digits of precision
DBL_EPSILON = 2.2204460492503131e-016  # smallest such that 1.0+DBL_EPSILON != 1.0
DBL_HAS_SUBNORM = 1                       # type does support subnormal numbers
DBL_MANT_DIG  = 53                      # # of bits in mantissa
DBL_MAX  = 1.7976931348623158e+308  # max value
DBL_MAX_10_EXP = 308                     # max decimal exponent
DBL_MAX_EXP = 1024                    # max binary exponent
DBL_MIN = 2.2250738585072014e-308  # min positive value
DBL_MIN_10_EXP = (-307)                  # min decimal exponent
DBL_MIN_EXP = (-1021)                 # min binary exponent
_DBL_RADIX = 2                       # exponent radix
DBL_TRUE_MIN = 4.9406564584124654e-324  # min positive value

FLT_DECIMAL_DIG = 9                       # # of decimal digits of rounding precision
FLT_DIG  = 6                       # # of decimal digits of precision
FLT_EPSILON = 1.192092896e-07        # smallest such that 1.0+FLT_EPSILON != 1.0
FLT_HAS_SUBNORM = 1                       # type does support subnormal numbers
FLT_GUARD = 0
FLT_MANT_DIG = 24                      # # of bits in mantissa
FLT_MAX = 3.402823466e+38        # max value
FLT_MAX_10_EXP = 38                      # max decimal exponent
FLT_MAX_EXP = 128                     # max binary exponent
FLT_MIN = 1.175494351e-38        # min normalized positive value
FLT_MIN_10_EXP = (-37)                   # min decimal exponent
FLT_MIN_EXP = (-125)                  # min binary exponent
FLT_NORMALIZE = 0
FLT_RADIX = 2                       # exponent radix
FLT_TRUE_MIN = 1.401298464e-45        # min positive value

LDBL_DIG = DBL_DIG                 # # of decimal digits of precision
LDBL_EPSILON = DBL_EPSILON             # smallest such that 1.0+LDBL_EPSILON != 1.0
LDBL_HAS_SUBNORM = DBL_HAS_SUBNORM         # type does support subnormal numbers
LDBL_MANT_DIG = DBL_MANT_DIG            # # of bits in mantissa
LDBL_MAX = DBL_MAX                 # max value
LDBL_MAX_10_EXP =  DBL_MAX_10_EXP          # max decimal exponent
LDBL_MAX_EXP = DBL_MAX_EXP             # max binary exponent
LDBL_MIN = DBL_MIN                 # min normalized positive value
LDBL_MIN_10_EXP = DBL_MIN_10_EXP          # min decimal exponent
LDBL_MIN_EXP = DBL_MIN_EXP             # min binary exponent
_LDBL_RADIX = _DBL_RADIX              # exponent radix
LDBL_TRUE_MIN = DBL_TRUE_MIN            # min positive value

DECIMAL_DIG = DBL_DECIMAL_DIG


X3DAUDIO_HANDLE_BYTESIZE = 20
X3DAUDIO_HANDLE = (BYTE * X3DAUDIO_HANDLE_BYTESIZE)


# speed of sound in meters per second for dry air at approximately 20C, used with X3DAudioInitialize
X3DAUDIO_SPEED_OF_SOUND = 343.5


X3DAUDIO_CALCULATE_MATRIX = 0x00000001  # enable matrix coefficient table calculation
X3DAUDIO_CALCULATE_DELAY = 0x00000002  # enable delay time array calculation (stereo final mix only)
X3DAUDIO_CALCULATE_LPF_DIRECT = 0x00000004  # enable LPF direct-path coefficient calculation
X3DAUDIO_CALCULATE_LPF_REVERB = 0x00000008  # enable LPF reverb-path coefficient calculation
X3DAUDIO_CALCULATE_REVERB = 0x00000010  # enable reverb send level calculation
X3DAUDIO_CALCULATE_DOPPLER = 0x00000020  # enable doppler shift factor calculation
X3DAUDIO_CALCULATE_EMITTER_ANGLE = 0x00000040  # enable emitter-to-listener interior angle calculation
X3DAUDIO_CALCULATE_ZEROCENTER = 0x00010000  # do not position to front center speaker, signal positioned to remaining speakers instead, front center destination channel will be zero in returned matrix coefficient table, valid only for matrix calculations with final mix formats that have a front center channel
X3DAUDIO_CALCULATE_REDIRECT_TO_LFE = 0x00020000  # apply equal mix of all source channels to LFE destination channel, valid only for matrix calculations with sources that have no LFE channel and final mix formats that have an LFE channel

default_dsp_calculation = X3DAUDIO_CALCULATE_MATRIX | X3DAUDIO_CALCULATE_DOPPLER

X3DAudioInitialize = x3d_lib.X3DAudioInitialize
X3DAudioInitialize.restype = HRESULT
X3DAudioInitialize.argtypes = [c_int, c_float, c_void_p]


X3DAudioCalculate = x3d_lib.X3DAudioCalculate
X3DAudioCalculate.restype = c_void
X3DAudioCalculate.argtypes = [POINTER(X3DAUDIO_HANDLE), POINTER(X3DAUDIO_LISTENER), POINTER(X3DAUDIO_EMITTER), UINT32, POINTER(X3DAUDIO_DSP_SETTINGS)]


AudioCategory_Other = 0
AudioCategory_ForegroundOnlyMedia = 1
AudioCategory_Communications = 3
AudioCategory_Alerts = 4
AudioCategory_SoundEffects = 5
AudioCategory_GameEffects = 6
AudioCategory_GameMedia = 7
AudioCategory_GameChat = 8
AudioCategory_Speech = 9
AudioCategory_Movie = 10
AudioCategory_Media = 11

# Reverb not implemented but if someone wants to take a stab at it.
class XAUDIO2FX_REVERB_PARAMETERS(Structure):
    _fields_ = [
        ('WetDryMix', c_float),  #  ratio of wet (processed) signal to dry (original) signal

        # Delay times
        ('ReflectionsDelay', UINT32),  #  [0, 300] in ms
        ('ReverbDelay', BYTE),  # [0, 85] in ms
        ('RearDelay', UINT32),  # 7.1: [0, 20] in ms, all other: [0, 5] in ms
        ('SideDelay', UINT32),  # .1: [0, 5] in ms, all other: not used, but still validated  # WIN 10 only.

        # Indexed Paremeters
        ('PositionLeft', BYTE),  # [0, 30] no units
        ('PositionRight', BYTE),  # 0, 30] no units, ignored when configured to mono
        ('PositionMatrixLeft', BYTE),  # [0, 30] no units
        ('PositionMatrixRight', BYTE),  # [0, 30] no units, ignored when configured to mono
        ('EarlyDiffusion', BYTE), # [0, 15] no units
        ('LateDiffusion', BYTE),  # [0, 15] no units
        ('LowEQGain', BYTE),  # [0, 12] no units
        ('LowEQCutoff', BYTE),  # [0, 9] no units
        ('LowEQCutoff', BYTE),  # [0, 8] no units
        ('HighEQCutoff', BYTE),  # [0, 14] no units

        # Direct parameters
        ('RoomFilterFreq', c_float),  # [20, 20000] in Hz
        ('RoomFilterMain', c_float),  # [-100, 0] in dB
        ('RoomFilterHF', c_float),  # [-100, 0] in dB
        ('ReflectionsGain', c_float),  # [-100, 20] in dB
        ('ReverbGain', c_float),  # [-100, 20] in dB
        ('DecayTime', c_float),  # [0.1, inf] in seconds
        ('Density', c_float),  # [0, 100] (percentage)
        ('RoomSize', c_float),  # [1, 100] in feet

        # component control
        ('DisableLateField', c_bool),  # TRUE to disable late field reflections
    ]


class IXAudio2(com.pIUnknown):
    _methods_ = [
        ('RegisterForCallbacks',
           com.STDMETHOD(POINTER(IXAudio2EngineCallback))),
        ('UnregisterForCallbacks',
           com.METHOD(ctypes.c_void_p, POINTER(IXAudio2EngineCallback))),
        ('CreateSourceVoice',
         com.STDMETHOD(POINTER(IXAudio2SourceVoice), POINTER(WAVEFORMATEX), UINT32, c_float,
                       POINTER(IXAudio2VoiceCallback), POINTER(XAUDIO2_VOICE_SENDS), POINTER(XAUDIO2_EFFECT_CHAIN))),
        ('CreateSubmixVoice',
         com.STDMETHOD(POINTER(IXAudio2SubmixVoice), UINT32, UINT32, UINT32, UINT32,
                       POINTER(XAUDIO2_VOICE_SENDS), POINTER(XAUDIO2_EFFECT_CHAIN))),
        ('CreateMasteringVoice',
         com.STDMETHOD(POINTER(IXAudio2MasteringVoice), UINT32, UINT32, UINT32, LPCWSTR, POINTER(XAUDIO2_EFFECT_CHAIN),
                       UINT32)),
        ('StartEngine',
         com.STDMETHOD()),
        ('StopEngine',
         com.STDMETHOD()),
        ('CommitChanges',
         com.STDMETHOD(UINT32)),
        ('GetPerformanceData',
         com.METHOD(c_void, POINTER(XAUDIO2_PERFORMANCE_DATA))),
        ('SetDebugConfiguration',
         com.STDMETHOD(POINTER(XAUDIO2_DEBUG_CONFIGURATION), c_void_p)),
    ]


XAudio2Create = xaudio2_lib.XAudio2Create
XAudio2Create.restype = HRESULT
XAudio2Create.argtypes = [POINTER(IXAudio2), UINT32, UINT32]

CreateAudioReverb = xaudio2_lib.CreateAudioReverb
CreateAudioReverb.restype = HRESULT
CreateAudioReverb.argtypes = [POINTER(com.pIUnknown)]

