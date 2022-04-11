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
import math
import struct
import random
import ctypes

from .codecs.base import Source, AudioFormat, AudioData

from collections import deque


class Envelope:
    """Base class for SynthesisSource amplitude envelopes."""
    def get_generator(self, sample_rate, duration):
        raise NotImplementedError


class FlatEnvelope(Envelope):
    """A flat envelope, providing basic amplitude setting.

    :Parameters:
        `amplitude` : float
            The amplitude (volume) of the wave, from 0.0 to 1.0.
            Values outside of this range will be clamped.
    """
    def __init__(self, amplitude=0.5):
        self.amplitude = max(min(1.0, amplitude), 0)

    def get_generator(self, sample_rate, duration):
        amplitude = self.amplitude
        while True:
            yield amplitude


class LinearDecayEnvelope(Envelope):
    """A linearly decaying envelope.

    This envelope linearly decays the amplitude from the peak value
    to 0, over the length of the waveform.

    :Parameters:
        `peak` : float
            The Initial peak value of the envelope, from 0.0 to 1.0.
            Values outside of this range will be clamped.
    """
    def __init__(self, peak=1.0):
        self.peak = max(min(1.0, peak), 0)

    def get_generator(self, sample_rate, duration):
        peak = self.peak
        total_bytes = int(sample_rate * duration)
        for i in range(total_bytes):
            yield (total_bytes - i) / total_bytes * peak


class ADSREnvelope(Envelope):
    """A four part Attack, Decay, Suspend, Release envelope.

    This is a four part ADSR envelope. The attack, decay, and release
    parameters should be provided in seconds. For example, a value of
    0.1 would be 100ms. The sustain_amplitude parameter affects the
    sustain volume. This defaults to a value of 0.5, but can be provided
    on a scale from 0.0 to 1.0.

    :Parameters:
        `attack` : float
            The attack time, in seconds.
        `decay` : float
            The decay time, in seconds.
        `release` : float
            The release time, in seconds.
        `sustain_amplitude` : float
            The sustain amplitude (volume), from 0.0 to 1.0.
    """
    def __init__(self, attack, decay, release, sustain_amplitude=0.5):
        self.attack = attack
        self.decay = decay
        self.release = release
        self.sustain_amplitude = max(min(1.0, sustain_amplitude), 0)

    def get_generator(self, sample_rate, duration):
        sustain_amplitude = self.sustain_amplitude
        total_bytes = int(sample_rate * duration)
        attack_bytes = int(sample_rate * self.attack)
        decay_bytes = int(sample_rate * self.decay)
        release_bytes = int(sample_rate * self.release)
        sustain_bytes = total_bytes - attack_bytes - decay_bytes - release_bytes
        decay_step = (1 - sustain_amplitude) / decay_bytes
        release_step = sustain_amplitude / release_bytes
        for i in range(1, attack_bytes + 1):
            yield i / attack_bytes
        for i in range(1, decay_bytes + 1):
            yield 1 - (i * decay_step)
        for i in range(1, sustain_bytes + 1):
            yield sustain_amplitude
        for i in range(1, release_bytes + 1):
            yield sustain_amplitude - (i * release_step)


class TremoloEnvelope(Envelope):
    """A tremolo envelope, for modulation amplitude.

    A tremolo envelope that modulates the amplitude of the
    waveform with a sinusoidal pattern. The depth and rate
    of modulation can be specified. Depth is calculated as
    a percentage of the maximum amplitude. For example:
    a depth of 0.2 and amplitude of 0.5 will fluctuate
    the amplitude between 0.4 an 0.5.

    :Parameters:
        `depth` : float
            The amount of fluctuation, from 0.0 to 1.0.
        `rate` : float
            The fluctuation frequency, in seconds.
        `amplitude` : float
            The peak amplitude (volume), from 0.0 to 1.0.
    """
    def __init__(self, depth, rate, amplitude=0.5):
        self.depth = max(min(1.0, depth), 0)
        self.rate = rate
        self.amplitude = max(min(1.0, amplitude), 0)

    def get_generator(self, sample_rate, duration):
        total_bytes = int(sample_rate * duration)
        period = total_bytes / duration
        max_amplitude = self.amplitude
        min_amplitude = max(0.0, (1.0 - self.depth) * self.amplitude)
        step = (math.pi * 2) / period / self.rate
        for i in range(total_bytes):
            value = math.sin(step * i)
            yield value * (max_amplitude - min_amplitude) + min_amplitude


class SynthesisSource(Source):
    """Base class for synthesized waveforms.

    :Parameters:
        `duration` : float
            The length, in seconds, of audio that you wish to generate.
        `sample_rate` : int
            Audio samples per second. (CD quality is 44100).
        `sample_size` : int
            The bit precision. Must be either 8 or 16.
    """
    def __init__(self, duration, sample_rate=44800, sample_size=16, envelope=None):

        self._duration = float(duration)
        self.audio_format = AudioFormat(
            channels=1,
            sample_size=sample_size,
            sample_rate=sample_rate)

        self._offset = 0
        self._sample_rate = sample_rate
        self._sample_size = sample_size
        self._bytes_per_sample = sample_size >> 3
        self._bytes_per_second = self._bytes_per_sample * sample_rate
        self._max_offset = int(self._bytes_per_second * self._duration)
        self.envelope = envelope or FlatEnvelope(amplitude=1.0)
        self._envelope_generator = self.envelope.get_generator(sample_rate, duration)

        if self._bytes_per_sample == 2:
            self._max_offset &= 0xfffffffe

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        """Return `num_bytes` bytes of audio data."""
        num_bytes = min(num_bytes, self._max_offset - self._offset)
        if num_bytes <= 0:
            return None

        timestamp = float(self._offset) / self._bytes_per_second
        duration = float(num_bytes) / self._bytes_per_second
        data = self._generate_data(num_bytes)
        self._offset += num_bytes

        return AudioData(data, num_bytes, timestamp, duration, [])

    def _generate_data(self, num_bytes):
        """Generate `num_bytes` bytes of data.

        Return data as ctypes array or string.
        """
        raise NotImplementedError('abstract')

    def seek(self, timestamp):
        self._offset = int(timestamp * self._bytes_per_second)

        # Bound within duration
        self._offset = min(max(self._offset, 0), self._max_offset)

        # Align to sample
        if self._bytes_per_sample == 2:
            self._offset &= 0xfffffffe

        self._envelope_generator = self.envelope.get_generator(self._sample_rate, self._duration)

    def save(self, filename):
        """Save the audio to disk as a standard RIFF Wave.

        A standard RIFF wave header will be added to the raw PCM
        audio data when it is saved to disk.

        :Parameters:
            `filename` : str
                The file name to save as.

        """
        self.seek(0)
        data = self.get_audio_data(self._max_offset).get_string_data()
        header = struct.pack('<4sI8sIHHIIHH4sI',
                             b"RIFF",
                             len(data) + 44 - 8,
                             b"WAVEfmt ",
                             16,                # Default for PCM
                             1,                 # Default for PCM
                             1,                 # Number of channels
                             self._sample_rate,
                             self._bytes_per_second,
                             self._bytes_per_sample,
                             self._sample_size,
                             b"data",
                             len(data))

        with open(filename, "wb") as f:
            f.write(header)
            f.write(data)


class Silence(SynthesisSource):
    """A silent waveform."""

    def _generate_data(self, num_bytes):
        if self._bytes_per_sample == 1:
            return b'\127' * num_bytes
        else:
            return b'\0' * num_bytes


class WhiteNoise(SynthesisSource):
    """A white noise, random waveform."""

    def _generate_data(self, num_bytes):
        return os.urandom(num_bytes)


class Sine(SynthesisSource):
    """A sinusoid (sine) waveform.

    :Parameters:
        `duration` : float
            The length, in seconds, of audio that you wish to generate.
        `frequency` : int
            The frequency, in Hz of the waveform you wish to produce.
        `sample_rate` : int
            Audio samples per second. (CD quality is 44100).
        `sample_size` : int
            The bit precision. Must be either 8 or 16.
    """
    def __init__(self, duration, frequency=440, **kwargs):
        super(Sine, self).__init__(duration, **kwargs)
        self.frequency = frequency

    def _generate_data(self, num_bytes):
        if self._bytes_per_sample == 1:
            samples = num_bytes
            bias = 127
            amplitude = 127
            data = (ctypes.c_ubyte * samples)()
        else:
            samples = num_bytes >> 1
            bias = 0
            amplitude = 32767
            data = (ctypes.c_short * samples)()
        step = self.frequency * (math.pi * 2) / self.audio_format.sample_rate
        envelope = self._envelope_generator
        for i in range(samples):
            data[i] = int(math.sin(step * i) * amplitude * next(envelope) + bias)
        return bytes(data)


class Triangle(SynthesisSource):
    """A triangle waveform.

    :Parameters:
        `duration` : float
            The length, in seconds, of audio that you wish to generate.
        `frequency` : int
            The frequency, in Hz of the waveform you wish to produce.
        `sample_rate` : int
            Audio samples per second. (CD quality is 44100).
        `sample_size` : int
            The bit precision. Must be either 8 or 16.
    """
    def __init__(self, duration, frequency=440, **kwargs):
        super(Triangle, self).__init__(duration, **kwargs)
        self.frequency = frequency
        
    def _generate_data(self, num_bytes):
        if self._bytes_per_sample == 1:
            samples = num_bytes
            value = 127
            maximum = 255
            minimum = 0
            data = (ctypes.c_ubyte * samples)()
        else:
            samples = num_bytes >> 1
            value = 0
            maximum = 32767
            minimum = -32768
            data = (ctypes.c_short * samples)()
        step = (maximum - minimum) * 2 * self.frequency / self.audio_format.sample_rate
        envelope = self._envelope_generator
        for i in range(samples):
            value += step
            if value > maximum:
                value = maximum - (value - maximum)
                step = -step
            if value < minimum:
                value = minimum - (value - minimum)
                step = -step
            data[i] = int(value * next(envelope))
        return bytes(data)


class Sawtooth(SynthesisSource):
    """A sawtooth waveform.

    :Parameters:
        `duration` : float
            The length, in seconds, of audio that you wish to generate.
        `frequency` : int
            The frequency, in Hz of the waveform you wish to produce.
        `sample_rate` : int
            Audio samples per second. (CD quality is 44100).
        `sample_size` : int
            The bit precision. Must be either 8 or 16.
    """
    def __init__(self, duration, frequency=440, **kwargs):
        super(Sawtooth, self).__init__(duration, **kwargs)
        self.frequency = frequency

    def _generate_data(self, num_bytes):
        if self._bytes_per_sample == 1:
            samples = num_bytes
            value = 127
            maximum = 255
            minimum = 0
            data = (ctypes.c_ubyte * samples)()
        else:
            samples = num_bytes >> 1
            value = 0
            maximum = 32767
            minimum = -32768
            data = (ctypes.c_short * samples)()
        step = (maximum - minimum) * self.frequency / self._sample_rate
        envelope = self._envelope_generator
        for i in range(samples):
            value += step
            if value > maximum:
                value = minimum + (value % maximum)
            data[i] = int(value * next(envelope))
        return bytes(data)


class Square(SynthesisSource):
    """A square (pulse) waveform.

    :Parameters:
        `duration` : float
            The length, in seconds, of audio that you wish to generate.
        `frequency` : int
            The frequency, in Hz of the waveform you wish to produce.
        `sample_rate` : int
            Audio samples per second. (CD quality is 44100).
        `sample_size` : int
            The bit precision. Must be either 8 or 16.
    """
    def __init__(self, duration, frequency=440, **kwargs):

        super(Square, self).__init__(duration, **kwargs)
        self.frequency = frequency

    def _generate_data(self, num_bytes):
        if self._bytes_per_sample == 1:
            samples = num_bytes
            bias = 127
            amplitude = 127
            data = (ctypes.c_ubyte * samples)()
        else:
            samples = num_bytes >> 1
            bias = 0
            amplitude = 32767
            data = (ctypes.c_short * samples)()
        half_period = self.audio_format.sample_rate / self.frequency / 2
        envelope = self._envelope_generator
        value = 1
        count = 0
        for i in range(samples):
            if count >= half_period:
                value = -value
                count %= half_period
            count += 1
            data[i] = int(value * amplitude * next(envelope) + bias)
        return bytes(data)


class FM(SynthesisSource):
    """A simple FM waveform.

    This is a simplistic frequency modulated waveform, based on the
    concepts by John Chowning. Basic sine waves are used for both
    frequency carrier and modulator inputs, of which the frequencies can
    be provided. The modulation index, or amplitude, can also be adjusted.

    :Parameters:
        `duration` : float
            The length, in seconds, of audio that you wish to generate.
        `carrier` : int
            The carrier frequency, in Hz.
        `modulator` : int
            The modulator frequency, in Hz.
        `mod_index` : int
            The modulation index.
        `sample_rate` : int
            Audio samples per second. (CD quality is 44100).
        `sample_size` : int
            The bit precision. Must be either 8 or 16.
    """
    def __init__(self, duration, carrier=440, modulator=440, mod_index=1, **kwargs):
        super(FM, self).__init__(duration, **kwargs)
        self.carrier = carrier
        self.modulator = modulator
        self.mod_index = mod_index

    def _generate_data(self, num_bytes):
        if self._bytes_per_sample == 1:
            samples = num_bytes
            bias = 127
            amplitude = 127
            data = (ctypes.c_ubyte * samples)()
        else:
            samples = num_bytes >> 1
            bias = 0
            amplitude = 32767
            data = (ctypes.c_short * samples)()
        car_step = 2 * math.pi * self.carrier
        mod_step = 2 * math.pi * self.modulator
        mod_index = self.mod_index
        sample_rate = self._sample_rate
        envelope = self._envelope_generator
        sin = math.sin
        # FM equation:  sin((2 * pi * carrier) + sin(2 * pi * modulator))
        for i in range(samples):
            increment = i / sample_rate
            data[i] = int(sin(car_step * increment + mod_index * sin(mod_step * increment))
                          * amplitude * next(envelope) + bias)
        return bytes(data)


class Digitar(SynthesisSource):
    """A guitar-like waveform.

    A guitar-like waveform, based on the Karplus-Strong algorithm.
    The sound is similar to a plucked guitar string. The resulting
    sound decays over time, and so the actual length will vary
    depending on the frequency. Lower frequencies require a longer
    `length` parameter to prevent cutting off abruptly.

    :Parameters:
        `duration` : float
            The length, in seconds, of audio that you wish to generate.
        `frequency` : int
            The frequency, in Hz of the waveform you wish to produce.
        `decay` : float
            The decay rate of the effect. Defaults to 0.996.
        `sample_rate` : int
            Audio samples per second. (CD quality is 44100).
        `sample_size` : int
            The bit precision. Must be either 8 or 16.
    """
    def __init__(self, duration, frequency=440, decay=0.996, **kwargs):
        super(Digitar, self).__init__(duration, **kwargs)
        self.frequency = frequency
        self.decay = decay
        self.period = int(self._sample_rate / self.frequency)

    def _generate_data(self, num_bytes):
        if self._bytes_per_sample == 1:
            samples = num_bytes
            bias = 127
            amplitude = 127
            data = (ctypes.c_ubyte * samples)()
        else:
            samples = num_bytes >> 1
            bias = 0
            amplitude = 32767
            data = (ctypes.c_short * samples)()
        random.seed(10)
        period = self.period
        ring_buffer = deque([random.uniform(-1, 1) for _ in range(period)], maxlen=period)
        decay = self.decay
        for i in range(samples):
            data[i] = int(ring_buffer[0] * amplitude + bias)
            ring_buffer.append(decay * (ring_buffer[0] + ring_buffer[1]) / 2)
        return bytes(data)
