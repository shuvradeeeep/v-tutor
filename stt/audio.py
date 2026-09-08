"""
audio.py
Small DSP helpers: format conversion, loudness measurement and a background
noise-floor tracker. No scipy / audioop required.
"""

from __future__ import annotations

import numpy as np
from livekit import rtc

TARGET_RATE = 16000
_SILENCE_DBFS = -100.0


def frames_to_int16(frames: list[rtc.AudioFrame]) -> tuple[np.ndarray, int]:
    """Concatenate LiveKit frames into a mono int16 array (still at input rate)."""
    if not frames:
        return np.array([], dtype=np.int16), TARGET_RATE

    in_rate = frames[0].sample_rate
    channels = frames[0].num_channels
    pcm = np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in frames])

    if channels > 1:
        # Trim any partial frame before reshaping, then average the channels.
        usable = (pcm.size // channels) * channels
        pcm = pcm[:usable].reshape(-1, channels).mean(axis=1).astype(np.int16)

    return pcm, in_rate


def frames_to_float32(frames: list[rtc.AudioFrame]) -> np.ndarray:
    """
    Convert LiveKit PCM frames to the 16kHz mono float32 [-1, 1] array Whisper
    expects.

    Resampling goes through LiveKit's SoX resampler rather than
    `audioop.ratecv`: it is properly band-limited (no aliasing artefacts for
    Whisper to hallucinate on) and `audioop` is gone from Python 3.13.
    """
    pcm, in_rate = frames_to_int16(frames)
    if pcm.size == 0:
        return np.array([], dtype=np.float32)

    if in_rate != TARGET_RATE:
        resampler = rtc.AudioResampler(
            input_rate=in_rate,
            output_rate=TARGET_RATE,
            num_channels=1,
            quality=rtc.AudioResamplerQuality.HIGH,
        )
        out = resampler.push(bytearray(pcm.tobytes())) + resampler.flush()
        if not out:
            return np.array([], dtype=np.float32)
        pcm = np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in out])

    return pcm.astype(np.float32) / 32768.0


def highpass(x: np.ndarray, cutoff_hz: float = 80.0, sample_rate: int = TARGET_RATE) -> np.ndarray:
    """
    Zero out everything below `cutoff_hz` (mains hum, fan rumble, desk thumps,
    mic handling noise). Speech fundamentals start around 85Hz for men, so 80Hz
    is safe. Done in the frequency domain to stay dependency-free.
    """
    if x.size < 64:
        return x
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / sample_rate)
    spectrum[freqs < cutoff_hz] = 0.0
    return np.fft.irfft(spectrum, n=x.size).astype(np.float32)


def rms_dbfs(x: np.ndarray) -> float:
    """Loudness of a float32 [-1, 1] signal, in dB relative to full scale."""
    if x.size == 0:
        return _SILENCE_DBFS
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    if rms <= 1e-9:
        return _SILENCE_DBFS
    return 20.0 * np.log10(rms)


def frames_dbfs(frames: list[rtc.AudioFrame]) -> float:
    """Loudness of raw int16 frames, without paying for a resample."""
    pcm, _ = frames_to_int16(frames)
    if pcm.size == 0:
        return _SILENCE_DBFS
    return rms_dbfs(pcm.astype(np.float32) / 32768.0)


class NoiseFloorTracker:
    """
    Running estimate of the room's background level in dBFS.

    Falls fast and rises slowly: a quiet moment immediately lowers the floor,
    but a burst of noise only raises it gradually. That keeps the SNR gate from
    being "trained" upward by the very noise it is supposed to reject.
    """

    def __init__(self, initial_dbfs: float = -60.0, floor: float = -80.0, ceiling: float = -25.0):
        self.value = initial_dbfs
        self._floor = floor
        self._ceiling = ceiling

    def update(self, dbfs: float) -> float:
        if dbfs <= _SILENCE_DBFS:
            return self.value
        alpha = 0.25 if dbfs < self.value else 0.02
        self.value = (1.0 - alpha) * self.value + alpha * dbfs
        self.value = min(max(self.value, self._floor), self._ceiling)
        return self.value
