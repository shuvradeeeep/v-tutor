"""
Audio in: microphone or LiveKit track -> stt.vad.AudioSegmenter -> Whisper ->
VoiceBridge. This is the only module that imports from stt/.

The segmenter's START_OF_SPEECH callback is the barge-in fast path; the
completed utterance goes to Whisper in a worker thread (CPU-bound) and the
text reaches the bridge as a transcript.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncIterator

from voice.bridge import VoiceBridge

logger = logging.getLogger("v-tutor.audio")

# Same knobs as stt/agent.py, same env names, so the two agree.
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
MIN_SPEECH_DURATION = float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.1"))
MIN_SILENCE_DURATION = float(os.getenv("VAD_MIN_SILENCE_DURATION", "0.5"))
MAX_UTTERANCE_DURATION = float(os.getenv("VAD_MAX_UTTERANCE_DURATION", "30.0"))

# Whisper's favourite hallucinations on near-silence; treated as no speech.
_NOISE = {"thank you.", "thanks.", "thank you", "you", "bye.", ".", "okay.", "hmm."}


class SpeechInput:
    """Owns the VAD segmenter and the Whisper model (both loaded once)."""

    def __init__(self, transcriber=None, segmenter=None) -> None:
        self._transcriber = transcriber
        self._segmenter = segmenter

    def load(self) -> None:
        if self._transcriber is None:
            from stt.transcriber import WhisperTranscriber
            t = time.perf_counter()
            self._transcriber = WhisperTranscriber(model_size=MODEL_SIZE, device=DEVICE,
                                                   compute_type=COMPUTE_TYPE, beam_size=BEAM_SIZE)
            logger.info("whisper %s loaded in %.1fs", MODEL_SIZE, time.perf_counter() - t)
        if self._segmenter is None:
            from stt.vad import AudioSegmenter
            self._segmenter = AudioSegmenter(min_speech_duration=MIN_SPEECH_DURATION,
                                             min_silence_duration=MIN_SILENCE_DURATION,
                                             max_utterance_duration=MAX_UTTERANCE_DURATION)

    async def run_frames(self, frames: AsyncIterator, bridge: VoiceBridge, label: str = "mic") -> None:
        self.load()
        async for utt in self._segmenter.process_frames(frames, on_speech_start=bridge.on_speech_start,
                                                        label=label):
            await self._handle(utt, bridge)

    async def run_track(self, track, bridge: VoiceBridge) -> None:
        self.load()
        async for utt in self._segmenter.process_track(track, on_speech_start=bridge.on_speech_start):
            await self._handle(utt, bridge)

    async def _handle(self, utt, bridge: VoiceBridge) -> None:
        try:
            text, lang, prob, whisper_ms = await asyncio.to_thread(self._transcriber.transcribe, utt.frames)
        except Exception:  # noqa: BLE001
            logger.exception("whisper failed")
            text, lang, prob, whisper_ms = "", None, 0.0, 0.0
        if text.strip().lower() in _NOISE and utt.vad_duration_sec < 1.0:
            text = ""
        logger.info("STT %.2fs -> %r (%s %.2f, %.0f ms)", utt.vad_duration_sec, text, lang, prob, whisper_ms)
        bridge.on_transcript(text, lang=lang, prob=prob, duration_s=utt.vad_duration_sec, whisper_ms=whisper_ms)


async def mic_frames(device: int | str | None = None, sample_rate: int = 16000,
                     block_ms: int = 20) -> AsyncIterator:
    """Laptop microphone as LiveKit AudioFrames (16 kHz mono int16)."""
    import sounddevice as sd
    from livekit import rtc

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    block = int(sample_rate * block_ms / 1000)

    def cb(indata, frames, time_info, status) -> None:  # noqa: ARG001
        if status:
            logger.debug("mic status: %s", status)
        data = bytes(indata)
        try:
            loop.call_soon_threadsafe(q.put_nowait, data)
        except RuntimeError:
            pass

    stream = sd.RawInputStream(samplerate=sample_rate, channels=1, dtype="int16", device=device,
                               blocksize=block, callback=cb)
    stream.start()
    try:
        while True:
            data = await q.get()
            yield rtc.AudioFrame(data=data, sample_rate=sample_rate, num_channels=1,
                                 samples_per_channel=len(data) // 2)
    finally:
        stream.stop()
        stream.close()
