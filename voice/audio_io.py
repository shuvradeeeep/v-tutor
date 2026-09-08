"""
Audio in: microphone or LiveKit track -> stt.vad.AudioSegmenter -> Whisper ->
VoiceBridge. This is the only module that imports from stt/.

The segmenter's START_OF_SPEECH callback is the barge-in fast path; the
completed utterance goes to Whisper in a worker thread (CPU-bound) and the
text reaches the bridge as a transcript.

Every knob comes from stt/settings.py -- the same module stt/agent.py reads,
so the tutor's ear and the standalone STT worker are always configured
identically. Nothing is declared here.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

from stt import settings
from stt.transcripts import TranscriptLog
from voice.bridge import VoiceBridge

logger = logging.getLogger("v-tutor.audio")
_HALLUCINATIONS = {p.strip().lower().strip(" .!,?") for p in settings.HALLUCINATION_PHRASES}


class SpeechInput:
    """Owns the VAD segmenter and the Whisper model (both loaded once)."""

    def __init__(self, transcriber=None, segmenter=None, transcript_log=None) -> None:
        self._transcriber = transcriber
        self._segmenter = segmenter
        # Same CSV the standalone stt/agent.py writes, same columns: a tutor
        # session leaves the same evidence trail as a bare STT session.
        self._log = transcript_log
        if self._log is None and settings.LOG_TRANSCRIPTS:
            self._log = TranscriptLog(settings.TRANSCRIPTS_CSV)

    def load(self) -> None:
        if self._transcriber is None:
            # Local faster-whisper or Whisper large-v3 on Groq, per STT_PROVIDER
            # (stt/settings.py). Either way the same transcribe(frames) contract.
            from stt.cloud import describe, make_transcriber
            self._transcriber = make_transcriber()
            if settings.WHISPER_WARMUP:
                # load() is called from prewarm / before the tutor speaks, so
                # the lazy init cost lands here and not on the learner's first
                # sentence (where it would add ~1s to the barge-in round trip).
                self._transcriber.warmup()
            logger.info("ear: %s", describe(self._transcriber))
        if self._segmenter is None:
            from stt.vad import AudioSegmenter
            self._segmenter = AudioSegmenter(min_speech_duration=settings.VAD_MIN_SPEECH_DURATION,
                                             min_silence_duration=settings.VAD_MIN_SILENCE_DURATION,
                                             max_utterance_duration=settings.VAD_MAX_UTTERANCE_DURATION)

    def describe(self) -> str:
        from stt.cloud import describe
        return describe(self._transcriber) if self._transcriber is not None else "not loaded"

    async def run_frames(self, frames: AsyncIterator, bridge: VoiceBridge, label: str = "mic") -> None:
        self.load()
        async for utt in self._segmenter.process_frames(frames, on_speech_start=bridge.on_speech_start,
                                                        label=label):
            await self._handle(utt, bridge, label)

    async def run_track(self, track, bridge: VoiceBridge, label: str = "learner") -> None:
        self.load()
        async for utt in self._segmenter.process_track(track, on_speech_start=bridge.on_speech_start):
            await self._handle(utt, bridge, label)

    async def _handle(self, utt, bridge: VoiceBridge, label: str = "learner") -> None:
        try:
            text, lang, prob, whisper_ms = await asyncio.to_thread(self._transcriber.transcribe, utt.frames)
        except Exception:  # noqa: BLE001
            logger.exception("whisper failed")
            text, lang, prob, whisper_ms = "", None, 0.0, 0.0
        norm = text.strip().lower().strip(" .!,?")
        if (text.strip().lower() in settings.SHORT_NOISE_PHRASES
                and utt.vad_duration_sec < settings.SHORT_NOISE_MAX_SEC):
            text = ""
        elif norm in _HALLUCINATIONS and utt.vad_duration_sec < settings.HALLUCINATION_MAX_SEC:
            # "Thank you." from 2 s of room noise became the answer to "which
            # language?" in a live session. Nothing a learner says to a tutor
            # is only one of these phrases, so treat it as silence.
            logger.info("STT dropped likely hallucination %r (%.1fs)", text, utt.vad_duration_sec)
            text = ""
        total_ms = (time.perf_counter() - utt.end_time) * 1000
        logger.info("STT %.2fs -> %r (%s %.2f, %.0f ms)", utt.vad_duration_sec, text, lang, prob, whisper_ms)
        # The graph first: it is what the learner is waiting for. The CSV row is
        # a buffered append (tens of microseconds) and can follow.
        bridge.on_transcript(text, lang=lang, prob=prob, duration_s=utt.vad_duration_sec, whisper_ms=whisper_ms)
        if self._log is not None and text:
            self._log.append(label, text, lang, prob, utt.vad_duration_sec, whisper_ms, total_ms)

    def close(self) -> None:
        if self._log is not None:
            self._log.close()


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
