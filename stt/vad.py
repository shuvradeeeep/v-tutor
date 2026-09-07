"""
vad.py
Handles Voice Activity Detection (VAD) and speech audio segmentation.
"""

import asyncio
import collections
import logging
import time
from typing import AsyncGenerator, AsyncIterator, Callable, Optional
from dataclasses import dataclass

from livekit import agents, rtc
from livekit.plugins import silero

logger = logging.getLogger("stt-orchestrator")


@dataclass
class SpeechUtterance:
    frames: list[rtc.AudioFrame]
    vad_duration_sec: float
    end_time: float


class AudioSegmenter:
    def __init__(
        self,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.5,
        max_utterance_duration: float = 30.0,
        preroll_frames: int = 15,
    ):
        """
        Sets up Silero VAD parameters.

        max_utterance_duration: safety cap in seconds. Without it, continuous
        speech with no pause long enough to trigger END_OF_SPEECH keeps
        growing the frame buffer, and you eventually pay for one huge, slow
        Whisper call instead of several fast ones. Default is generous (30s)
        so it only kicks in for genuinely long, unbroken speech.
        """
        self.vad = silero.VAD.load(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
        )
        self.max_utterance_duration = max_utterance_duration
        self.preroll_frames = preroll_frames

    async def process_track(
        self,
        track: rtc.Track,
        on_speech_start: Optional[Callable[[], None]] = None,
    ) -> AsyncGenerator[SpeechUtterance, None]:
        """
        Listens to a LiveKit track, processes frames through Silero VAD,
        and yields completed SpeechUtterance instances whenever the user
        finishes talking (or the max-duration safety cap is hit).

        on_speech_start: optional callback fired the instant START_OF_SPEECH is
        detected, before any transcript exists. The tutor uses this as the
        barge-in fast path (stop playback now, classify the words later).
        """
        audio_stream = rtc.AudioStream(track)

        async def frames() -> AsyncIterator[rtc.AudioFrame]:
            async for audio_event in audio_stream:
                yield audio_event.frame

        async for utt in self.process_frames(frames(), on_speech_start=on_speech_start, label=track.sid):
            yield utt

    async def process_frames(
        self,
        frames: AsyncIterator[rtc.AudioFrame],
        on_speech_start: Optional[Callable[[], None]] = None,
        label: str = "frames",
    ) -> AsyncGenerator[SpeechUtterance, None]:
        """
        Same as process_track, but over any async iterator of AudioFrames --
        a LiveKit track, a local microphone, or a test fixture.
        """
        vad_stream = self.vad.stream()
        preroll = collections.deque(maxlen=self.preroll_frames)  # ~0.3s rolling buffer
        utterance_frames: list[rtc.AudioFrame] = []
        speech_start_time: float | None = None
        is_speaking = False

        # Background task to push audio frames into VAD and buffers
        async def push_audio():
            nonlocal is_speaking
            try:
                async for frame in frames:
                    vad_stream.push_frame(frame)
                    preroll.append(frame)
                    if is_speaking:
                        utterance_frames.append(frame)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Without this, a crash here kills the feeder silently and
                # process_frames just stops yielding with no indication why.
                logger.exception("push_audio failed for %s", label)
            finally:
                vad_stream.end_input()

        push_task = asyncio.create_task(push_audio())

        try:
            async for event in vad_stream:
                now = time.perf_counter()

                if event.type == agents.vad.VADEventType.START_OF_SPEECH:
                    speech_start_time = now
                    is_speaking = True
                    utterance_frames = list(preroll)
                    if on_speech_start is not None:
                        try:
                            on_speech_start()
                        except Exception:
                            logger.exception("on_speech_start callback failed")

                elif event.type == agents.vad.VADEventType.END_OF_SPEECH:
                    is_speaking = False
                    if utterance_frames:
                        duration = (now - speech_start_time) if speech_start_time else 0.0
                        frames_to_emit = utterance_frames
                        utterance_frames = []
                        yield SpeechUtterance(
                            frames=frames_to_emit,
                            vad_duration_sec=duration,
                            end_time=now,
                        )
                    speech_start_time = None

                elif (
                    is_speaking
                    and speech_start_time is not None
                    and (now - speech_start_time) >= self.max_utterance_duration
                ):
                    # Relies on the VAD stream emitting events continuously
                    # while speech is ongoing (true for silero's frame-by-frame
                    # stream). Forces a flush so one long monologue doesn't
                    # turn into a single very slow Whisper call.
                    if utterance_frames:
                        frames_to_emit = utterance_frames
                        utterance_frames = []
                        yield SpeechUtterance(
                            frames=frames_to_emit,
                            vad_duration_sec=now - speech_start_time,
                            end_time=now,
                        )
                    speech_start_time = now  # restart the clock for the continuing segment
        finally:
            if utterance_frames:
                # Note: can't safely yield here (yielding during generator
                # teardown/GeneratorExit raises a RuntimeError), so trailing
                # speech on track disconnect is logged rather than emitted.
                logger.warning(
                    "Dropping %d buffered frame(s) on %s teardown (utterance in progress)",
                    len(utterance_frames), label,
                )
            push_task.cancel()
            try:
                await push_task
            except asyncio.CancelledError:
                pass