"""
vad.py
Handles Voice Activity Detection (VAD) and speech audio segmentation.
"""

import collections
import time
from typing import AsyncGenerator
from dataclasses import dataclass

from livekit import agents, rtc
from livekit.plugins import silero


@dataclass
class SpeechUtterance:
    frames: list[rtc.AudioFrame]
    vad_duration_sec: float
    end_time: float


class AudioSegmenter:
    def __init__(self, min_speech_duration: float = 0.1, min_silence_duration: float = 0.5):
        """
        Sets up Silero VAD parameters.
        """
        self.vad = silero.VAD.load(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
        )

    async def process_track(self, track: rtc.Track) -> AsyncGenerator[SpeechUtterance, None]:
        """
        Listens to a LiveKit track, processes frames through Silero VAD,
        and yields completed SpeechUtterance instances whenever the user finishes talking.
        """
        vad_stream = self.vad.stream()
        preroll = collections.deque(maxlen=15)  # ~0.3s rolling buffer
        utterance_frames: list[rtc.AudioFrame] = []
        speech_start_time: float | None = None
        is_speaking = False

        audio_stream = rtc.AudioStream(track)

        # Background task to push audio frames into VAD and buffers
        async def push_audio():
            nonlocal is_speaking
            async for audio_event in audio_stream:
                frame = audio_event.frame
                vad_stream.push_frame(frame)
                preroll.append(frame)
                if is_speaking:
                    utterance_frames.append(frame)

        import asyncio
        push_task = asyncio.create_task(push_audio())

        try:
            async for event in vad_stream:
                if event.type == agents.vad.VADEventType.START_OF_SPEECH:
                    speech_start_time = time.perf_counter()
                    is_speaking = True
                    utterance_frames = list(preroll)

                elif event.type == agents.vad.VADEventType.END_OF_SPEECH:
                    is_speaking = False
                    t_end = time.perf_counter()
                    duration = (t_end - speech_start_time) if speech_start_time else 0.0

                    frames_to_emit = list(utterance_frames)
                    utterance_frames.clear()

                    if frames_to_emit:
                        yield SpeechUtterance(
                            frames=frames_to_emit,
                            vad_duration_sec=duration,
                            end_time=t_end,
                        )
        finally:
            push_task.cancel()