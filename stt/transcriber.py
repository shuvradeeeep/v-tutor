"""
transcriber.py
Handles audio format conversion and Whisper inference.
"""

import time
import numpy as np
from faster_whisper import WhisperModel
from livekit import rtc

# If running on Python 3.13+, make sure `pip install audioop-lts` is installed.
import audioop


class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        cpu_threads: int = 0,
    ):
        """
        Loads the faster-whisper model into memory once upon startup.

        beam_size: the main latency/accuracy knob for decoding. Lower it
        (e.g. 1, greedy decoding) for noticeably faster responses at a small
        accuracy cost; the default of 5 matches faster-whisper's own default.
        cpu_threads: 0 lets CTranslate2 pick based on available cores. Set
        explicitly if you're running multiple model instances on one box and
        want to avoid them fighting over threads.
        """
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        self.beam_size = beam_size

    def frames_to_float32(self, frames: list[rtc.AudioFrame]) -> np.ndarray:
        """
        Converts raw LiveKit PCM frames to 16kHz mono float32 numpy array.
        Whisper strictly expects 16,000 samples per second in [-1.0, 1.0].
        """
        if not frames:
            return np.array([], dtype=np.float32)

        pcm_bytes = b"".join(f.data.tobytes() for f in frames)
        in_rate = frames[0].sample_rate
        num_channels = frames[0].num_channels

        # 1. Downmix to mono if stereo
        if num_channels > 1:
            pcm_bytes = audioop.tomono(pcm_bytes, 2, 0.5, 0.5)

        # 2. Resample to 16kHz
        if in_rate != 16000:
            pcm_bytes, _ = audioop.ratecv(pcm_bytes, 2, 1, in_rate, 16000, None)

        # 3. Normalize 16-bit integers to float32 in [-1.0, 1.0]
        int16_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        return int16_array.astype(np.float32) / 32768.0

    def transcribe(self, frames: list[rtc.AudioFrame]) -> tuple[str, str, float, float]:
        """
        Runs inference on captured audio frames.

        This is a blocking, CPU-bound call. The caller (agent.py) runs it via
        asyncio.to_thread rather than awaiting it directly, so it doesn't
        stall the event loop -- and therefore doesn't stall VAD/audio
        processing for other participants in the room -- while it runs.

        Returns: (transcribed_text, detected_language, language_prob, inference_duration_ms)
        """
        audio_np = self.frames_to_float32(frames)
        if audio_np.size == 0:
            return "", "", 0.0, 0.0

        t_start = time.perf_counter()
        segments, info = self.model.transcribe(
            audio_np,
            language=None,
            beam_size=self.beam_size,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        t_end = time.perf_counter()

        duration_ms = (t_end - t_start) * 1000
        return text, info.language, info.language_probability, duration_ms