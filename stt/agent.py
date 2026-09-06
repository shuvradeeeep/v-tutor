"""
STT-only pipeline: LiveKit (transport) + Silero (VAD) + Whisper (speech-to-text)
-- fully local, no OpenAI account/API key needed --

Flow:
    mic audio --> LiveKit room --> Silero VAD detects start/end of speech
              --> buffered speech frames --> local Whisper model transcribes
              --> transcript + per-stage latency printed to console

Whisper here runs locally via `faster-whisper` (a CTranslate2-based
reimplementation of Whisper -- same model weights, much lighter and faster
than the official openai-whisper package, no PyTorch required). The first
run will download the model weights once (a few hundred MB depending on
size) and cache them locally; after that it runs fully offline.

NOTE: LiveKit Agents' Python SDK API has changed across versions. The VAD
event shape used below (VADEventType.START_OF_SPEECH / END_OF_SPEECH,
event.frames) matches the API as of my training data. If you hit
AttributeErrors, check https://docs.livekit.io/agents/ for the current
VAD event interface -- the overall approach (push frames into the VAD
stream, buffer frames between start/end-of-speech, transcribe the buffered
segment) will still apply even if a name shifted.
"""

import asyncio
import audioop
import collections
import csv
import logging
import os
import time
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from livekit import agents, rtc
from livekit.plugins import silero

# Load values from .env into the environment (LIVEKIT_URL, etc.) --
# without this call, having a .env file present does nothing on its own.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),                     # keep printing to console
        logging.FileHandler("agent.log", encoding="utf-8"),  # also save everything to a file
    ],
)
logger = logging.getLogger("stt-only")

# --- Transcript output file (CSV, one row per utterance) ---
# Appends across runs -- doesn't overwrite previous sessions.
TRANSCRIPT_CSV_PATH = "transcripts.csv"
_csv_is_new = not os.path.exists(TRANSCRIPT_CSV_PATH)
_csv_file = open(TRANSCRIPT_CSV_PATH, mode="a", newline="", encoding="utf-8")
_csv_writer = csv.writer(_csv_file)
if _csv_is_new:
    _csv_writer.writerow([
        "timestamp", "language", "language_confidence",
        "vad_utterance_seconds", "whisper_ms", "total_latency_ms", "transcript",
    ])
    _csv_file.flush()


def log_transcript_row(language, language_prob, vad_duration, whisper_ms, total_ms, text):
    _csv_writer.writerow([
        datetime.now().isoformat(timespec="seconds"),
        language,
        f"{language_prob:.2f}",
        f"{vad_duration:.2f}" if vad_duration is not None else "",
        f"{whisper_ms:.0f}",
        f"{total_ms:.0f}",
        text,
    ])
    _csv_file.flush()  # write immediately so you can tail the file live

# --- Local Whisper model, loaded once at startup ---
# Sizes (accuracy vs speed): tiny < base < small < medium < large
# "base" is a reasonable default on CPU. Go smaller ("tiny") for lower
# latency, larger ("small"/"medium") for better accuracy if your machine
# can handle it (much faster with a GPU: device="cuda").
WHISPER_MODEL_SIZE = "base"
logger.info("loading local Whisper model '%s' (first run downloads it)...", WHISPER_MODEL_SIZE)
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
logger.info("Whisper model loaded.")


def frames_to_float32(frames: list[rtc.AudioFrame]) -> np.ndarray:
    """Concatenate buffered LiveKit audio frames into one float32 array
    at 16kHz mono, range [-1, 1] -- the format Whisper expects.

    LiveKit typically captures mic audio at 48kHz. Whisper was trained on
    16kHz audio, so feeding it 48kHz samples directly makes it think 1
    real second of audio is 3 seconds long, and the "speech" it hears is
    effectively pitch-shifted/stretched -- producing garbage transcripts.
    This resamples down to 16kHz (and downmixes to mono) before handing
    audio to Whisper.
    """
    if not frames:
        return np.array([], dtype=np.float32)

    pcm_bytes = b"".join(f.data.tobytes() for f in frames)
    in_rate = frames[0].sample_rate
    num_channels = frames[0].num_channels

    if num_channels > 1:
        pcm_bytes = audioop.tomono(pcm_bytes, 2, 0.5, 0.5)

    if in_rate != 16000:
        pcm_bytes, _ = audioop.ratecv(pcm_bytes, 2, 1, in_rate, 16000, None)

    int16_array = np.frombuffer(pcm_bytes, dtype=np.int16)
    return int16_array.astype(np.float32) / 32768.0


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    logger.info("connected to room: %s", ctx.room.name)

    # --- Silero VAD: flags start-of-speech / end-of-speech in real time ---
    vad = silero.VAD.load(
        min_speech_duration=0.1,   # ignore blips shorter than this
        min_silence_duration=0.5,  # how much silence = "end of utterance"
    )

    async def transcribe_track(track: rtc.Track):
        vad_stream = vad.stream()
        speech_start_time = None
        is_speaking = False
        utterance_frames: list[rtc.AudioFrame] = []
        # Small rolling buffer of recent frames, so the utterance capture
        # includes a bit of audio from just before VAD detects speech
        # started (VAD needs a few frames of context before it fires).
        preroll: collections.deque = collections.deque(maxlen=15)  # ~0.3s

        async def consume_vad_events():
            nonlocal speech_start_time, is_speaking, utterance_frames
            async for event in vad_stream:
                if event.type == agents.vad.VADEventType.START_OF_SPEECH:
                    speech_start_time = time.perf_counter()
                    is_speaking = True
                    utterance_frames = list(preroll)  # seed with pre-speech audio
                    logger.info("[VAD] speech started")

                elif event.type == agents.vad.VADEventType.END_OF_SPEECH:
                    is_speaking = False
                    t_end_of_speech = time.perf_counter()
                    vad_latency = (
                        t_end_of_speech - speech_start_time
                        if speech_start_time else None
                    )
                    logger.info(
                        "[VAD] speech ended (utterance duration: %.2fs)",
                        vad_latency if vad_latency else -1.0,
                    )

                    # Grab this utterance's frames and reset the buffer
                    # immediately, so the NEXT utterance starts clean
                    # instead of accumulating old audio.
                    frames_to_transcribe = utterance_frames
                    utterance_frames = []

                    if not frames_to_transcribe:
                        logger.warning("[VAD] end-of-speech with no buffered audio, skipping")
                        continue

                    logger.info(
                        "[Audio] captured at %dHz, %d channel(s) -- resampling to 16kHz for Whisper",
                        frames_to_transcribe[0].sample_rate,
                        frames_to_transcribe[0].num_channels,
                    )

                    audio_np = frames_to_float32(frames_to_transcribe)
                    t_whisper_start = time.perf_counter()
                    segments, info = whisper_model.transcribe(
                        audio_np,
                        language=None,  # None = auto-detect (multilingual)
                    )
                    text = " ".join(seg.text.strip() for seg in segments).strip()
                    t_whisper_end = time.perf_counter()

                    whisper_latency = t_whisper_end - t_whisper_start
                    total_latency = t_whisper_end - t_end_of_speech

                    if text:
                        logger.info("TRANSCRIPT: %s", text)
                        logger.info(
                            "[LATENCY] language=%s  whisper=%.0fms  "
                            "end_of_speech_to_text=%.0fms",
                            info.language, whisper_latency * 1000, total_latency * 1000,
                        )
                        log_transcript_row(
                            language=info.language,
                            language_prob=info.language_probability,
                            vad_duration=vad_latency,
                            whisper_ms=whisper_latency * 1000,
                            total_ms=total_latency * 1000,
                            text=text,
                        )
                    else:
                        logger.info("[Whisper] no speech detected in buffered audio")

        asyncio.create_task(consume_vad_events())

        audio_stream = rtc.AudioStream(track)
        async for audio_event in audio_stream:
            frame = audio_event.frame
            vad_stream.push_frame(frame)
            preroll.append(frame)
            if is_speaking:
                utterance_frames.append(frame)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info("subscribed to audio track from %s", participant.identity)
            asyncio.create_task(transcribe_track(track))


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
    
    