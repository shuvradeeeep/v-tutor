"""
agent.py
Main entrypoint: Connects LiveKit Room -> VAD Segmenter -> Transcriber.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents, rtc

# Runnable as `python stt/agent.py dev` or from inside stt/; either way the
# repo root goes on the path so this and voice/audio_io.py load the same
# modules (and therefore the same settings) rather than two copies.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()  # before settings: it reads the environment at import time

from stt import settings  # noqa: E402
from stt.transcriber import WhisperTranscriber  # noqa: E402
from stt.transcripts import TranscriptLog, row as transcript_row  # noqa: E402
from stt.vad import AudioSegmenter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("stt-orchestrator")

# ---- Config ----
# All knobs (and their env overrides) live in stt/settings.py, shared with
# voice/audio_io.py so the two can't drift apart.
CSV_FILE = settings.TRANSCRIPTS_CSV

# Model load is expensive (disk + memory) -> do it once at process startup, not per-job.
transcriber = WhisperTranscriber(
    model_size=settings.WHISPER_MODEL_SIZE,
    device=settings.WHISPER_DEVICE,
    compute_type=settings.WHISPER_COMPUTE_TYPE,
    beam_size=settings.WHISPER_BEAM_SIZE,
    cpu_threads=settings.WHISPER_CPU_THREADS,
    single_pass=settings.WHISPER_SINGLE_PASS,
)
if settings.WHISPER_WARMUP:
    # Pay the lazy-init cost now instead of on the first real utterance.
    transcriber.warmup()
segmenter = AudioSegmenter(
    min_speech_duration=settings.VAD_MIN_SPEECH_DURATION,
    min_silence_duration=settings.VAD_MIN_SILENCE_DURATION,
    max_utterance_duration=settings.VAD_MAX_UTTERANCE_DURATION,
)

# ---- CSV writer ----
# A single background task owns the log and drains a queue. This keeps disk I/O
# (write/flush) off the event loop and serializes writes across concurrent
# tracks. The log itself is stt/transcripts.py, shared with the tutor, so both
# entrypoints append the same schema to the same file.
csv_queue: asyncio.Queue = asyncio.Queue()
transcript_log = TranscriptLog(CSV_FILE)


async def csv_writer_task():
    try:
        while True:
            row = await csv_queue.get()
            if row is None:  # shutdown sentinel
                csv_queue.task_done()
                break
            try:
                transcript_log.append_row(row)
            finally:
                csv_queue.task_done()
    finally:
        transcript_log.close()


# ---- Background task bookkeeping ----
# create_task() results must be held onto or they can be garbage-collected
# mid-run, and any exception inside them otherwise vanishes silently.
background_tasks: set[asyncio.Task] = set()


def _track(task: asyncio.Task):
    background_tasks.add(task)

    def _on_done(t: asyncio.Task):
        background_tasks.discard(t)
        if not t.cancelled() and t.exception() is not None:
            logger.error("Background task failed", exc_info=t.exception())

    task.add_done_callback(_on_done)


async def handle_track(track: rtc.Track, participant_identity: str):
    """
    Consumes utterances yielded by the VAD segmenter and passes them to Whisper.
    Whisper inference is CPU-bound and blocking, so it's run in a worker thread
    -- otherwise one participant's transcription stalls VAD/audio processing
    for every other participant in the room.
    """
    async for utterance in segmenter.process_track(track):
        logger.info(
            "[VAD] Utterance detected from %s (%.2fs)",
            participant_identity, utterance.vad_duration_sec,
        )

        try:
            text, lang, prob, whisper_ms = await asyncio.to_thread(
                transcriber.transcribe, utterance.frames
            )
        except Exception:
            logger.exception("Whisper transcription failed for %s", participant_identity)
            continue

        total_latency_ms = (time.perf_counter() - utterance.end_time) * 1000

        if text:
            logger.info("TRANSCRIPT[%s]: %s", participant_identity, text)
            logger.info(
                "[LATENCY] participant=%s lang=%s whisper=%.0fms total=%.0fms",
                participant_identity, lang, whisper_ms, total_latency_ms,
            )
            # Non-blocking: hands the row to the writer task instead of doing
            # file I/O inline on the event loop.
            if settings.LOG_TRANSCRIPTS:
                csv_queue.put_nowait(transcript_row(
                    participant_identity, text, lang, prob,
                    utterance.vad_duration_sec, whisper_ms, total_latency_ms,
                ))
        else:
            logger.info("[Whisper] No speech detected in segment from %s.", participant_identity)


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    logger.info("Connected to room: %s", ctx.room.name)

    writer_task = asyncio.create_task(csv_writer_task())
    _track(writer_task)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info("Subscribed to audio from %s", participant.identity)
            task = asyncio.create_task(handle_track(track, participant.identity))
            _track(task)

    @ctx.room.on("disconnected")
    def on_disconnected():
        logger.info("Room disconnected, shutting down background tasks")
        for task in list(background_tasks):
            if task is not writer_task:
                task.cancel()
        csv_queue.put_nowait(None)  # let the writer drain and close cleanly


if __name__ == "__main__":
    options = agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="stt-agent",
    )
    agents.cli.run_app(options)