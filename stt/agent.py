"""
agent.py
Main entrypoint: Connects LiveKit Room -> VAD Segmenter -> Transcriber.
"""

import asyncio
import csv
import logging
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from livekit import agents, rtc

from vad import AudioSegmenter
from transcriber import WhisperTranscriber

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("stt-orchestrator")

# ---- Config (env-overridable so tuning doesn't require code edits) ----
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
CSV_FILE = os.getenv("TRANSCRIPTS_CSV", "transcripts.csv")
MIN_SPEECH_DURATION = float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.1"))
MIN_SILENCE_DURATION = float(os.getenv("VAD_MIN_SILENCE_DURATION", "0.5"))
MAX_UTTERANCE_DURATION = float(os.getenv("VAD_MAX_UTTERANCE_DURATION", "30.0"))

# Model load is expensive (disk + memory) -> do it once at process startup, not per-job.
transcriber = WhisperTranscriber(
    model_size=MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    beam_size=BEAM_SIZE,
)
segmenter = AudioSegmenter(
    min_speech_duration=MIN_SPEECH_DURATION,
    min_silence_duration=MIN_SILENCE_DURATION,
    max_utterance_duration=MAX_UTTERANCE_DURATION,
)

# ---- CSV writer ----
# A single background task owns the file handle and drains a queue. This keeps
# disk I/O (open/write/flush) off the event loop and serializes writes across
# concurrent tracks without needing an explicit lock.
csv_queue: asyncio.Queue = asyncio.Queue()


async def csv_writer_task():
    file_exists = os.path.exists(CSV_FILE)
    csv_handle = open(CSV_FILE, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(csv_handle)

    if not file_exists:
        writer.writerow([
            "timestamp", "participant", "language", "language_confidence",
            "vad_duration_sec", "whisper_ms", "total_latency_ms", "transcript"
        ])
        csv_handle.flush()

    try:
        while True:
            row = await csv_queue.get()
            if row is None:  # shutdown sentinel
                csv_queue.task_done()
                break
            try:
                writer.writerow(row)
                csv_handle.flush()
            except Exception:
                logger.exception("Failed to write transcript row")
            finally:
                csv_queue.task_done()
    finally:
        csv_handle.close()


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
            csv_queue.put_nowait([
                datetime.now().isoformat(timespec="seconds"),
                participant_identity,
                lang,
                f"{prob:.2f}",
                f"{utterance.vad_duration_sec:.2f}",
                f"{whisper_ms:.0f}",
                f"{total_latency_ms:.0f}",
                text,
            ])
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