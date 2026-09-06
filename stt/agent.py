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

# CSV Setup
CSV_FILE = "transcripts.csv"
file_exists = os.path.exists(CSV_FILE)
csv_handle = open(CSV_FILE, mode="a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_handle)

if not file_exists:
    csv_writer.writerow([
        "timestamp", "language", "language_confidence",
        "vad_duration_sec", "whisper_ms", "total_latency_ms", "transcript"
    ])
    csv_handle.flush()

# Initialize Transcriber & Segmenter instances once
transcriber = WhisperTranscriber(model_size="base", device="cpu", compute_type="int8")
segmenter = AudioSegmenter(min_speech_duration=0.1, min_silence_duration=0.5)


async def handle_track(track: rtc.Track):
    """
    Consumes utterances yielded by the VAD segmenter and passes them to Whisper.
    """
    async for utterance in segmenter.process_track(track):
        logger.info("[VAD] Utterance detected (%.2fs)", utterance.vad_duration_sec)

        text, lang, prob, whisper_ms = transcriber.transcribe(utterance.frames)
        total_latency_ms = (time.perf_counter() - utterance.end_time) * 1000

        if text:
            logger.info("TRANSCRIPT: %s", text)
            logger.info("[LATENCY] lang=%s whisper=%.0fms total=%.0fms", lang, whisper_ms, total_latency_ms)

            csv_writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                lang,
                f"{prob:.2f}",
                f"{utterance.vad_duration_sec:.2f}",
                f"{whisper_ms:.0f}",
                f"{total_latency_ms:.0f}",
                text,
            ])
            csv_handle.flush()
        else:
            logger.info("[Whisper] No speech detected in segment.")


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    logger.info("Connected to room: %s", ctx.room.name)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info("Subscribed to audio from %s", participant.identity)
            asyncio.create_task(handle_track(track))


if __name__ == "__main__":
    options = agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="stt-agent",
    )
    agents.cli.run_app(options)