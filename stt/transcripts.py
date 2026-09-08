"""
transcripts.py
The transcript log both entrypoints append to: stt/agent.py (standalone STT
worker) and voice/audio_io.py (the tutor's ear). One file, one schema, so a
tutor session and a bare STT session produce comparable rows.

Columns match the file that already exists (stt/transcripts.csv):

    timestamp, participant, language, language_confidence,
    vad_duration_sec, whisper_ms, total_latency_ms, transcript

`total_latency_ms` is what the learner actually waits: end of speech (the VAD's
END_OF_SPEECH) to transcript in hand. `whisper_ms` is the inference alone --
the gap between the two is buffering and orchestration, normally a few ms.

The handle is opened on first write and every row is flushed, so `Get-Content
-Wait stt\\transcripts.csv` follows a live session.
"""

from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("stt-orchestrator")

HEADER = [
    "timestamp", "participant", "language", "language_confidence",
    "vad_duration_sec", "whisper_ms", "total_latency_ms", "transcript",
]


def row(participant: str, text: str, language: str | None, probability: float,
        vad_duration_sec: float, whisper_ms: float, total_latency_ms: float) -> list[str]:
    """Format one utterance. Kept separate from writing so a caller can build
    the row on the event loop and hand it to a writer thread/task."""
    return [
        datetime.now().isoformat(timespec="seconds"),
        participant,
        language or "",
        f"{probability:.2f}",
        f"{vad_duration_sec:.2f}",
        f"{whisper_ms:.0f}",
        f"{total_latency_ms:.0f}",
        text,
    ]


class TranscriptLog:
    """Append-only CSV. Safe to call from several threads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = 0
        self._lock = threading.Lock()
        self._handle = None
        self._writer = None

    def _ensure_open(self) -> None:
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.path.exists() or self.path.stat().st_size == 0
        self._handle = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        if fresh:
            self._writer.writerow(HEADER)
            self._handle.flush()

    def append_row(self, values: list[str]) -> None:
        with self._lock:
            try:
                self._ensure_open()
                self._writer.writerow(values)
                self._handle.flush()
                self.rows += 1
            except Exception:
                # A failed transcript write must never take the voice loop down.
                logger.exception("failed to write transcript row to %s", self.path)

    def append(self, participant: str, text: str, language: str | None, probability: float,
               vad_duration_sec: float, whisper_ms: float, total_latency_ms: float) -> None:
        self.append_row(row(participant, text, language, probability,
                            vad_duration_sec, whisper_ms, total_latency_ms))

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
                self._writer = None
