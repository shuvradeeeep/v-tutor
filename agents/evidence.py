"""
Writes every Deps event to a CSV so claims in RIME_EVIDENCE.md point at a file,
not a sentence. One row per event: wall-clock time, ms since session start,
event name, JSON payload. The audio benchmark adds its own columns later.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path


class EvidenceWriter:
    def __init__(self, path: str | Path, session_id: str = "") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        if new:
            self._w.writerow(["timestamp", "elapsed_ms", "session", "event", "payload"])
        self.session_id = session_id
        self._t0 = time.perf_counter()
        self.rows = 0

    def __call__(self, name: str, payload: dict) -> None:
        self._w.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            f"{(time.perf_counter() - self._t0) * 1000:.0f}",
            self.session_id, name,
            json.dumps(payload, ensure_ascii=False, default=str),
        ])
        self._fh.flush()
        self.rows += 1

    def close(self) -> None:
        self._fh.close()
