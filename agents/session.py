"""
Live, in-process objects that sit OUTSIDE the graph.

Two things cannot live in graph state because they must change while a node is
mid-run:

  * TurnClock -- the barge-in counter. The VAD callback bumps it the instant the
    learner speaks, even if web_search_node is still sleeping. fence_check
    compares a branch's birth turn against this live value.
  * Speaker  -- the only way text becomes audio. Every speak() call is fenced.

`Deps` bundles these with the swappable services (LLMs, retriever, fetchers) so
the graph can be built identically for production, the text harness, and tests.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class TurnClock:
    """Monotonic barge-in counter. Thread-safe. Bumped by the realtime layer."""

    def __init__(self) -> None:
        self._id = 0
        self._lock = threading.Lock()

    def current(self) -> int:
        with self._lock:
            return self._id

    def bump(self) -> int:
        with self._lock:
            self._id += 1
            return self._id


@dataclass
class SpokenLine:
    turn_id: int
    text: str
    lang: str
    speaker: str
    speed_alpha: float
    t: float = field(default_factory=time.perf_counter)


class Speaker(Protocol):
    """Anything that can turn text into speech. RimeSpeaker in production,
    TextSpeaker in the harness and tests."""

    def speak(self, text: str, *, turn_id: int, lang: str, speaker: str,
              speed_alpha: float) -> None: ...

    def stop(self) -> None:
        """Barge-in: stop playback immediately (server clear + local flush)."""
        ...


class TextSpeaker:
    """Records every line and optionally prints it. Used by the harness/tests."""

    def __init__(self, echo: bool = False) -> None:
        self.lines: list[SpokenLine] = []
        self.echo = echo
        self.stops = 0

    def speak(self, text: str, *, turn_id: int, lang: str, speaker: str,
              speed_alpha: float) -> None:
        line = SpokenLine(turn_id, text, lang, speaker, speed_alpha)
        self.lines.append(line)
        if self.echo:
            speed = "" if abs(speed_alpha - 1.0) < 1e-9 else f" x{speed_alpha:.2f}"
            print(f"\n  [tutor | turn {turn_id} | {lang} | {speaker}{speed}]\n  {text}\n")

    def stop(self) -> None:
        self.stops += 1

    def since(self, index: int) -> list[SpokenLine]:
        return self.lines[index:]


# --------------------------------------------------------------------------
# Service protocols. Concrete implementations live in llm.py / retrieval.py /
# material.py; tests pass lambdas.
# --------------------------------------------------------------------------

class LLM(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return text. Return "" to signal 'unavailable' -- every node has a
        deterministic fallback for that case, so the tutor keeps working."""
        ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> Any: ...   # -> np.ndarray [n, d]


WebSearchFn = Callable[[str], list[dict]]             # -> [{"title","snippet","url"}]
WikiFetchFn = Callable[[str, str], "tuple[list[dict], str, str] | None"]
PdfParseFn = Callable[[list[str]], "tuple[list[dict], str, str] | None"]


@dataclass
class Deps:
    clock: TurnClock
    speaker: Speaker
    llm_fast: LLM
    llm_strong: LLM
    embedder: Embedder
    web_search: WebSearchFn
    wiki_fetch: WikiFetchFn
    pdf_parse: PdfParseFn
    stress_delay_ms: int = 0
    gap_filler: Any | None = None       # agents.gap_filler.GapFiller, optional
    # per-session heavy objects: {session_id: {"retriever": ..., "section_retriever": ...}}
    stores: dict[str, dict[str, Any]] = field(default_factory=dict)
    # optional observer for evidence: called with (event_name, payload)
    on_event: Callable[[str, dict], None] | None = None

    def emit(self, name: str, **payload: Any) -> None:
        if self.on_event:
            self.on_event(name, payload)
