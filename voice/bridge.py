"""
VoiceBridge: the realtime layer's view of the tutor graph.

Three things happen here and nowhere else:

  1. Fast path. `on_speech_start()` is called by the VAD the moment the learner
     starts talking. It bumps the TurnClock and flushes the player. No graph,
     no LLM, no transcript. Anything the graph is doing right now is stale from
     this instant and will be fenced.

  2. Slow path. `on_transcript()` arrives 300-1500 ms later from Whisper. It is
     delivered to the parked graph as a `user_barge_in` event carrying the heard
     cursor the player recorded at flush time. An empty transcript (a cough)
     still goes in: the graph treats it as a backchannel and resumes.

  3. Cadence. `playback_confirmed` is sent exactly when the graph is parked AND
     the player has drained AND no barge-in happened since the audio was
     queued. Either alone would skip beats or confirm audio nobody heard.

Every graph call runs on ONE worker thread, in order (LangGraph threads are
not re-entrant). The fast path never waits for that thread.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

import config
from agents import llm
from agents.evidence import EvidenceWriter
from agents.gap_filler import GapFiller
from agents.graph import TutorRunner, make_checkpointer
from agents.material import fetch_wikipedia, parse_pdf
from agents.retrieval import make_embedder
from agents.session import Deps, TurnClock
from agents.web import make_web_search
from voice.player import Player
from voice.tts import RimeHTTP, RimeSpeaker

logger = logging.getLogger("v-tutor.voice")


class GraphWorker:
    """One thread owns every graph invocation."""

    def __init__(self, on_idle: Callable[[], None] | None = None) -> None:
        self._q: queue.Queue = queue.Queue()
        self._busy = False
        self._lock = threading.Lock()
        self.on_idle = on_idle
        self.errors: list[BaseException] = []
        self._thread = threading.Thread(target=self._run, name="graph-worker", daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[[], Any], label: str = "") -> None:
        self._q.put((fn, label))

    def idle(self) -> bool:
        with self._lock:
            return not self._busy and self._q.empty()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until the queue is empty and nothing is running (tests, shutdown)."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self.idle():
                return True
            time.sleep(0.01)
        return False

    def close(self) -> None:
        self._q.put(None)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            fn, label = item
            with self._lock:
                self._busy = True
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 -- keep the voice loop alive
                self.errors.append(exc)
                logger.exception("graph call %s failed", label)
            finally:
                with self._lock:
                    self._busy = False
            if self.on_idle:
                try:
                    self.on_idle()
                except Exception:  # noqa: BLE001
                    logger.exception("on_idle failed")


class VoiceBridge:
    def __init__(self, runner: TutorRunner, clock: TurnClock, speaker: RimeSpeaker, player: Player,
                 on_event: Callable[[str, dict], None] | None = None) -> None:
        self.runner = runner
        self.clock = clock
        self.speaker = speaker
        self.player = player
        self.on_event = on_event
        self._lock = threading.Lock()
        self._spoke_since_confirm = False
        self._interrupted = False
        self._closed = False
        self.turns: list[dict] = []                    # per-utterance timing, for evidence
        self._t_speech_start: float | None = None
        self._started = threading.Event()              # set once runner.start() has run on the worker
        self.worker = GraphWorker(on_idle=self._maybe_confirm)
        self.player.set_on_drained(self._maybe_confirm)
        self._wrap_speaker()

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        def _start() -> None:
            try:
                self.runner.start()
            finally:
                self._started.set()
        self.worker.submit(_start, "start")

    @property
    def finished(self) -> bool:
        """True only once the graph has run and reached END. Before the worker
        has executed start(), the checkpoint is empty and would look 'finished'."""
        if not self._started.is_set():
            return False
        try:
            return self.runner.finished
        except Exception:  # noqa: BLE001
            return False

    def wait_idle(self, timeout: float = 30.0) -> bool:
        return self.worker.wait_idle(timeout)

    def close(self) -> None:
        self._closed = True
        self.worker.close()
        self.player.close()

    def _emit(self, name: str, **payload) -> None:
        if self.on_event:
            self.on_event(name, payload)

    # --------------------------------------------------------- speak hook
    def _wrap_speaker(self) -> None:
        """Note every successful enqueue so we know a confirm is owed."""
        orig = self.speaker.speak

        def speak(text: str, **kw) -> None:
            orig(text, **kw)
            if kw.get("turn_id") == self.clock.current():
                with self._lock:
                    self._spoke_since_confirm = True
        self.speaker.speak = speak  # type: ignore[method-assign]

    # ------------------------------------------------------------ fast path
    def on_speech_start(self) -> None:
        """VAD: the learner started talking. Stop NOW; the words come later."""
        t = time.perf_counter()
        turn = self.clock.bump()
        with self._lock:
            self._spoke_since_confirm = False          # nothing queued counts as heard-through
            self._interrupted = True
            self._t_speech_start = t
        self.speaker.stop()
        cur = self.speaker.last_cursor
        stop_ms = (time.perf_counter() - t) * 1000
        self._emit("vad_start", turn=turn, stop_ms=round(stop_ms, 2),
                   cursor=(cur.__dict__ if cur else None))

    # ------------------------------------------------------------ slow path
    def on_transcript(self, text: str, lang: str | None = None, prob: float = 0.0,
                      duration_s: float = 0.0, whisper_ms: float = 0.0) -> None:
        text = (text or "").strip()
        with self._lock:
            interrupted = self._interrupted
            self._interrupted = False
            t0 = self._t_speech_start
            self._t_speech_start = None
        if not interrupted:
            # Transcript without a VAD start (should not happen); make the stop happen anyway.
            self.clock.bump()
            self.speaker.stop()
        cur = self.speaker.take_cursor()
        ev: dict = {"type": "user_barge_in", "text": text}
        if lang in config.SUPPORTED_LANGS:
            ev["detected_lang"] = lang
        if cur is not None and cur.words_heard is not None:
            # Interrupted mid-item. If it was not the last item queued, the
            # learner heard none of the later ones (the lesson beat included).
            ev["cursor"] = {"word_index": cur.words_heard if cur.was_last_item else 0}
        rec = {"text": text, "lang": lang, "prob": round(prob, 2), "vad_s": round(duration_s, 2),
               "whisper_ms": round(whisper_ms), "cursor": ev.get("cursor"),
               "since_vad_start_ms": round((time.perf_counter() - t0) * 1000) if t0 else None}
        self.turns.append(rec)
        self._emit("transcript", **rec)

        def deliver() -> None:
            t = time.perf_counter()
            self.runner.send(ev)
            self._emit("graph_turn_done", ms=round((time.perf_counter() - t) * 1000),
                       intent=self.runner.state.get("intent"), turn=self.runner.state.get("turn_id"))
        self.worker.submit(deliver, f"barge_in:{text[:30]}")

    # -------------------------------------------------------------- cadence
    def _maybe_confirm(self) -> None:
        if self._closed:
            return
        with self._lock:
            owed = self._spoke_since_confirm
            if not owed or not self.worker.idle() or not self.player.is_idle():
                return
            if self.player.last_turn() is not None and self.player.last_turn() != self.clock.current():
                return                                 # a barge-in landed since this audio was queued
            if self.finished:
                self._spoke_since_confirm = False
                return
            self._spoke_since_confirm = False
        self._emit("playback_confirmed", turn=self.clock.current())
        self.worker.submit(self.runner.confirm_playback, "playback_confirmed")


# --------------------------------------------------------------------------
# Construction: everything the harness wires up, but with a real Player.
# --------------------------------------------------------------------------

def make_bridge(player: Player, *, session_id: str, pdf_paths: list[str] | None = None,
                preset_lang: str | None = None, evidence_path: str | Path | None = None,
                checkpoint_db: str | None = None, on_text: Callable[[str, dict], None] | None = None,
                on_event: Callable[[str, dict], None] | None = None, stress_delay_ms: int | None = None,
                synth: RimeHTTP | None = None, deps_overrides: dict | None = None) -> VoiceBridge:
    evidence = EvidenceWriter(evidence_path, session_id) if evidence_path else None

    def emit(name: str, p: dict) -> None:
        if evidence:
            evidence(name, p)
        if on_event:
            on_event(name, p)

    clock = TurnClock()
    speaker = RimeSpeaker(player, clock, synth=synth, on_text=on_text, on_event=emit)
    kw: dict = dict(
        clock=clock, speaker=speaker, llm_fast=llm.fast(), llm_strong=llm.strong(),
        embedder=make_embedder(config.EMBEDDER), web_search=make_web_search(),
        wiki_fetch=fetch_wikipedia, pdf_parse=parse_pdf,
        stress_delay_ms=config.STRESS_DELAY_MS if stress_delay_ms is None else stress_delay_ms,
        on_event=emit, gap_filler=GapFiller(clock, config.GAP_FILLER_DEADLINE_MS),
    )
    kw.update(deps_overrides or {})
    deps = Deps(**kw)
    runner = TutorRunner(deps, session_id, pdf_paths=pdf_paths, preset_lang=preset_lang,
                         checkpointer=make_checkpointer(checkpoint_db))
    bridge = VoiceBridge(runner, clock, speaker, player, on_event=emit)
    bridge.evidence = evidence  # type: ignore[attr-defined]
    return bridge
