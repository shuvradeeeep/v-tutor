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

import re
from collections import deque

import config
from agents import intent, llm
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


# --------------------------------------------------------------------------
# Self-echo guard
#
# There is no acoustic echo cancellation on the local path: without headphones
# the mic hears the tutor, Whisper transcribes it, and the tutor answers its
# own voice. A real session degenerated in four turns -- "Let me check that."
# came back as "Check that.", "I'm not sure." as "sure." -- and the tutor
# ended up teaching an article about a musician called Dean Blunt.
#
# A transcript that is a run of words the tutor just said is treated as
# nothing heard. The bridge already delivers an empty transcript as a
# backchannel, so the lesson resumes instead of derailing. Barge-in still
# works: the stop happened on VAD start, and anything the tutor did not just
# say passes through untouched.
# --------------------------------------------------------------------------

def _words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", (text or "").lower(), re.UNICODE)


def looks_like_echo(heard: str, spoken: str) -> bool:
    """True when every word heard appears, in order and adjacent, in `spoken`."""
    h, s = _words(heard), _words(spoken)
    if not h or len(h) > len(s):
        return False
    if len(h) == 1 and len(h[0]) < 4:
        return False              # "a", "is" -- too common to blame on the speaker
    if is_reply_word(heard):
        # The tutor says "say continue when you're ready" and the learner says
        # "continue" -- an invited reply, not an echo. Dropping it left the
        # lesson paused and, worse, counted towards disabling barge-in.
        return False
    return any(s[i:i + len(h)] == h for i in range(len(s) - len(h) + 1))


def is_reply_word(text: str) -> bool:
    """
    A short utterance the rules recognise as an instruction to the tutor.

    Backchannels are deliberately NOT included: an echoed "sure" and a real
    "sure" both mean "carry on", so nothing is lost by treating it as an echo,
    and the echo counter stays honest. Losing a "continue" or a "stop" is a
    different matter -- that leaves the lesson stuck.
    """
    if len(_words(text)) > 3:
        return False
    c = intent.classify_rules(text)
    return bool(c and (c.session_cmd or c.command))


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
                 on_event: Callable[[str, dict], None] | None = None,
                 half_duplex: bool | None = None) -> None:
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
        self._recent_speech: deque[tuple[float, str]] = deque(maxlen=6)   # echo guard
        self.echo_drops = 0
        self.long_echo_drops = 0
        # Headphones: full duplex, barge-in works. Speakers: the mic hears the
        # tutor, so listening while speaking has to stop -- either because the
        # caller said so, or because we caught enough echoes to be sure.
        self.half_duplex = config.HALF_DUPLEX if half_duplex is None else half_duplex
        self._suppressed_start = False
        self.suppressed = 0
        self._held: dict | None = None                 # an incomplete transcript waiting for its second half
        self._hold_timer: threading.Timer | None = None
        self.holds = 0
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
        with self._lock:
            if self._hold_timer is not None:
                self._hold_timer.cancel()
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
                    self._recent_speech.append((time.perf_counter(), text))
        self.speaker.speak = speak  # type: ignore[method-assign]

    def _is_self_echo(self, text: str) -> str | None:
        """The line the tutor just said that this transcript is an echo of."""
        if not config.ECHO_GUARD or not text:
            return None
        now = time.perf_counter()
        with self._lock:
            recent = [(t, s) for t, s in self._recent_speech if now - t <= config.ECHO_GUARD_SEC]
        for _, spoken in reversed(recent):
            if looks_like_echo(text, spoken):
                return spoken
        return None

    # ------------------------------------------------------------ fast path
    def on_speech_start(self) -> None:
        """VAD: the learner started talking. Stop NOW; the words come later."""
        t = time.perf_counter()
        if self.half_duplex and not self.player.is_idle():
            # Speakers, not headphones: most of what the mic hears right now is
            # the tutor itself, so stopping on the VAD alone would make every
            # beat interrupt itself. Do not stop yet -- wait for the words. If
            # they turn out to be an echo they are dropped; if the learner
            # really did speak, on_transcript stops then. The interruption
            # costs one Whisper pass instead of a millisecond, which is the
            # price of not wearing headphones.
            self.suppressed += 1
            self._suppressed_start = True
            with self._lock:
                self._t_speech_start = t   # still measure VAD start -> transcript
            self._emit("barge_in_deferred", reason="half_duplex")
            return
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
        # A deferred start did not stop playback. If these words survive the
        # echo guard they are real speech, and the "no VAD start" branch below
        # performs the stop before the graph is told anything.
        deferred, self._suppressed_start = self._suppressed_start, False
        if (echoed := self._is_self_echo(text)):
            self.echo_drops += 1
            # Only multi-word echoes count towards the half-duplex switch: a
            # one-word match is the kind that turns out to be a real reply, and
            # disabling barge-in for a whole session off the back of one is far
            # too aggressive. Counted before `text` is cleared below.
            if len(_words(text)) >= 3:
                self.long_echo_drops += 1
            self._emit("echo_drop", heard=text, spoke=echoed[:60], deferred=deferred)
            logger.info("echo: dropped %r (tutor said %r)", text, echoed[:60])
            if deferred:
                return          # nothing was stopped, so there is nothing to resume
            # Playback was already stopped by the VAD. Deliver silence: the
            # graph reads that as a backchannel and picks the lesson back up.
            text = ""
            if (not self.half_duplex and config.ECHO_AUTO_HALF_DUPLEX
                    and self.long_echo_drops >= config.ECHO_AUTO_HALF_DUPLEX):
                # Repeated echoes mean there are no headphones on. Stop
                # listening while speaking, or the lesson stutters forever:
                # every beat gets interrupted by itself and replayed.
                self.half_duplex = True
                self._emit("half_duplex_on", after_echoes=self.long_echo_drops)
                logger.warning("mic is hearing the tutor (%d echoes): barge-in disabled for this "
                               "session. Use headphones to keep it.", self.long_echo_drops)
        with self._lock:
            interrupted = self._interrupted
            self._interrupted = False
            t0 = self._t_speech_start
            self._t_speech_start = None
        if not interrupted:
            # Either a deferred half-duplex start (the stop was held back until
            # we knew this was not an echo) or a transcript with no VAD start
            # at all. Either way, stop now, before the graph is told.
            self.clock.bump()
            self.speaker.stop()
        cur = self.speaker.take_cursor()

        # A held fragment ("and, um,") joins the transcript that followed it.
        with self._lock:
            held, self._held = self._held, None
            if self._hold_timer is not None:
                self._hold_timer.cancel()
                self._hold_timer = None
        holds = 0
        if held:
            text = intent.strip_leading_fillers(f"{held['text']} {text}".strip())
            if cur is None or cur.words_heard is None:
                cur = held["cursor"]                    # the stop that mattered was the first one
            lang = lang or held["lang"]
            t0 = held["t0"] or t0
            duration_s += held["duration_s"]
            holds = held["holds"]
        if (config.FRAGMENT_HOLD_S > 0 and text and holds < config.FRAGMENT_HOLD_MAX
                and intent.is_incomplete(text)):
            # Playback is already stopped, so the tutor stays quiet while the
            # learner finishes the thought. Delivered as-is if nothing follows.
            self.holds += 1
            rec = {"text": text, "lang": lang, "prob": round(prob, 2), "vad_s": round(duration_s, 2),
                   "whisper_ms": round(whisper_ms), "cursor": None, "since_vad_start_ms": None, "held": True}
            self._emit("transcript", **rec)
            with self._lock:
                self._held = {"text": text, "cursor": cur, "lang": lang, "t0": t0, "prob": prob,
                              "duration_s": duration_s, "whisper_ms": whisper_ms, "holds": holds + 1}
                self._hold_timer = threading.Timer(config.FRAGMENT_HOLD_S, self._release_hold)
                self._hold_timer.daemon = True
                self._hold_timer.start()
            self._emit("fragment_hold", text=text, seconds=config.FRAGMENT_HOLD_S)
            return
        self._dispatch(text, lang, prob, duration_s, whisper_ms, cur, t0)

    def _release_hold(self) -> None:
        """Nothing followed the fragment: deliver it as it was."""
        with self._lock:
            held, self._held = self._held, None
            self._hold_timer = None
        if held is None or self._closed:
            return
        self._emit("fragment_release", text=held["text"])
        self._dispatch(held["text"], held["lang"], held["prob"], held["duration_s"], held["whisper_ms"],
                       held["cursor"], held["t0"])

    def _dispatch(self, text: str, lang: str | None, prob: float, duration_s: float, whisper_ms: float,
                  cur, t0: float | None) -> None:
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


    def load_pdf(self, paths: list[str]) -> None:
        """Inject new PDF paths mid-session and trigger the tutor to teach from them."""
        if self._closed or not paths:
            return
        self.speaker.stop()
        self.clock.bump()

        def _deliver() -> None:
            self.runner.load_pdf(paths)
            self._emit("ingest", pdf_paths=paths)
        self.worker.submit(_deliver, f"load_pdf:{paths[0][:40]}")


# --------------------------------------------------------------------------
# Construction: everything the harness wires up, but with a real Player.
# --------------------------------------------------------------------------
def make_bridge(player: Player, *, session_id: str, pdf_paths: list[str] | None = None,
                preset_lang: str | None = None, evidence_path: str | Path | None = None,
                checkpoint_db: str | None = None, on_text: Callable[[str, dict], None] | None = None,
                on_event: Callable[[str, dict], None] | None = None, stress_delay_ms: int | None = None,
                synth: RimeHTTP | None = None, deps_overrides: dict | None = None,
                half_duplex: bool | None = None) -> VoiceBridge:
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
    bridge = VoiceBridge(runner, clock, speaker, player, on_event=emit, half_duplex=half_duplex)
    bridge.evidence = evidence  # type: ignore[attr-defined]
    return bridge
