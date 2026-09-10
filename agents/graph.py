"""
Graph wiring + a runner that drives it the way the realtime layer will.

The runner is the seam between audio and reasoning:
  * barge_in(text, words_heard)  -- what the VAD/STT path calls
  * confirm_playback()           -- what the playback engine calls
Both resume the parked graph with a Command. barge_in also performs the stop
(clock bump + speaker.stop()) BEFORE resuming, exactly like production.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

try:  # langgraph >= 1.0
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver  # type: ignore

from agents.material import detect_lang
from agents.nodes import TutorNodes
from agents.session import Deps
from agents.state import TutorState, initial_state


def make_checkpointer(path: str | None = None):
    """SQLite when a path is given (sessions survive a restart), memory otherwise."""
    if not path:
        return InMemorySaver()
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def build_graph(deps: Deps, checkpointer: Any | None = None):
    n = TutorNodes(deps)
    g = StateGraph(TutorState)

    for name in (
        "choose_language", "choose_source", "parse_pdf", "fetch_material", "ingest_material",
        "teach_step", "await_event", "handle_interrupt", "classify_intent", "clarify",
        "session_handler", "command_handler", "find_section", "explain", "qa_retrieve",
        "session_status",
        "direct_answer", "web_search_node", "compose_answer", "discard", "resume_controller",
        "promote_queued",
    ):
        g.add_node(name, getattr(n, name))

    # ---- onboarding: two spoken questions, then material -------------------
    g.add_edge(START, "choose_language")
    g.add_conditional_edges("choose_language", n.route_onboarding,
                            {"ask": "resume_controller", "next": "choose_source", "quit": END})
    g.add_conditional_edges("choose_source", n.route_onboarding,
                            {"ask": "resume_controller", "pdf": "parse_pdf", "topic": "fetch_material",
                             "quit": END})
    g.add_conditional_edges("parse_pdf", n.route_parse,
                            {"ok": "ingest_material", "unreadable": "resume_controller"})
    g.add_conditional_edges("fetch_material", n.route_fetch,
                            {"ok": "ingest_material", "not_found": "resume_controller"})
    g.add_edge("ingest_material", "teach_step")

    # ---- lesson loop --------------------------------------------------------
    g.add_conditional_edges("teach_step", n.route_after_teach, {
        "wait": "await_event",
        "queued": "promote_queued",       # "go to X *and* explain it": read X, then explain
        "finished": END,
    })
    g.add_edge("promote_queued", "classify_intent")
    g.add_conditional_edges("await_event", n.route_event, {
        "playback_confirmed": "teach_step",
        "user_barge_in": "handle_interrupt",
        "stay_parked": "await_event",
        "lesson_complete": END,
    })
    g.add_conditional_edges("handle_interrupt", n.route_after_interrupt, {
        "onboarding_language": "choose_language",
        "onboarding_source": "choose_source",
        "lesson": "classify_intent",
    })

    # ---- seven intents -------------------------------------------------------
    g.add_conditional_edges("classify_intent", n.route_intent, {
        "unknown": "clarify",
        "session": "session_handler",
        "command": "command_handler",
        "navigate": "find_section",
        "explain": "explain",
        "question": "qa_retrieve",
        "backchannel": "resume_controller",
        "meta": "session_status",       # "how long will this take?"
    })
    g.add_conditional_edges("session_handler", n.route_session, {
        "pause": "await_event",
        "continue": "teach_step",
        "restart": "teach_step",
        "quit": END,
    })
    g.add_conditional_edges("find_section", n.route_nav,
                            {"found": "teach_step", "not_found": "resume_controller",
                             "switch": "fetch_material"})   # "I wanted respiration, not reproduction"
    # Not in the notes: ask the strong model first (trivial / general-knowledge
    # questions need no search); it says LOOKUP when the web is really needed.
    g.add_conditional_edges("qa_retrieve", n.route_retrieval,
                            {"grounded": "compose_answer", "direct": "direct_answer",
                             "needs_web": "web_search_node"})
    g.add_conditional_edges("direct_answer", n.route_direct,
                            {"answered": "fence_direct", "lookup": "web_search_node"})
    g.add_edge("web_search_node", "compose_answer")

    # ---- every path that produces speech passes the fence -------------------
    fenced = {"current": "resume_controller", "stale": "discard"}
    for node in ("compose_answer", "command_handler", "explain", "clarify", "session_status"):
        g.add_conditional_edges(node, n.fence_check, fenced)
    # direct_answer has two exits, so its fence is a pass-through node.
    g.add_node("fence_direct", lambda state: {})
    g.add_conditional_edges("fence_direct", n.fence_check, fenced)
    g.add_edge("discard", "await_event")

    # ---- after speaking: drain a queued request, resume the lesson, or wait --
    g.add_conditional_edges("resume_controller", n.route_after_speak, {
        "queued": "promote_queued",
        "resume_lesson": "teach_step",
        "done": "await_event",
    })

    return g.compile(checkpointer=checkpointer or InMemorySaver())


class TutorRunner:
    """Drives one session. Used by the text harness, the tests, and main.py."""

    def __init__(self, deps: Deps, session_id: str = "session-1",
                 pdf_paths: list[str] | None = None, checkpointer: Any | None = None,
                 recursion_limit: int = 80, preset_lang: str | None = None) -> None:
        self.deps = deps
        self.session_id = session_id
        self.pdf_paths = pdf_paths or []
        self.preset_lang = preset_lang
        self.app = build_graph(deps, checkpointer)
        self.config = {"configurable": {"thread_id": session_id}, "recursion_limit": recursion_limit}

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> dict:
        """Begin a session, or resume one the checkpointer already knows about."""
        if self.app.get_state(self.config).values:
            return self.state                       # resumed: graph is parked where it was
        self.app.invoke(initial_state(self.session_id, self.pdf_paths, preset_lang=self.preset_lang),
                        self.config)
        return self.state

    def send(self, event: dict) -> dict:
        self.app.invoke(Command(resume=event), self.config)
        return self.state

    # -- what the realtime layer calls ------------------------------------------
    def barge_in(self, text: str, words_heard: int | None = None,
                 detected_lang: str | None = None) -> dict:
        """The learner spoke. Stop first (as the VAD callback does), then tell the graph."""
        self.deps.clock.bump()
        self.deps.speaker.stop()
        ev: dict = {"type": "user_barge_in", "text": text,
                    "detected_lang": detected_lang or detect_lang(text)}
        if words_heard is not None:
            ev["cursor"] = {"word_index": int(words_heard)}
        return self.send(ev)

    def press_language_button(self, lang_code: str) -> dict:
        """A screen button during the opening question. Same path as speech."""
        from config import LANG_NAMES
        return self.barge_in(LANG_NAMES.get(lang_code, lang_code), detected_lang=lang_code)

    def confirm_playback(self) -> dict:
        return self.send({"type": "playback_confirmed"})

    def load_pdf(self, paths: list[str]) -> dict:
        """Inject new PDF paths at runtime and trigger the graph to teach from them.

        Sets ``onboarding_step = "source"`` so that ``route_after_interrupt``
        sends the next event to ``choose_source``, which immediately detects
        ``pdf_paths`` is set and routes to ``parse_pdf → ingest → teach``.
        """
        self.pdf_paths = paths
        # Update the live graph state: inject paths, set onboarding_step to
        # "source" so the next user_barge_in arrives at choose_source, and
        # clear any finished lesson so the graph does not think we're done.
        self.app.update_state(
            self.config,
            {"pdf_paths": paths, "onboarding_step": "source",
             "awaiting_document": False,
             "beats": [], "beat_index": 0, "beat_spoken": False,
             "topic": "", "source_title": "", "sections": [], "section_index": 0,
             "paused": False, "finished": False},
        )
        # Drive the graph: a synthetic barge-in wakes await_event.
        # route_after_interrupt sees onboarding_step="source" → choose_source.
        # choose_source sees pdf_paths set → parse_pdf → ingest → teach.
        return self.send({"type": "user_barge_in", "text": ""})

    # -- introspection ----------------------------------------------------------
    @property
    def state(self) -> dict:
        return self.app.get_state(self.config).values

    @property
    def finished(self) -> bool:
        return not self.app.get_state(self.config).next
