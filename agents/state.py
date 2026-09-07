"""
Graph state for the tutor. Everything here must be JSON-serialisable so the
LangGraph checkpointer can persist it -- no numpy arrays, no live objects.
Heavy per-session objects (the retriever, embeddings) live in `Deps.stores`,
keyed by session_id. See agents/session.py.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

Intent = Literal[
    "question", "explain", "navigate", "command", "session", "backchannel", "unknown"
]
Command = Literal["repeat", "slower", "faster", "switch_lesson_lang"]
SessionCmd = Literal["pause", "continue", "restart", "quit"]
OnboardingStep = Literal["language", "source", "done"]
EventType = Literal["user_barge_in", "playback_confirmed", "lesson_complete"]


class Cursor(TypedDict, total=False):
    """Where the learner actually stopped hearing. Produced by the playback
    engine (or the text harness), frozen into state by handle_interrupt."""
    beat_index: int
    sentence_index: int
    word_index: int        # last word fully heard, within the beat
    char_offset: int
    seconds: float


class Event(TypedDict, total=False):
    """What the realtime layer hands to await_event on resume."""
    type: EventType
    text: str              # transcript, for user_barge_in
    detected_lang: str     # from STT
    cursor: Cursor         # from the playback engine


class Beat(TypedDict, total=False):
    id: str
    section_id: str
    section_title: str
    text: str              # spoken version (translated / simplified)
    sentences: list[str]


class Section(TypedDict, total=False):
    id: str
    title: str
    text: str              # original text, retrieval side
    source_url: str


class TutorState(TypedDict, total=False):
    session_id: str

    # ---------- onboarding ----------
    onboarding_step: OnboardingStep
    source_kind: Literal["pdf", "topic"] | None
    pdf_paths: list[str]
    topic: str | None
    grade: str | None
    source_lang: str | None
    source_title: str | None
    source_url: str | None
    source_asks: int                 # how many times we've asked for topic/grade

    # ---------- material ----------
    sections: list[Section]
    lesson_plan: list[Beat]
    beat_index: int
    beat_spoken: bool                # current beat fully delivered?
    lesson_done: bool

    # ---------- FENCING -- the core invariant ----------
    turn_id: int                     # mirrors TurnClock at last handle_interrupt
    born_turn_id: int                # turn the current branch started under
    heard_cursor: Cursor | None      # frozen at interrupt
    heard_sentence: str | None       # sentence being spoken at interrupt
    pending_text: str                # last text we asked the voice to speak
    spoken_sentences: list[str]      # exactly what was last sent to the voice
    spoken_kind: str                 # lesson | answer | system
    spoken_offset: int               # prefix sentences (title/preface) before beat text
    spoken_start_sentence: int       # beat sentence index the spoken text started at
    preface: str | None              # one-off line before the first beat

    # ---------- current interrupt turn ----------
    event: Event | None
    user_utterance: str | None
    detected_lang: str | None
    intent: Intent | None
    command: Command | None
    command_arg: str | None          # e.g. target language for switch_lesson_lang
    session_cmd: SessionCmd | None
    nav_target: dict | None          # {"kind": prev|next|index|topic, "value": ...}
    queued_request: str | None       # second clause of "do A and B"
    recent_exchanges: list[dict]     # last N {"q":..., "a":...}
    retrieved: list[dict]            # [{"text", "section_id", "section_title", "score", "source"}]
    retrieval_score: float
    answer_mode: str | None          # notes | direct | web -- where the answer came from
    web_aborted: bool                # web search bailed because the turn went stale
    answer: str | None
    reply_lang: str | None           # one-off language for this reply only
    clarify_count: int
    stale_drops: int                 # fence_check discards, for evidence

    # ---------- session / delivery ----------
    paused: bool
    active_lang: str                 # BCP-47; lesson language AND Rime language
    speaker: str                     # Rime voice for active_lang
    speed_alpha: float

    transcript: Annotated[list[dict], operator.add]   # append-only audit log


def initial_state(session_id: str, pdf_paths: list[str] | None = None,
                  default_lang: str = "en", preset_lang: str | None = None) -> TutorState:
    """`preset_lang`: the student tapped a language button before the session
    started, so the spoken language question is skipped."""
    from config import LANG_SPEAKER, SPEED_ALPHA_DEFAULT, SUPPORTED_LANGS

    if preset_lang and preset_lang in SUPPORTED_LANGS:
        default_lang = preset_lang
    return TutorState(
        session_id=session_id,
        onboarding_step="source" if preset_lang in SUPPORTED_LANGS else "language",
        source_kind=None,
        pdf_paths=list(pdf_paths or []),
        topic=None, grade=None, source_lang=None, source_title=None,
        source_url=None, source_asks=0,
        sections=[], lesson_plan=[], beat_index=0, beat_spoken=False,
        lesson_done=False,
        turn_id=0, born_turn_id=0, heard_cursor=None, heard_sentence=None,
        pending_text="", spoken_sentences=[], spoken_kind="system",
        spoken_offset=0, spoken_start_sentence=0, preface=None,
        event=None, user_utterance=None, detected_lang=None, intent=None,
        command=None, command_arg=None, session_cmd=None, nav_target=None,
        queued_request=None,
        recent_exchanges=[], retrieved=[], retrieval_score=0.0, answer_mode=None, web_aborted=False,
        answer=None,
        reply_lang=None, clarify_count=0, stale_drops=0,
        paused=False,
        active_lang=default_lang,
        speaker=LANG_SPEAKER[default_lang],
        speed_alpha=SPEED_ALPHA_DEFAULT,
        transcript=[],
    )
