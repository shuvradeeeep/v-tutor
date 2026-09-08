"""
Single source of truth for every setting the README has to state exactly:
Rime model ID, speaker, language, endpoint, audio format, transport -- plus the
agent-layer knobs (retrieval threshold, stress-test delay, LLM roles).

Nothing else in the codebase hard-codes these. `scripts/preflight.py` validates
the Rime values against the live catalog before every demo run.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default


# ---------------------------------------------------------------------------
# Rime -- the primary and default speech provider
# ---------------------------------------------------------------------------
RIME_WS_ENDPOINT = "wss://users-ws.rime.ai/ws3"            # streaming, persistent
RIME_HTTP_ENDPOINT = "https://users.rime.ai/v1/rime-tts"    # one-shot (preflight, fillers)
RIME_CATALOG_URL = "https://users.rime.ai/data/voices/all-v2.json"

# Explicit on purpose: omitting modelId makes Rime default to mistv3.
RIME_MODEL_ID = _env("RIME_MODEL_ID", "coda")

# Without headphones the mic hears the tutor and the tutor answers itself.
# There is no AEC on the local path, so a transcript that repeats what was just
# spoken is discarded (see voice/bridge.py). Set ECHO_GUARD=0 if you are on
# headphones and want every word through.
ECHO_GUARD = (_env("ECHO_GUARD", "1") or "1").lower() not in ("0", "false", "no")
ECHO_GUARD_SEC = float(_env("ECHO_GUARD_SEC", "8"))
# Half duplex: ignore the mic entirely while the tutor is speaking. This is the
# only way a speaker-only session stays coherent -- otherwise the tutor's own
# voice keeps triggering the barge-in and the lesson stutters. It costs
# barge-in, so it is off for headphone users and switched on automatically
# after this many echoes are caught (0 disables the automatic switch).
HALF_DUPLEX = (_env("HALF_DUPLEX", "0") or "0").lower() in ("1", "true", "yes")
ECHO_AUTO_HALF_DUPLEX = int(_env("ECHO_AUTO_HALF_DUPLEX", "2"))

# Which speech provider voice/tts.py uses: rime | sapi | silent | auto.
# "auto" is Rime when RIME_API_KEY is set, otherwise the Windows synthesiser,
# so the full STT -> agent -> TTS loop is audible without a key.
TTS_PROVIDER = _env("TTS_PROVIDER", "auto")
RIME_AUDIO_FORMAT = "pcm"                                    # no MP3 frame lag on flush
RIME_SAMPLE_RATE = int(_env("RIME_SAMPLE_RATE", "16000"))    # verified by preflight
RIME_TRANSPORT = "websocket -> LiveKit WebRTC audio track"

# Study languages offered at onboarding (the whole lesson + the tutor's voice).
# Both exist on coda. BCP-47 for synthesis; the catalog uses 3-letter keys.
SUPPORTED_LANGS: tuple[str, ...] = ("en", "hi")
LANG_NAMES = {"en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German",
              "it": "Italian", "pt": "Portuguese", "ja": "Japanese"}

# One pinned coda voice per language Rime can speak. Study languages use these
# for the whole lesson; every other entry is for a one-off "explain that in X",
# after which the tutor returns to the study language and voice.
# Verified against the live catalog on 2026-09-07 (first listed voice per language,
# except en/hi which are deliberate picks). Arabic exists on coda but its voice
# names were not confirmed, so it is left out until preflight sees them.
LANG_SPEAKER = {
    "en": _env("RIME_SPEAKER_EN", "ana"),
    "hi": _env("RIME_SPEAKER_HI", "nadi"),
    "es": _env("RIME_SPEAKER_ES", "abril"),
    "fr": _env("RIME_SPEAKER_FR", "aurelie"),
    "de": _env("RIME_SPEAKER_DE", "greta"),
    "it": _env("RIME_SPEAKER_IT", "livia"),
    "pt": _env("RIME_SPEAKER_PT", "alzira"),
    "ja": _env("RIME_SPEAKER_JA", "akari"),
}
REPLY_LANGS: tuple[str, ...] = tuple(LANG_SPEAKER)      # valid for one-off explanations

# Catalog language codes differ from synthesis codes. Trap #2 in ARCHITECTURE.md.
LANG_TO_CATALOG = {
    "en": "eng", "hi": "hin", "es": "spa", "de": "ger", "fr": "fra",
    "ar": "ara", "ja": "jpn", "pt": "por", "it": "ita",
}

# ---------------------------------------------------------------------------
# Speed control ("slower" / "faster")
# ---------------------------------------------------------------------------
SPEED_ALPHA_DEFAULT = 1.0
SPEED_ALPHA_STEP = 0.15
SPEED_ALPHA_MIN = 0.6
SPEED_ALPHA_MAX = 1.5
# Trap #1 in ARCHITECTURE.md: the direction of speed_alpha is INVERTED between
# mist/mistv2 (lower = faster) and mistv3/arcana (lower = slower). Coda is
# undocumented. Phase 0 verifies this by ear and commits the clips; until then
# this flag is the single place to flip it.
SPEED_LOWER_IS_SLOWER = (_env("SPEED_LOWER_IS_SLOWER", "true") or "true").lower() == "true"

# ---------------------------------------------------------------------------
# Agent layer
# ---------------------------------------------------------------------------
RETRIEVAL_TAU = float(_env("RETRIEVAL_TAU", "0.35"))
RETRIEVAL_TOP_K = 4
CHUNK_SENTENCES = 3
CHUNK_OVERLAP = 1
BEAT_SENTENCES = 2
EMBEDDER = _env("EMBEDDER", "hash")                 # hash | fastembed
FASTEMBED_MODEL = "intfloat/multilingual-e5-small"

# PS full-duplex test: fixed delay injected into the web-search tool call.
STRESS_DELAY_MS = int(_env("STRESS_DELAY_MS", "0"))

LLM_FAST_PROVIDER = _env("LLM_FAST_PROVIDER", "stub")
LLM_FAST_MODEL = _env("LLM_FAST_MODEL")
LLM_STRONG_PROVIDER = _env("LLM_STRONG_PROVIDER", "stub")
LLM_STRONG_MODEL = _env("LLM_STRONG_MODEL")
# Reasoning models (gpt-oss on Groq) get "low" automatically; set to override,
# or "none" to send nothing. Reasoning tokens count against LLM_MAX_TOKENS.
LLM_REASONING_EFFORT = _env("LLM_REASONING_EFFORT")
# max_tokens is RESERVED against the provider's tokens-per-minute limit, not
# just billed on use: Groq counted a 195-token prompt with max_tokens=1200 as
# ~1198 tokens of quota. Measured on openai/gpt-oss-120b at reasoning "low":
# an answer completes in ~50 tokens, a section rewrite in ~220. So these are
# sized to the work, with headroom, instead of to the model's ceiling.
LLM_MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "400"))            # answers, intent, explain
LLM_MAX_TOKENS_LONG = int(_env("LLM_MAX_TOKENS_LONG", "800"))  # translate/simplify a whole section
LLM_TIMEOUT_S = float(_env("LLM_TIMEOUT_S", "20"))
# Tokens per minute the provider allows per model (Groq free tier: 8000).
# Requests are paced to stay under it instead of failing: a 429 silently
# degrades the tutor to its deterministic fallback, which sounds like the model
# got worse. 0 disables pacing.
LLM_TPM_LIMIT = int(_env("LLM_TPM_LIMIT", "8000"))
# How long a call may wait for quota. Background section prep can afford to
# wait; the answer path cannot, so it uses a fraction of this.
LLM_TPM_MAX_WAIT = float(_env("LLM_TPM_MAX_WAIT", "20"))
WEB_SEARCH_PROVIDER = _env("WEB_SEARCH_PROVIDER", "stub")

GAP_FILLER_DEADLINE_MS = 700
# Question/answer pairs carried into every answer prompt. Two was enough to
# resolve "and why?" but the tutor forgot anything said earlier in the lesson
# and re-explained things it had just explained. These are short spoken turns,
# so six costs little and makes the session feel continuous.
RECENT_EXCHANGES_KEEP = int(_env("RECENT_EXCHANGES_KEEP", "6"))
# Questions (without answers) kept for the whole session. Twelve of these cost
# a line or two of prompt but let an hour-long lesson refer back to something
# asked at the start, which the six-exchange window has long forgotten.
ASKED_QUESTIONS_KEEP = int(_env("ASKED_QUESTIONS_KEEP", "12"))
CLARIFY_MAX_ASKS = 1

# Lesson size for a class-level session: Wikipedia articles run to 80+ beats.
MAX_SECTIONS = int(_env("MAX_SECTIONS", "8"))
MAX_BEATS = int(_env("MAX_BEATS", "30"))
# Sections localised before the first beat is spoken; the rest are prepared in
# the background. Each one costs a strong-model call (~1.5-2 s) before the tutor
# can start, so 1 keeps time-to-first-beat lowest.
PREPARE_UPFRONT_SECTIONS = int(_env("PREPARE_UPFRONT_SECTIONS", "1"))
LOCALIZE_RETRIES = 1                  # re-ask once if the model returns the wrong line count

CHECKPOINT_DB = _env("CHECKPOINT_DB", "sessions.db")
EVIDENCE_DIR = _env("EVIDENCE_DIR", "evidence")


def rime_summary() -> dict:
    """The exact block the PS asks the README to state. Printed by preflight."""
    return {
        "model_id": RIME_MODEL_ID,
        "speakers": dict(LANG_SPEAKER),
        "languages": list(SUPPORTED_LANGS),
        "endpoint": RIME_WS_ENDPOINT,
        "audio_format": RIME_AUDIO_FORMAT,
        "sample_rate": RIME_SAMPLE_RATE,
        "transport": RIME_TRANSPORT,
    }
