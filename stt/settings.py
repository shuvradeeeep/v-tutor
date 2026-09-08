"""
settings.py
Every tunable knob for the STT pipeline lives here.

This is the single source for both entrypoints -- `stt/agent.py` (standalone
STT worker, CSV of transcripts) and `voice/audio_io.py` (the tutor's ear).
Before, each of them declared its own copy of the same env vars, so tuning one
silently left the other behind.

Each value can be overridden with an environment variable of the same name
(e.g. `set WHISPER_MODEL_SIZE=small`) so you can tune during a demo without
editing code.
"""

import os
from pathlib import Path


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _s(name: str, default: str) -> str:
    return os.getenv(name, default)


def _b(name: str, default: bool) -> bool:
    return _s(name, "1" if default else "0") not in ("0", "false", "False", "")


# ===========================================================================
# LIVE KNOBS -- read by stt/agent.py and voice/audio_io.py
# ===========================================================================

# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------
# "base" is the demo default: ~0.7s per utterance on 8 cores. It is genuinely
# weak at Hindi/Indic -- use "small" (~3x slower) when accuracy matters more.
WHISPER_MODEL_SIZE = _s("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = _s("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = _s("WHISPER_COMPUTE_TYPE", "int8")
# CTranslate2's own default (0) spreads over logical cores and loses ~25% to
# SMT contention. Physical cores is the fast setting on this class of CPU.
WHISPER_CPU_THREADS = _i("WHISPER_CPU_THREADS", max(1, (os.cpu_count() or 4) // 2))
# beam_size=1 (greedy) is faster than 5 and barely less accurate once the
# language is pinned by detection. Bump to 3-5 for accuracy over latency.
WHISPER_BEAM_SIZE = _i("WHISPER_BEAM_SIZE", 1)
# One encoder pass instead of two (see stt/transcriber.py). Set 0 to force the
# plain faster-whisper API -- ~2x slower, identical output.
WHISPER_SINGLE_PASS = _b("WHISPER_SINGLE_PASS", True)
# Languages the tutor can actually teach in. Spoken Hindi is acoustically
# almost identical to Urdu, and Whisper regularly labels it "ur" and writes the
# transcript in Arabic script -- which then goes to retrieval, to a web search,
# and comes back as a wall of Urdu text read out by an English voice. Anything
# detected outside this set is decoded again, pinned to the mapped language.
WHISPER_ALLOWED_LANGUAGES = set(
    _s("WHISPER_ALLOWED_LANGUAGES", "en,hi").replace(" ", "").split(","))
# What to re-decode a rejected language as. Urdu/Punjabi/Nepali/Sanskrit heard
# from a Hindi speaker are Hindi.
WHISPER_LANGUAGE_ALIASES = {
    "ur": "hi", "pa": "hi", "ne": "hi", "sa": "hi", "mr": "hi", "bh": "hi",
}
# Fallback when the detected language is not allowed and has no alias above.
WHISPER_LANGUAGE_FALLBACK = _s("WHISPER_LANGUAGE_FALLBACK", "en")
# Run a throwaway inference at startup so the first learner sentence doesn't
# pay for lazy model init.
WHISPER_WARMUP = _b("WHISPER_WARMUP", True)

# ---------------------------------------------------------------------------
# Which ear: local faster-whisper or Whisper large-v3 on Groq (stt/cloud.py)
# ---------------------------------------------------------------------------
# Local `base` int8 is fast (~0.4 s) but mishears the words that matter most:
# "photosynthesis" -> "Porto's", "Autos"; "English" -> "Fuck."; "class 6" ->
# "glass 6th". Whisper large-v3-turbo on Groq is ~0.3-0.6 s over the network
# and gets those right. The local model stays loaded as the fallback when the
# network call fails.
#   auto   groq when GROQ_API_KEY is set, else local     (default)
#   groq   always the cloud model (falls back to local on error)
#   local  never leave the machine
STT_PROVIDER = _s("STT_PROVIDER", "auto").strip().lower()
# large-v3 over the pruned "turbo" decoder: a live browser session on turbo heard
# "Hindi" as "Henry", "slowly" as "Sloane" and "yes" as "Yes, Lloyd". The full
# decoder costs ~150 ms more on Groq and is markedly better on short commands.
STT_CLOUD_MODEL = _s("STT_CLOUD_MODEL", "whisper-large-v3")
STT_CLOUD_TIMEOUT_S = _f("STT_CLOUD_TIMEOUT_S", 6.0)
# Whisper conditions on the prompt as if it were the preceding transcript, so
# it works best as sentences in the register the learner will use, containing
# the exact words the tutor must not mishear.
STT_PROMPT = _s("STT_PROMPT",
                "English. Hindi. Photosynthesis for class six. The heart for class 6. Slower, please. "
                "Speak slowly. Say that again. Pause. Continue. Go back. Skip this. Stop for today. "
                "Wait, what does that mean? How many chambers does the heart have? Explain that again.")
# Whisper invents these from silence and noise ("Thank you.", "Thanks for
# watching"). A transcript that is nothing but one of them is treated as
# nothing heard, up to this many seconds of audio.
HALLUCINATION_MAX_SEC = _f("HALLUCINATION_MAX_SEC", 3.0)

# ---------------------------------------------------------------------------
# Silero VAD / segmentation (stage 1: is anyone talking at all?)
# ---------------------------------------------------------------------------
# A door slam / cough is < 0.2s. Raise towards 0.25 to kill those.
VAD_MIN_SPEECH_DURATION = _f("VAD_MIN_SPEECH_DURATION", 0.1)
# Trailing silence before an utterance is considered finished. This is added
# to the transcript latency the learner feels, so it is the other half of the
# tuning story: 0.5s is responsive, 0.7s is safer for speakers who pause
# mid-sentence when code-switching.
VAD_MIN_SILENCE_DURATION = _f("VAD_MIN_SILENCE_DURATION", 0.5)
# Hard cap on one utterance. Whisper degrades badly past 30s anyway, and a
# single huge call is slower than several small ones.
VAD_MAX_UTTERANCE_DURATION = _f("VAD_MAX_UTTERANCE_DURATION", 30.0)

# ---------------------------------------------------------------------------
# Noise filter used by voice/audio_io.py
# ---------------------------------------------------------------------------
# Whisper's favourite hallucinations on near-silence. A *short* utterance whose
# entire transcript is one of these is treated as "nothing was said" (the
# bridge then handles it as a backchannel). Deliberately a small subset of
# HALLUCINATION_PHRASES below -- widening it would start dropping real one-word
# answers.
SHORT_NOISE_PHRASES = {"thank you.", "thanks.", "thank you", "you", "bye.", ".", "okay.", "hmm."}
SHORT_NOISE_MAX_SEC = _f("SHORT_NOISE_MAX_SEC", 1.0)

# ---------------------------------------------------------------------------
# Transcript log (stt/transcripts.py) -- written by BOTH entrypoints
# ---------------------------------------------------------------------------
# Absolute by default, so `python main.py local` from the repo root and
# `python agent.py dev` from inside stt/ append to the same file instead of
# leaving one transcripts.csv per working directory.
TRANSCRIPTS_CSV = _s("TRANSCRIPTS_CSV", str(Path(__file__).resolve().parent / "transcripts.csv"))
LOG_TRANSCRIPTS = _b("LOG_TRANSCRIPTS", True)


# ===========================================================================
# REFERENCE VALUES -- for the gating layer sketched in stt/audio.py.
# Nothing below is read by the running pipeline yet; voice/audio_io.py does its
# own minimal noise filtering. Kept here so the thresholds aren't re-derived
# from scratch when that layer lands.
# ===========================================================================

# ---------------------------------------------------------------------------
# Silero thresholds (not exposed by stt/vad.py's AudioSegmenter yet)
# ---------------------------------------------------------------------------
# activation_threshold is the single most important noise knob. Silero's 0.5
# default lets fans, keyboards and distant chatter through. 0.6-0.7 keeps
# close-mic speech and rejects most room noise.
VAD_ACTIVATION_THRESHOLD = _f("VAD_ACTIVATION_THRESHOLD", 0.65)
# Once speaking, we only exit below this. The hysteresis gap stops the VAD from
# chopping a sentence into fragments at every short pause.
VAD_DEACTIVATION_THRESHOLD = _f("VAD_DEACTIVATION_THRESHOLD", 0.35)
# Audio kept *before* the trigger, so the first syllable isn't clipped.
VAD_PREFIX_PADDING = _f("VAD_PREFIX_PADDING", 0.40)

# ---------------------------------------------------------------------------
# Utterance gate (stage 2: was that *your* voice, or the room?)
# ---------------------------------------------------------------------------
# Anything shorter than this is not a useful sentence.
MIN_SPEECH_SEC = _f("MIN_SPEECH_SEC", 0.35)
# Speech must be this many dB above the learned background noise floor.
# Raise to 10-12 in a noisy room; lower to 4 if your mic is far away.
MIN_SNR_DB = _f("MIN_SNR_DB", 6.0)
# Absolute loudness floor. Anything quieter is the room, not you.
MIN_SPEECH_DBFS = _f("MIN_SPEECH_DBFS", -45.0)
# Average Silero confidence across the utterance.
MIN_MEAN_SPEECH_PROB = _f("MIN_MEAN_SPEECH_PROB", 0.60)
# Fraction of 32ms windows that were confidently voiced.
MIN_VOICED_RATIO = _f("MIN_VOICED_RATIO", 0.35)

# Whisper's own quality gates. A segment failing any of these is dropped.
NO_SPEECH_THRESHOLD = _f("NO_SPEECH_THRESHOLD", 0.60)
LOG_PROB_THRESHOLD = _f("LOG_PROB_THRESHOLD", -1.0)
COMPRESSION_RATIO_THRESHOLD = _f("COMPRESSION_RATIO_THRESHOLD", 2.4)

# ---------------------------------------------------------------------------
# Language gate (stage 3: foreign language == noise)
# ---------------------------------------------------------------------------
# Whisper language codes for English + every Indian/South-Asian language the
# model actually supports. Odia, Maithili and Bhojpuri have no Whisper token,
# so Odia speech usually surfaces as "bn" or "hi".
INDIC_LANGUAGES = {
    "as",  # Assamese
    "bn",  # Bengali
    "gu",  # Gujarati
    "hi",  # Hindi
    "kn",  # Kannada
    "ml",  # Malayalam
    "mr",  # Marathi
    "ne",  # Nepali
    "pa",  # Punjabi
    "sa",  # Sanskrit
    "sd",  # Sindhi
    "si",  # Sinhala
    "ta",  # Tamil
    "te",  # Telugu
    "ur",  # Urdu
}
ALLOWED_LANGUAGES = {"en"} | INDIC_LANGUAGES

# If you know the session is Hindi+English only, set PIN_LANGUAGE=hi. Pinning
# is the strongest possible anti-hallucination measure.
PIN_LANGUAGE = _s("PIN_LANGUAGE", "").strip().lower() or None

# Whisper must put at least this much total probability mass on the allowed
# languages, otherwise we call the audio "foreign / not speech" and drop it.
MIN_ALLOWED_LANG_MASS = _f("MIN_ALLOWED_LANG_MASS", 0.50)

# ---------------------------------------------------------------------------
# Hallucination filters (stage 4: Whisper invented text from silence)
# ---------------------------------------------------------------------------
# Whole-transcript matches only. These are Whisper's well-known "trained on
# YouTube subtitles" outputs that appear when it is fed near-silence.
HALLUCINATION_PHRASES = {
    "",
    ".",
    "you",
    "bye",
    "bye bye",
    "okay",
    "ok",
    "hmm",
    "mm",
    "uh",
    "so",
    "yeah",
    "thank you",
    "thanks",
    "thank you very much",
    "thank you so much",
    "thank you for watching",
    "thanks for watching",
    "thanks for watching!",
    "please subscribe",
    "subscribe",
    "like and subscribe",
    "subscribe to my channel",
    "see you next time",
    "see you in the next video",
    "have a good time",
    "have a nice day",
    "the end",
    "music",
    "applause",
    "laughter",
    "silence",
    "amara.org",
    "subtitles by the amara.org community",
    "shukriya",
    "dhanyavaad",
    "namaste",
    "धन्यवाद",
    "शुक्रिया",
    "नमस्ते",
    "अपने चैनल को सब्सक्राइब करें",
}

# A transcript with fewer than this many characters is only kept if the
# utterance was reasonably long and clean.
MIN_TRANSCRIPT_CHARS = _i("MIN_TRANSCRIPT_CHARS", 2)
# Repetition loops ("dhu dhu dhu dhu") -- reject below this unique-word ratio.
MIN_UNIQUE_WORD_RATIO = _f("MIN_UNIQUE_WORD_RATIO", 0.34)
# ...or if the same word repeats this many times back to back.
MAX_WORD_RUN = _i("MAX_WORD_RUN", 4)
# Fraction of letters that must belong to a Latin/Indic/Arabic script.
MIN_SCRIPT_RATIO = _f("MIN_SCRIPT_RATIO", 0.80)

# ---------------------------------------------------------------------------
# Realtime behaviour
# ---------------------------------------------------------------------------
# Utterances waiting longer than this for a free Whisper slot are dropped
# rather than transcribed late.
MAX_QUEUE_AGE_SEC = _f("MAX_QUEUE_AGE_SEC", 12.0)
PENDING_QUEUE_SIZE = _i("PENDING_QUEUE_SIZE", 3)

# Log every rejected utterance to the CSV too, so you can tune the thresholds
# from real data instead of guessing.
LOG_REJECTED = _b("LOG_REJECTED", True)
