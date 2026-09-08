"""
Whisper large-v3 on Groq as the tutor's ear, with the local model as fallback.

Same contract as stt.transcriber.WhisperTranscriber.transcribe(frames):
    (text, language, language_probability, inference_ms)

Why: the local `base` model runs in 0.4 s but mishears the words a lesson hangs
on ("photosynthesis" -> "Porto's", "class 6" -> "glass 6th"). large-v3-turbo
over Groq returns in 0.3-0.6 s on a normal connection and gets them right. The
network is the new failure mode, so every call that raises or times out is
answered by the local model instead, and the switch is logged once.

The audio leaves the machine only in this backend. STT_PROVIDER=local keeps it
on the laptop.
"""
from __future__ import annotations

import io
import logging
import os
import time
import wave

import numpy as np

from stt import settings
from stt.audio import frames_to_float32

logger = logging.getLogger("v-tutor.stt.cloud")

# verbose_json returns the language as a word, not a code.
_LANG_NAMES = {
    "english": "en", "hindi": "hi", "urdu": "ur", "punjabi": "pa", "nepali": "ne", "marathi": "mr",
    "sanskrit": "sa", "bengali": "bn", "tamil": "ta", "telugu": "te", "gujarati": "gu", "kannada": "kn",
    "malayalam": "ml", "spanish": "es", "french": "fr", "german": "de", "italian": "it",
    "portuguese": "pt", "japanese": "ja", "chinese": "zh", "arabic": "ar", "russian": "ru",
}


def _lang_code(name: str | None) -> str:
    if not name:
        return ""
    n = name.strip().lower()
    return _LANG_NAMES.get(n, n if len(n) == 2 else "")


def _wav_bytes(audio: np.ndarray, rate: int = 16000) -> bytes:
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class GroqTranscriber:
    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float | None = None,
                 prompt: str | None = None, allowed_languages: set[str] | None = None,
                 language_aliases: dict[str, str] | None = None, language_fallback: str = "en",
                 fallback=None) -> None:
        from openai import OpenAI
        self.model = model or settings.STT_CLOUD_MODEL
        self.prompt = settings.STT_PROMPT if prompt is None else prompt
        self.allowed = set(allowed_languages if allowed_languages is not None else settings.WHISPER_ALLOWED_LANGUAGES)
        self.aliases = dict(language_aliases if language_aliases is not None else settings.WHISPER_LANGUAGE_ALIASES)
        self.fallback_lang = language_fallback
        self._client = OpenAI(api_key=api_key or os.getenv("GROQ_API_KEY"),
                              base_url="https://api.groq.com/openai/v1",
                              timeout=timeout or settings.STT_CLOUD_TIMEOUT_S, max_retries=0)
        self._fallback = fallback          # a WhisperTranscriber, or a callable returning one
        self.calls = 0
        self.failures = 0
        self.fallbacks = 0
        self.last_ms = 0.0

    # ------------------------------------------------------------------ api
    def _request(self, wav: bytes, language: str | None) -> tuple[str, str]:
        kw = dict(file=("utterance.wav", wav, "audio/wav"), model=self.model, response_format="verbose_json",
                  temperature=0.0)
        if self.prompt:
            kw["prompt"] = self.prompt
        if language:
            kw["language"] = language
        r = self._client.audio.transcriptions.create(**kw)
        text = (getattr(r, "text", None) or "").strip()
        lang = _lang_code(getattr(r, "language", None)) or (language or "")
        return text, lang

    def _allowed_lang(self, lang: str) -> str:
        if not self.allowed or not lang or lang in self.allowed:
            return lang
        mapped = self.aliases.get(lang, self.fallback_lang)
        return mapped if mapped in self.allowed else next(iter(self.allowed))

    def _local(self):
        if callable(self._fallback) and not hasattr(self._fallback, "transcribe_array"):
            self._fallback = self._fallback()
        return self._fallback

    # ------------------------------------------------------------ inference
    def transcribe_array(self, audio: np.ndarray) -> tuple[str, str, float, float]:
        if audio.size == 0:
            return "", "", 0.0, 0.0
        t0 = time.perf_counter()
        try:
            text, lang = self._request(_wav_bytes(audio), None)
            self.calls += 1
            fixed = self._allowed_lang(lang)
            if fixed != lang:
                # Detected a language the tutor cannot teach (Hindi heard as
                # Urdu). Decode again pinned to the alias; one more round trip,
                # and only on the rare utterance where it happens.
                logger.info("cloud whisper detected %r, decoding as %r", lang, fixed)
                text, _ = self._request(_wav_bytes(audio), fixed)
                self.calls += 1
                lang = fixed
            self.last_ms = (time.perf_counter() - t0) * 1000
            # verbose_json has no language probability; 1.0 means "not measured".
            return text, lang, 1.0, self.last_ms
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            local = self._local()
            if local is None:
                raise
            self.fallbacks += 1
            if self.fallbacks == 1:
                logger.warning("cloud STT failed (%s); using local whisper for this utterance", exc)
            else:
                logger.info("cloud STT failed (%s); local whisper", exc)
            return local.transcribe_array(audio)

    def transcribe(self, frames) -> tuple[str, str, float, float]:
        return self.transcribe_array(frames_to_float32(frames))

    def warmup(self, seconds: float = 0.5) -> float:
        """One tiny request so the TLS handshake and connection pool are paid
        for before the learner's first sentence. Failure here is not fatal."""
        t = time.perf_counter()
        try:
            self._request(_wav_bytes(np.zeros(int(16000 * seconds), dtype=np.float32)), "en")
        except Exception as exc:  # noqa: BLE001
            logger.warning("cloud STT warmup failed: %s (will fall back to local whisper if it keeps failing)", exc)
        ms = (time.perf_counter() - t) * 1000
        logger.info("cloud whisper %s warm in %.0f ms", self.model, ms)
        return ms


def make_local():
    from stt.transcriber import WhisperTranscriber
    t = time.perf_counter()
    m = WhisperTranscriber(
        model_size=settings.WHISPER_MODEL_SIZE, device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE, beam_size=settings.WHISPER_BEAM_SIZE,
        cpu_threads=settings.WHISPER_CPU_THREADS, single_pass=settings.WHISPER_SINGLE_PASS,
        allowed_languages=settings.WHISPER_ALLOWED_LANGUAGES,
        language_aliases=settings.WHISPER_LANGUAGE_ALIASES,
        language_fallback=settings.WHISPER_LANGUAGE_FALLBACK)
    logger.info("whisper %s loaded in %.1fs", settings.WHISPER_MODEL_SIZE, time.perf_counter() - t)
    return m


def make_transcriber():
    """The ear both entrypoints use, chosen by STT_PROVIDER."""
    choice = settings.STT_PROVIDER
    if choice == "auto":
        choice = "groq" if os.getenv("GROQ_API_KEY") else "local"
    if choice == "groq":
        if not os.getenv("GROQ_API_KEY"):
            logger.warning("STT_PROVIDER=groq but GROQ_API_KEY is unset; using local whisper")
            return make_local()
        # The local model is the fallback. Load it now, not on the first
        # network error, so a failure mid-lesson costs 0.4 s and not 10.
        local = make_local()
        if settings.WHISPER_WARMUP:
            local.warmup()
        return GroqTranscriber(fallback=local)
    return make_local()


def describe(transcriber) -> str:
    if isinstance(transcriber, GroqTranscriber):
        return f"groq/{transcriber.model} (fallback local {settings.WHISPER_MODEL_SIZE})"
    return f"local {settings.WHISPER_MODEL_SIZE} (beam {settings.WHISPER_BEAM_SIZE}, {settings.WHISPER_CPU_THREADS} threads)"
