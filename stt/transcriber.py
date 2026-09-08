"""
transcriber.py
Audio format conversion + Whisper inference. Shared by both entrypoints:
`stt/agent.py` (standalone STT worker) and `voice/audio_io.py` (the tutor).

Latency notes -- everything here exists to keep "user stopped talking ->
transcript" short, because on CPU that number is ~100% Whisper:

  1. Single encoder pass. `WhisperModel.transcribe(language=None)` runs the
     encoder TWICE on a short utterance: once inside detect_language(), then
     again in generate_segments(), which throws the first result away. For a
     2s utterance on `base`/int8 the encoder is ~60% of the whole call, so
     that duplicate pass is roughly half the latency. `_single_encode()`
     encodes once, reads the language off that output, and hands the same
     encoder output to generate_segments(). Language detection is unchanged --
     same model, same features, same probabilities.
  2. cpu_threads pinned to physical cores (see settings.WHISPER_CPU_THREADS).
     CTranslate2's default oversubscribes SMT siblings and is ~25% slower.
  3. Greedy decoding, no timestamp tokens, no cross-window prompt.

`_single_encode()` reaches into faster-whisper internals, so it is version
sensitive. Every failure falls back to the plain public API permanently, which
is only slower -- never wrong. Set WHISPER_SINGLE_PASS=0 to disable it.
"""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import fields as dataclass_fields

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import TranscriptionOptions, get_suppressed_tokens
from livekit import rtc

try:  # imported as `stt.transcriber` (tutor) or as `transcriber` (stt/agent.py)
    from stt.audio import frames_to_float32
except ImportError:  # pragma: no cover
    from audio import frames_to_float32

logger = logging.getLogger("stt-orchestrator")

# Decoder settings that cost accuracy nowhere but save tokens: we never read
# timestamps, and each utterance is transcribed standalone, so there is no
# previous text worth conditioning on.
_DECODE_OVERRIDES = {
    "without_timestamps": True,
    "condition_on_previous_text": False,
}


class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        cpu_threads: int = 0,
        single_pass: bool = True,
        allowed_languages: set[str] | None = None,
        language_aliases: dict[str, str] | None = None,
        language_fallback: str = "en",
    ):
        """
        Loads the faster-whisper model into memory once upon startup.

        beam_size: decoding width. 1 (greedy) is the default; with the language
        already pinned by detection it is a few percent faster than 5 and
        barely less accurate. Raise it if you care about accuracy over latency.
        cpu_threads: 0 lets CTranslate2 pick, which oversubscribes on SMT CPUs.
        Pass the physical core count (settings.WHISPER_CPU_THREADS does).
        single_pass: use the one-encoder-pass fast path (see module docstring).
        """
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        self.beam_size = beam_size
        self.single_pass = single_pass
        # Empty set = accept whatever Whisper detects.
        self.allowed_languages = set(allowed_languages or ())
        self.language_aliases = dict(language_aliases or {})
        self.language_fallback = language_fallback
        self._tokenizers: dict[str, Tokenizer] = {}
        self._options: dict[str, TranscriptionOptions] = {}

    # ------------------------------------------------------------------ audio
    def frames_to_float32(self, frames: list[rtc.AudioFrame]) -> np.ndarray:
        """LiveKit PCM frames -> 16kHz mono float32 in [-1.0, 1.0]."""
        return frames_to_float32(frames)

    # ------------------------------------------------------------- fast path
    def _build_options(self, tokenizer: Tokenizer) -> TranscriptionOptions:
        """
        Reproduce the TranscriptionOptions that WhisperModel.transcribe() would
        have built, with our overrides applied.

        Defaults are read off transcribe()'s own signature rather than hardcoded,
        so a faster-whisper upgrade that changes a default is picked up instead
        of silently diverging. A field with no matching parameter raises, which
        sends the caller back to the public API.
        """
        defaults = {
            name: p.default
            for name, p in inspect.signature(WhisperModel.transcribe).parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        overrides = {"beam_size": self.beam_size, **_DECODE_OVERRIDES}

        values = {}
        for field in dataclass_fields(TranscriptionOptions):
            name = field.name
            if name in overrides:
                values[name] = overrides[name]
            elif name == "temperatures":  # the one field that isn't 1:1 named
                temp = defaults["temperature"]
                values[name] = list(temp) if isinstance(temp, (list, tuple)) else [temp]
            elif name == "suppress_tokens":
                tokens = defaults.get("suppress_tokens")
                values[name] = get_suppressed_tokens(tokenizer, tokens) if tokens else tokens
            elif name in defaults:
                values[name] = defaults[name]
            else:
                raise RuntimeError(f"no default for TranscriptionOptions.{name}")
        return TranscriptionOptions(**values)

    def _for_language(self, language: str) -> tuple[Tokenizer, TranscriptionOptions]:
        if language not in self._tokenizers:
            tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task="transcribe",
                language=language,
            )
            self._tokenizers[language] = tokenizer
            self._options[language] = self._build_options(tokenizer)
        return self._tokenizers[language], self._options[language]

    def _single_encode(self, audio: np.ndarray) -> tuple[str, str, float]:
        """Encode once, detect the language from that output, then decode it."""
        model = self.model
        nb_max_frames = model.feature_extractor.nb_max_frames

        features = model.feature_extractor(audio)
        # Exactly the window WhisperModel.detect_language() and the first
        # iteration of generate_segments() would each have encoded.
        encoder_output = model.encode(pad_or_trim(features[..., :nb_max_frames]))

        token, probability = model.model.detect_language(encoder_output)[0][0]
        language = token[2:-2]  # "<|hi|>" -> "hi"
        language = self._allowed(language)

        tokenizer, options = self._for_language(language)
        segments = model.generate_segments(features, tokenizer, options, False, encoder_output)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text, language, probability

    def _allowed(self, language: str) -> str:
        """
        Map a detected language the tutor cannot teach onto one it can.

        Spoken Hindi is regularly detected as Urdu, and the transcript then
        comes back in Arabic script -- unusable for retrieval, for the lesson,
        and for an English or Hindi voice. Because the fix happens before
        decoding, and the encoder output is reused, correcting it is free: the
        same audio is simply decoded with the right language token.
        """
        if not self.allowed_languages or language in self.allowed_languages:
            return language
        mapped = self.language_aliases.get(language, self.language_fallback)
        if mapped not in self.allowed_languages:
            mapped = next(iter(self.allowed_languages))
        logger.info("whisper detected %r, decoding as %r", language, mapped)
        return mapped

    def _public_api(self, audio: np.ndarray) -> tuple[str, str, float]:
        segments, info = self.model.transcribe(
            audio,
            language=None,
            beam_size=self.beam_size,
            **_DECODE_OVERRIDES,
        )
        language = self._allowed(info.language)
        if language != info.language:
            # Costs a whole second pass here; the single-encode path corrects
            # the language before decoding and pays nothing.
            segments, info = self.model.transcribe(
                audio, language=language, beam_size=self.beam_size, **_DECODE_OVERRIDES)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text, language, info.language_probability

    # ------------------------------------------------------------- inference
    def transcribe_array(self, audio: np.ndarray) -> tuple[str, str, float, float]:
        """
        Same contract as transcribe(), for audio that is already 16kHz mono
        float32 (benchmarks, tests, wav fixtures).
        """
        if audio.size == 0:
            return "", "", 0.0, 0.0

        t_start = time.perf_counter()
        if self.single_pass:
            try:
                text, language, probability = self._single_encode(audio)
            except Exception:
                # Internals moved (faster-whisper upgrade). Say so once, then
                # stay on the public API for the rest of the process.
                self.single_pass = False
                logger.exception("single-encode path failed; falling back to WhisperModel.transcribe")
                text, language, probability = self._public_api(audio)
        else:
            text, language, probability = self._public_api(audio)
        duration_ms = (time.perf_counter() - t_start) * 1000

        return text, language, probability, duration_ms

    def transcribe(self, frames: list[rtc.AudioFrame]) -> tuple[str, str, float, float]:
        """
        Runs inference on captured audio frames.

        This is a blocking, CPU-bound call. Callers (stt/agent.py,
        voice/audio_io.py) run it via asyncio.to_thread rather than awaiting it
        directly, so it doesn't stall the event loop -- and therefore doesn't
        stall VAD/audio processing or the barge-in fast path -- while it runs.

        Returns: (transcribed_text, detected_language, language_prob, inference_duration_ms)
        """
        return self.transcribe_array(self.frames_to_float32(frames))

    def warmup(self, seconds: float = 1.0) -> float:
        """
        Run one throwaway inference so the first real utterance doesn't pay for
        lazy weight loading and allocator warmup (~1s extra, otherwise landing
        on the learner's very first sentence). Returns the ms it took.
        """
        t = time.perf_counter()
        self.transcribe_array(np.zeros(int(16000 * seconds), dtype=np.float32))
        elapsed = (time.perf_counter() - t) * 1000
        logger.info("whisper warmup: %.0f ms (single_pass=%s)", elapsed, self.single_pass)
        return elapsed
