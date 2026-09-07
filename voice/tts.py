"""
Text -> speech, fenced.

RimeSpeaker implements agents.session.Speaker. Every speak() call:
  1. synthesises the line with Rime's one-shot HTTP endpoint (coda, PCM 16 kHz),
     serving fixed phrases (fillers, bridges, onboarding) from a disk cache so
     they cost nothing after the first run;
  2. re-checks the turn clock -- a barge-in during synthesis means the audio is
     never queued (this is the transport fence, layer 2 in ARCHITECTURE.md);
  3. hands the PCM to the Player tagged with its turn.

stop() flushes the player and remembers the heard cursor for the bridge.

If Rime fails (no key, network), the speaker still logs the text and calls
on_text, so the pipeline keeps working silently rather than crashing.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

import config
from agents.material import word_count
from voice.player import HeardCursor, PcmItem, Player, pcm_for_words

logger = logging.getLogger("v-tutor.tts")


class RimeHTTP:
    """One-shot synthesis. Returns raw int16 PCM at `sample_rate` (verified:
    Accept: audio/pcm on coda returns s16le at the requested samplingRate)."""

    def __init__(self, api_key: str | None = None, model_id: str | None = None,
                 sample_rate: int = 16000, timeout: float = 20.0,
                 cache_dir: str | Path | None = ".cache/tts") -> None:
        # None -> read the environment; "" -> deliberately disabled (tests).
        self.api_key = os.getenv("RIME_API_KEY") if api_key is None else (api_key or None)
        self.model_id = model_id or config.RIME_MODEL_ID
        self.sample_rate = sample_rate
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.cache_hits = 0
        self.last_ms = 0.0

    def _key(self, text: str, speaker: str, lang: str, speed: float) -> Path | None:
        if not self.cache_dir:
            return None
        h = hashlib.sha1(f"{self.model_id}|{speaker}|{lang}|{speed:.3f}|{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.pcm"

    def synth(self, text: str, *, speaker: str, lang: str, speed_alpha: float = 1.0) -> bytes | None:
        path = self._key(text, speaker, lang, speed_alpha)
        if path and path.exists():
            self.cache_hits += 1
            return path.read_bytes()
        if not self.api_key:
            return None
        import requests
        body = {"speaker": speaker, "text": text, "modelId": self.model_id,
                "lang": config.LANG_TO_CATALOG.get(lang, lang), "samplingRate": self.sample_rate,
                "speedAlpha": speed_alpha}
        t = time.perf_counter()
        try:
            r = requests.post(config.RIME_HTTP_ENDPOINT, json=body, timeout=self.timeout,
                              headers={"Authorization": f"Bearer {self.api_key}", "Accept": "audio/pcm",
                                       "Content-Type": "application/json"})
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.warning("rime synth failed (%s): %s", text[:40], exc)
            return None
        self.calls += 1
        self.last_ms = (time.perf_counter() - t) * 1000
        pcm = r.content
        if len(pcm) % 2:
            pcm = pcm[:-1]
        if path:
            try:
                path.write_bytes(pcm)
            except OSError:
                pass
        return pcm


class SilentSynth(RimeHTTP):
    """No network, no cache: every line becomes silence of a plausible length.
    For tests and dry runs."""

    def __init__(self) -> None:
        super().__init__(api_key="", cache_dir=None)

    def synth(self, text: str, **kw) -> bytes | None:  # noqa: ARG002
        return None


class RimeSpeaker:
    """agents.session.Speaker backed by Rime + a Player."""

    def __init__(self, player: Player, clock, synth: RimeHTTP | None = None,
                 on_text: Callable[[str, dict], None] | None = None,
                 on_event: Callable[[str, dict], None] | None = None,
                 silent_fallback: bool = True) -> None:
        self.player = player
        self.clock = clock
        self.synth = synth if synth is not None else RimeHTTP()
        self.on_text = on_text
        self.on_event = on_event
        self.silent_fallback = silent_fallback
        self.provider = "rime" if self.synth.api_key else "text-only"
        self.stops = 0
        self.last_cursor: HeardCursor | None = None
        self._lock = threading.Lock()

    def _emit(self, name: str, **payload) -> None:
        if self.on_event:
            self.on_event(name, payload)

    def speak(self, text: str, *, turn_id: int, lang: str, speaker: str, speed_alpha: float) -> None:
        text = (text or "").strip()
        if not text:
            return
        if self.on_text:
            self.on_text(text, {"turn": turn_id, "lang": lang, "speaker": speaker})
        t0 = time.perf_counter()
        pcm = self.synth.synth(text, speaker=speaker, lang=lang, speed_alpha=speed_alpha)
        synth_ms = (time.perf_counter() - t0) * 1000
        n_words = word_count(text)
        if pcm is None:
            if not self.silent_fallback:
                return
            # No audio available: queue silence of a plausible length so the
            # playback_confirmed cadence (and the demo) still works.
            pcm = pcm_for_words(n_words)
            self._emit("tts_fallback", turn=turn_id, text=text[:60])
        live = self.clock.current()
        if live != turn_id:
            # Learner spoke while Rime was rendering: this audio is stale.
            self._emit("tts_drop_stale", born=turn_id, live=live, text=text[:60])
            return
        self.player.enqueue(PcmItem(pcm=pcm, turn_id=turn_id, text=text, n_words=n_words))
        self._emit("tts", turn=turn_id, words=n_words, synth_ms=round(synth_ms),
                   audio_s=round(len(pcm) / (self.synth.sample_rate * 2), 2), cached=synth_ms < 5)

    def stop(self) -> None:
        """Barge-in: drop everything queued and remember where the learner was."""
        with self._lock:
            self.stops += 1
            self.last_cursor = self.player.flush()
        self._emit("stop", cursor=self.last_cursor.__dict__ if self.last_cursor else None)

    def take_cursor(self) -> HeardCursor | None:
        with self._lock:
            cur, self.last_cursor = self.last_cursor, None
            return cur
