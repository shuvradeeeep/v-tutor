"""
Text -> speech, fenced.

RimeSpeaker implements agents.session.Speaker. Every speak() call:
  1. synthesises the line with Rime coda -- over a persistent websocket
     (RimeWS: first audio in ~0.4 s, word timestamps, server-side `clear` on
     barge-in) or the one-shot HTTP endpoint (RimeHTTP, ~2.7 s, no timestamps)
     -- serving fixed phrases (fillers, bridges, onboarding) from a disk cache
     so they cost nothing after the first run;
  2. re-checks the turn clock -- a barge-in during synthesis means the audio is
     never queued (this is the transport fence, layer 2 in ARCHITECTURE.md);
  3. hands the PCM to the Player tagged with its turn, with the word timestamps
     when Rime gave them, so the heard cursor is exact rather than estimated.

stop() flushes the player, tells Rime to drop anything still queued for the
old turn, and remembers the heard cursor for the bridge.

If Rime fails (no key, network), the speaker still logs the text and calls
on_text, so the pipeline keeps working silently rather than crashing.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
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


def time_scale_factor(pace: float) -> float:
    """The graph's pace (1.0 normal, lower = slower) -> coda's `timeScaleFactor`
    (1.0 normal, HIGHER = slower, clamped to 0.4-2.5). Direction measured, not
    assumed: scripts/speed_check.py, evidence/speed/."""
    pace = max(0.05, float(pace or 1.0))
    return round(min(config.TIME_SCALE_MAX, max(config.TIME_SCALE_MIN, 1.0 / pace)), 3)


class RimeHTTP:
    """One-shot synthesis. Returns raw int16 PCM at `sample_rate` (verified:
    Accept: audio/pcm on coda returns s16le at the requested samplingRate)."""

    provider = "rime"

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
        self.last_cached = False      # whether the most recent synth() came from disk
        self.last_word_ends: list[float] | None = None    # seconds, per word, when Rime gave them
        self.last_first_audio_ms: float | None = None     # websocket only: time to first chunk

    def _key(self, text: str, speaker: str, lang: str, speed: float) -> Path | None:
        if not self.cache_dir:
            return None
        h = hashlib.sha1(f"{self.model_id}|{speaker}|{lang}|{speed:.3f}|{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.pcm"

    def synth(self, text: str, *, speaker: str, lang: str, speed_alpha: float = 1.0) -> bytes | None:
        path = self._key(text, speaker, lang, speed_alpha)
        self.last_word_ends = None
        self.last_first_audio_ms = None
        if path and path.exists():
            self.cache_hits += 1
            self.last_cached = True
            side = path.with_suffix(".json")
            if side.exists():
                try:
                    self.last_word_ends = json.loads(side.read_text(encoding="utf-8")).get("end")
                except (OSError, ValueError):
                    pass
            return path.read_bytes()
        self.last_cached = False
        pcm = self._render(text, speaker=speaker, lang=lang, speed_alpha=speed_alpha)
        if pcm is None:
            return None
        if len(pcm) % 2:
            pcm = pcm[:-1]
        if path:
            try:
                path.write_bytes(pcm)
                if self.last_word_ends:
                    path.with_suffix(".json").write_text(json.dumps({"end": self.last_word_ends}), encoding="utf-8")
            except OSError:
                pass
        return pcm

    def cache_lookup(self, text: str, *, speaker: str, lang: str, speed_alpha: float = 1.0
                     ) -> tuple[bytes, list[float] | None] | None:
        """(pcm, word_ends) from disk, or None. Counts as a hit like synth()."""
        path = self._key(text, speaker, lang, speed_alpha)
        if not (path and path.exists()):
            return None
        self.cache_hits += 1
        self.last_cached = True
        self.last_first_audio_ms = None
        ends = None
        side = path.with_suffix(".json")
        if side.exists():
            try:
                ends = json.loads(side.read_text(encoding="utf-8")).get("end")
            except (OSError, ValueError):
                pass
        self.last_word_ends = ends
        return path.read_bytes(), ends

    def cache_store(self, text: str, *, speaker: str, lang: str, speed_alpha: float,
                    pcm: bytes, word_ends: list[float] | None) -> None:
        path = self._key(text, speaker, lang, speed_alpha)
        if not path:
            return
        try:
            path.write_bytes(pcm)
            if word_ends:
                path.with_suffix(".json").write_text(json.dumps({"end": word_ends}), encoding="utf-8")
        except OSError:
            pass

    def cancel(self) -> None:
        """Barge-in: nothing to cancel on the one-shot path."""

    def _render(self, text: str, *, speaker: str, lang: str, speed_alpha: float) -> bytes | None:
        """The actual synthesis call. Subclasses swap this; caching is shared."""
        if not self.api_key:
            return None
        import requests
        # Per the coda HTTP reference: BCP-47 `lang`, `timeScaleFactor` for pace.
        body = {"speaker": speaker, "text": text, "modelId": self.model_id, "lang": lang,
                "samplingRate": self.sample_rate, "timeScaleFactor": time_scale_factor(speed_alpha)}
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
        return r.content


class RimeWS(RimeHTTP):
    """
    Coda over Rime's persistent websocket (config.RIME_WS_ENDPOINT).

    Why this exists: measured on 2026-09-09, the same sentence took ~2.7 s to
    come back over HTTP and 0.39 s to first audio over the socket, and only the
    socket returns `timestamps` (per-word start/end), which is what makes the
    heard cursor exact. It also accepts `{"operation": "clear"}`, the
    server-side half of the barge-in kill path.

    One socket per (speaker, lang, timeScaleFactor) -- those are connection
    query parameters -- reused across lines. Requests are sequential (the
    tutor speaks one line at a time), each tagged with a contextId so chunks
    left over from a cleared context are recognised and dropped. Any socket
    error falls back to the HTTP request for that line and reconnects on the
    next one, so the socket can only make things faster, never silent.
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="rime-ws")
        self._thread.start()
        self._conns: dict[tuple, object] = {}
        self._cancel: asyncio.Event | None = None
        # A barge-in can land before the request coroutine has even started
        # (the first connect takes ~1.5 s). The flag survives that gap; the
        # Event is only for waking a request that is already waiting.
        self._cancel_requested = False
        self._ctx = 0
        self.ws_calls = 0
        self.ws_failures = 0
        self.http_fallbacks = 0
        self.cancelled = 0

    # ------------------------------------------------------------ connection
    def _url(self, speaker: str, lang: str, tsf: float) -> str:
        return (f"{config.RIME_WS_ENDPOINT}?speaker={speaker}&modelId={self.model_id}&audioFormat=pcm"
                f"&samplingRate={self.sample_rate}&lang={lang}&segment=immediate&timeScaleFactor={tsf}")

    async def _connect(self, key: tuple) -> object:
        import websockets
        ws = self._conns.get(key)
        if ws is not None:
            return ws
        speaker, lang, tsf = key
        ws = await asyncio.wait_for(
            websockets.connect(self._url(speaker, lang, tsf), max_size=None,
                               additional_headers={"Authorization": f"Bearer {self.api_key}"}),
            self.timeout)
        self._conns[key] = ws
        return ws

    async def _drop(self, key: tuple) -> None:
        ws = self._conns.pop(key, None)
        if ws is not None:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    # --------------------------------------------------------------- request
    async def _request(self, key: tuple, text: str, sink=None, on_first: Callable[[], bool] | None = None
                       ) -> tuple[bytes, list[float] | None, float | None] | None:
        """One line over the socket. With `sink` (a PcmItem) every chunk is
        appended as it lands and `on_first` runs on the first one -- returning
        False from it aborts (the turn moved on). Returns None if cancelled."""
        cancel = asyncio.Event()
        self._cancel = cancel
        if self._cancel_requested:
            cancel.set()
        ws = await self._connect(key)
        if cancel.is_set():
            return None                      # the learner spoke while we were connecting
        self._ctx += 1
        ctx = f"c{self._ctx}"
        t0 = time.perf_counter()
        await ws.send(json.dumps({"text": text, "contextId": ctx}))
        await ws.send(json.dumps({"operation": "flush"}))
        pcm = bytearray()
        ends: list[float] | None = None
        first: float | None = None
        try:
            while True:
                recv = asyncio.ensure_future(ws.recv())
                stop = asyncio.ensure_future(cancel.wait())
                done, _ = await asyncio.wait({recv, stop}, timeout=self.timeout,
                                             return_when=asyncio.FIRST_COMPLETED)
                if stop in done:
                    recv.cancel()
                    # Layer 1 of the kill path: whatever Rime still has queued
                    # for this context is dropped server-side.
                    await ws.send(json.dumps({"operation": "clear"}))
                    return None
                stop.cancel()
                if recv not in done:
                    raise TimeoutError(f"no message from Rime in {self.timeout}s")
                m = json.loads(recv.result())
                if m.get("contextId") not in (None, ctx):
                    continue                         # a cleared context's leftovers
                typ = m.get("type")
                if typ == "chunk":
                    data = base64.b64decode(m["data"])
                    if first is None:
                        first = (time.perf_counter() - t0) * 1000
                        if on_first is not None and not on_first():
                            await ws.send(json.dumps({"operation": "clear"}))
                            return None
                    pcm += data
                    if sink is not None:
                        sink.append(data)
                elif typ == "timestamps":
                    ends = [float(x) for x in (m.get("word_timestamps") or {}).get("end") or []] or None
                elif typ == "done":
                    return bytes(pcm), ends, first
                elif typ == "error":
                    raise RuntimeError(m.get("message") or "rime error")
        finally:
            if self._cancel is cancel:
                self._cancel = None

    def stream(self, text: str, *, speaker: str, lang: str, speed_alpha: float, item,
               on_first: Callable[[], bool]) -> str:
        """Streaming synthesis into `item` (a PcmItem enqueued by `on_first`).
        Returns "done" (audio complete, cached), "http" (socket failed before
        any audio; the line came back whole over HTTP), "truncated" (socket
        died mid-line; what arrived is what plays), "cancelled" (barge-in or
        the turn had moved on), or "failed" (no audio from anywhere)."""
        if not self.api_key:
            return "failed"
        key = (speaker, lang, time_scale_factor(speed_alpha))
        t = time.perf_counter()
        self.last_first_audio_ms = None
        self.last_word_ends = None
        self.last_cached = False
        self._cancel_requested = False
        for attempt in (1, 2):
            fut = asyncio.run_coroutine_threadsafe(self._request(key, text, item, on_first), self._loop)
            try:
                res = fut.result(timeout=self.timeout + 5)
            except Exception as exc:  # noqa: BLE001
                self.ws_failures += 1
                asyncio.run_coroutine_threadsafe(self._drop(key), self._loop).result(timeout=5)
                if len(item.pcm):
                    # Audio is already playing; a second copy over HTTP would
                    # repeat the line. Close the item where it stopped.
                    logger.warning("rime ws died mid-line (%s); playing what arrived", exc)
                    item.finish(None)
                    return "truncated"
                if attempt == 1:
                    logger.info("rime ws failed (%s), reconnecting once", exc)
                    continue
                self.http_fallbacks += 1
                logger.warning("rime ws failed twice (%s); HTTP for this line", exc)
                pcm = super()._render(text, speaker=speaker, lang=lang, speed_alpha=speed_alpha)
                if not pcm:
                    return "failed"
                if not on_first():
                    return "cancelled"
                item.append(pcm)
                item.finish(None)
                self.cache_store(text, speaker=speaker, lang=lang, speed_alpha=speed_alpha, pcm=pcm, word_ends=None)
                return "http"
            if res is None:
                self.cancelled += 1
                item.finish(None)                    # never stall a player that holds it
                return "cancelled"
            pcm, ends, first = res
            if not pcm:
                item.finish(None)
                return "failed"
            item.finish(ends)
            self.calls += 1
            self.ws_calls += 1
            self.last_ms = (time.perf_counter() - t) * 1000
            self.last_first_audio_ms = first
            self.last_word_ends = item.word_ends
            self.cache_store(text, speaker=speaker, lang=lang, speed_alpha=speed_alpha, pcm=pcm, word_ends=ends)
            return "done"
        return "failed"

    def _render(self, text: str, *, speaker: str, lang: str, speed_alpha: float) -> bytes | None:
        if not self.api_key:
            return None
        key = (speaker, lang, time_scale_factor(speed_alpha))
        t = time.perf_counter()
        self._cancel_requested = False
        for attempt in (1, 2):
            fut = asyncio.run_coroutine_threadsafe(self._request(key, text), self._loop)
            try:
                res = fut.result(timeout=self.timeout + 5)
            except Exception as exc:  # noqa: BLE001
                self.ws_failures += 1
                asyncio.run_coroutine_threadsafe(self._drop(key), self._loop).result(timeout=5)
                if attempt == 1:
                    logger.info("rime ws failed (%s), reconnecting once", exc)
                    continue
                self.http_fallbacks += 1
                logger.warning("rime ws failed twice (%s); HTTP for this line", exc)
                return super()._render(text, speaker=speaker, lang=lang, speed_alpha=speed_alpha)
            if res is None:                          # cancelled by a barge-in
                self.cancelled += 1
                return None
            pcm, ends, first = res
            if not pcm:
                self.ws_failures += 1
                logger.warning("rime ws returned no audio for %r; HTTP for this line", text[:40])
                self.http_fallbacks += 1
                return super()._render(text, speaker=speaker, lang=lang, speed_alpha=speed_alpha)
            self.calls += 1
            self.ws_calls += 1
            self.last_ms = (time.perf_counter() - t) * 1000
            self.last_first_audio_ms = first
            self.last_word_ends = ends
            return pcm
        return None

    def cancel(self) -> None:
        """Barge-in while a line is rendering (or connecting): unblock the
        waiting request and send Rime a `clear` for it."""
        self._cancel_requested = True
        ev = self._cancel
        if ev is not None:
            self._loop.call_soon_threadsafe(ev.set)

    def close(self) -> None:
        for key in list(self._conns):
            try:
                asyncio.run_coroutine_threadsafe(self._drop(key), self._loop).result(timeout=3)
            except Exception:  # noqa: BLE001
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)


class SilentSynth(RimeHTTP):
    """No network, no cache: every line becomes silence of a plausible length.
    For tests and dry runs."""

    def __init__(self) -> None:
        super().__init__(api_key="", cache_dir=None)

    def synth(self, text: str, **kw) -> bytes | None:  # noqa: ARG002
        return None


class SapiSynth(RimeHTTP):
    """
    The Windows speech synthesiser, rendered to the same 16 kHz mono PCM Rime
    returns. No key, no network -- this is what makes the full
    STT -> agent -> TTS loop checkable offline.

    It is a fallback, not the product: the voice is robotic, Hindi is only
    available if a Hindi voice is installed on the machine, and each uncached
    line costs a PowerShell process (~0.5-1 s). Cached lines are free, and the
    cache is shared with Rime's (different key, same directory).
    """

    provider = "sapi"

    def __init__(self, sample_rate: int = 16000, cache_dir: str | Path | None = ".cache/tts") -> None:
        super().__init__(api_key="", cache_dir=cache_dir, sample_rate=sample_rate)
        self.model_id = "sapi"          # keeps its cache keys distinct from Rime's

    def _render(self, text: str, *, speaker: str, lang: str, speed_alpha: float) -> bytes | None:  # noqa: ARG002
        import subprocess
        import tempfile
        import wave

        # SAPI rate is -10..10 around normal. config.SPEED_ALPHA_* runs 0.6-1.5.
        rate = max(-10, min(10, round((speed_alpha - 1.0) * 10)))
        tmp = Path(tempfile.mkdtemp(prefix="sapi_"))
        txt_path, wav_path = tmp / "line.txt", tmp / "line.wav"
        # The text goes through a file: no quoting or escaping rules to get
        # wrong, whatever the tutor decided to say.
        txt_path.write_text(text, encoding="utf-8")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {rate}; "
            "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
            f"{self.sample_rate},[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,"
            "[System.Speech.AudioFormat.AudioChannel]::Mono); "
            f'$s.SetOutputToWaveFile("{wav_path}",$f); '
            f'$s.Speak((Get-Content -Raw -Encoding UTF8 "{txt_path}")); $s.Dispose()'
        )
        t = time.perf_counter()
        try:
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                           check=True, capture_output=True, timeout=self.timeout)
            with wave.open(str(wav_path), "rb") as w:
                pcm = w.readframes(w.getnframes())
        except Exception as exc:  # noqa: BLE001
            logger.warning("sapi synth failed (%s): %s", text[:40], exc)
            return None
        finally:
            for p in (txt_path, wav_path):
                p.unlink(missing_ok=True)
            tmp.rmdir()
        self.calls += 1
        self.last_ms = (time.perf_counter() - t) * 1000
        return pcm


def default_synth() -> RimeHTTP:
    """
    Pick a speech provider. `TTS_PROVIDER` (config.py) forces one:

        rime    the product path; needs RIME_API_KEY
        sapi    Windows TTS, offline, robotic -- for checking the loop
        silent  no audio at all, timing only
        auto    Rime if RIME_API_KEY is set, else sapi   (default)
    """
    choice = (config.TTS_PROVIDER or "auto").lower()
    if choice == "silent":
        return SilentSynth()
    if choice == "sapi":
        return SapiSynth()
    rime = RimeWS() if config.RIME_TRANSPORT_MODE == "ws" else RimeHTTP()
    if choice == "rime" or rime.api_key:
        return rime
    logger.info("no RIME_API_KEY: falling back to Windows TTS (set TTS_PROVIDER=silent to disable audio)")
    return SapiSynth()


class RimeSpeaker:
    """agents.session.Speaker backed by Rime + a Player."""

    def __init__(self, player: Player, clock, synth: RimeHTTP | None = None,
                 on_text: Callable[[str, dict], None] | None = None,
                 on_event: Callable[[str, dict], None] | None = None,
                 silent_fallback: bool = True) -> None:
        self.player = player
        self.clock = clock
        self.synth = synth if synth is not None else default_synth()
        self.on_text = on_text
        self.on_event = on_event
        self.silent_fallback = silent_fallback
        self.provider = getattr(self.synth, "provider",
                                "rime" if self.synth.api_key else "text-only")
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
        n_words = word_count(text)
        t0 = time.perf_counter()
        cached = None
        if isinstance(self.synth, RimeWS) and self.synth.api_key:
            cached = self.synth.cache_lookup(text, speaker=speaker, lang=lang, speed_alpha=speed_alpha)
            if cached is None:
                self._speak_streaming(text, turn_id=turn_id, lang=lang, speaker=speaker,
                                      speed_alpha=speed_alpha, n_words=n_words, t0=t0)
                return
        pcm = cached[0] if cached else self.synth.synth(text, speaker=speaker, lang=lang, speed_alpha=speed_alpha)
        synth_ms = (time.perf_counter() - t0) * 1000
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
        audio_s = len(pcm) / (self.synth.sample_rate * 2)
        ends = getattr(self.synth, "last_word_ends", None)
        if ends and ends[-1] > 0 and audio_s > 0:
            # Rime's timestamp clock and the delivered audio drift apart by up
            # to ~20% at non-default speeds (measured). Relative positions are
            # right, so pin the last word to the end of the clip.
            k = audio_s / ends[-1]
            ends = [e * k for e in ends]
        self.player.enqueue(PcmItem(pcm=pcm, turn_id=turn_id, text=text, n_words=n_words, word_ends=ends))
        self._emit("tts", turn=turn_id, words=n_words, synth_ms=round(synth_ms), audio_s=round(audio_s, 2),
                   # The PS wants cached and uncached measurements labelled
                   # separately. Ask the synth rather than guessing from timing:
                   # a disk read can take 10 ms on a busy laptop.
                   cached=bool(getattr(self.synth, "last_cached", synth_ms < 5)),
                   first_audio_ms=(round(fa) if (fa := getattr(self.synth, "last_first_audio_ms", None)) else None),
                   timestamps=bool(ends), transport="ws" if isinstance(self.synth, RimeWS) else self.provider)

    def _speak_streaming(self, text: str, *, turn_id: int, lang: str, speaker: str, speed_alpha: float,
                         n_words: int, t0: float) -> None:
        """Websocket path, uncached line: the item goes to the player on Rime's
        first chunk (~0.4 s) and grows while it plays. The fence runs at that
        moment, not after the whole line has rendered."""
        item = PcmItem(pcm=bytearray(), turn_id=turn_id, text=text, n_words=n_words, complete=False)
        state = {"enqueued": False}

        def on_first() -> bool:
            live = self.clock.current()
            if live != turn_id:
                self._emit("tts_drop_stale", born=turn_id, live=live, text=text[:60])
                return False
            self.player.enqueue(item)
            state["enqueued"] = True
            return True

        status = self.synth.stream(text, speaker=speaker, lang=lang, speed_alpha=speed_alpha,
                                   item=item, on_first=on_first)
        synth_ms = (time.perf_counter() - t0) * 1000
        if status == "failed":
            if not self.silent_fallback:
                return
            self._emit("tts_fallback", turn=turn_id, text=text[:60])
            live = self.clock.current()
            if live != turn_id:
                return
            self.player.enqueue(PcmItem(pcm=pcm_for_words(n_words), turn_id=turn_id, text=text, n_words=n_words))
            return
        if status == "cancelled" and not state["enqueued"]:
            return                                  # fenced before any audio; already logged
        self._emit("tts", turn=turn_id, words=n_words, synth_ms=round(synth_ms), audio_s=round(item.duration_s, 2),
                   cached=False, first_audio_ms=(round(fa) if (fa := self.synth.last_first_audio_ms) else None),
                   timestamps=bool(item.word_ends), transport="ws" if status in ("done", "truncated", "cancelled")
                   else "http", streamed=status != "http", status=status)

    def stop(self) -> None:
        """Barge-in: drop everything queued and remember where the learner was."""
        with self._lock:
            self.stops += 1
            self.last_cursor = self.player.flush()
        try:
            self.synth.cancel()             # Rime-side: drop what is still rendering
        except Exception:  # noqa: BLE001
            logger.exception("synth cancel failed")
        self._emit("stop", cursor=self.last_cursor.__dict__ if self.last_cursor else None)

    def take_cursor(self) -> HeardCursor | None:
        with self._lock:
            cur, self.last_cursor = self.last_cursor, None
            return cur
