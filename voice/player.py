"""
Playback engines. A Player takes PCM (16 kHz, mono, int16) items tagged with
the turn they belong to, plays them in order, and can be flushed instantly.

It is also where the heard cursor comes from: how many words of the item that
was playing had actually been rendered when the learner interrupted. Rime's
HTTP path gives no word timestamps, so words are spread evenly over the clip's
duration -- accurate to a word or two, which is what the graph rounds to a
sentence boundary anyway.

Three implementations:
  LocalPlayer     laptop speakers via sounddevice (demo without LiveKit)
  LiveKitPlayer   publishes into a LiveKit room as the tutor's audio track
  ScriptedPlayer  tests: "plays" instantly or on command, no audio
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Protocol

logger = logging.getLogger("v-tutor.player")

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


@dataclass
class PcmItem:
    pcm: bytes
    turn_id: int
    text: str
    n_words: int
    kind: str = "speech"
    played_bytes: int = 0
    seq: int = 0

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    @property
    def words_heard(self) -> int:
        if not self.pcm or self.n_words == 0:
            return 0
        frac = min(1.0, self.played_bytes / len(self.pcm))
        return int(round(frac * self.n_words))


@dataclass
class HeardCursor:
    """What the learner had heard when playback stopped."""
    turn_id: int
    words_heard: int | None     # None = everything queued had been heard
    text: str
    was_last_item: bool
    playing: bool


class Player(Protocol):
    def enqueue(self, item: PcmItem) -> None: ...
    def flush(self) -> HeardCursor: ...
    def is_idle(self) -> bool: ...
    def last_turn(self) -> int | None: ...
    def set_on_drained(self, cb: Callable[[], None] | None) -> None: ...
    def close(self) -> None: ...


class _QueueMixin:
    """Shared bookkeeping: ordered queue, cursor, drained callback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._q: deque[PcmItem] = deque()
        self._current: PcmItem | None = None
        self._seq = 0
        self._last_turn: int | None = None
        self._on_drained: Callable[[], None] | None = None

    def set_on_drained(self, cb: Callable[[], None] | None) -> None:
        self._on_drained = cb

    def last_turn(self) -> int | None:
        return self._last_turn

    def is_idle(self) -> bool:
        with self._lock:
            return self._current is None and not self._q

    def _push(self, item: PcmItem) -> None:
        with self._lock:
            self._seq += 1
            item.seq = self._seq
            self._q.append(item)
            self._last_turn = item.turn_id

    def _cursor_locked(self) -> HeardCursor:
        cur = self._current
        if cur is None:
            if self._q:                                   # queued but not started
                nxt = self._q[0]
                return HeardCursor(nxt.turn_id, 0, nxt.text, False, False)
            return HeardCursor(self._last_turn or 0, None, "", True, False)
        was_last = not self._q
        return HeardCursor(cur.turn_id, cur.words_heard, cur.text, was_last, True)

    def _drop_all_locked(self) -> None:
        self._q.clear()
        self._current = None

    def _fire_drained(self) -> None:
        cb = self._on_drained
        if cb is not None:
            try:
                cb()
            except Exception:  # noqa: BLE001
                logger.exception("on_drained callback failed")


# --------------------------------------------------------------------------
# Local speakers
# --------------------------------------------------------------------------

class LocalPlayer(_QueueMixin):
    """sounddevice OutputStream; the callback pulls from the queue so the
    played-bytes counter is exact (it is what the sound card consumed)."""

    def __init__(self, device: int | str | None = None, block_ms: int = 20) -> None:
        super().__init__()
        import sounddevice as sd
        self._sd = sd
        self._closed = False
        self._stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=device,
            blocksize=int(SAMPLE_RATE * block_ms / 1000), callback=self._callback,
        )
        self._stream.start()

    def enqueue(self, item: PcmItem) -> None:
        self._push(item)

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ARG002
        need = frames * BYTES_PER_SAMPLE
        out = bytearray()
        drained = False
        with self._lock:
            while len(out) < need:
                if self._current is None:
                    if not self._q:
                        break
                    self._current = self._q.popleft()
                cur = self._current
                chunk = cur.pcm[cur.played_bytes:cur.played_bytes + (need - len(out))]
                if not chunk:
                    self._current = None
                    if not self._q:
                        drained = True
                    continue
                out += chunk
                cur.played_bytes += len(chunk)
                if cur.played_bytes >= len(cur.pcm):
                    self._current = None
                    if not self._q:
                        drained = True
        if len(out) < need:
            out += bytes(need - len(out))
        outdata[:] = bytes(out)
        if drained:
            self._fire_drained()

    def flush(self) -> HeardCursor:
        with self._lock:
            cur = self._cursor_locked()
            self._drop_all_locked()
        return cur

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------
# LiveKit audio track
# --------------------------------------------------------------------------

class LiveKitPlayer(_QueueMixin):
    """Feeds an rtc.AudioSource from a worker thread in 20 ms frames.
    capture_frame() blocks while the source's small queue is full, so the
    played-bytes counter tracks real time to within `queue_size_ms`."""

    FRAME_MS = 20

    def __init__(self, source, loop: asyncio.AbstractEventLoop, queue_size_ms: int = 200) -> None:
        super().__init__()
        from livekit import rtc
        self._rtc = rtc
        self._source = source
        self._loop = loop
        self._queue_ms = queue_size_ms
        self._wake = threading.Event()
        self._closed = False
        self._gen = 0                                   # bumped on flush; the feeder checks it
        self._thread = threading.Thread(target=self._run, name="lk-player", daemon=True)
        self._thread.start()

    def enqueue(self, item: PcmItem) -> None:
        self._push(item)
        self._wake.set()

    def _run(self) -> None:
        frame_bytes = SAMPLE_RATE * BYTES_PER_SAMPLE * self.FRAME_MS // 1000
        while not self._closed:
            with self._lock:
                if self._current is None and self._q:
                    self._current = self._q.popleft()
                cur, gen = self._current, self._gen
            if cur is None:
                self._wake.wait(timeout=0.1)
                self._wake.clear()
                continue
            chunk = cur.pcm[cur.played_bytes:cur.played_bytes + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk = chunk + bytes(frame_bytes - len(chunk))
            frame = self._rtc.AudioFrame(data=chunk, sample_rate=SAMPLE_RATE, num_channels=1,
                                         samples_per_channel=frame_bytes // BYTES_PER_SAMPLE)
            try:
                fut = asyncio.run_coroutine_threadsafe(self._source.capture_frame(frame), self._loop)
                fut.result(timeout=2.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("capture_frame failed: %s", exc)
                time.sleep(self.FRAME_MS / 1000)
            drained = False
            with self._lock:
                if self._gen != gen or self._current is not cur:
                    continue                            # flushed while we were capturing
                cur.played_bytes += frame_bytes
                if cur.played_bytes >= len(cur.pcm):
                    self._current = None
                    drained = not self._q
            if drained:
                time.sleep(self._queue_ms / 1000)       # let the source's queue actually play out
                with self._lock:
                    still = self._current is None and not self._q and self._gen == gen
                if still:
                    self._fire_drained()

    def flush(self) -> HeardCursor:
        with self._lock:
            cur = self._cursor_locked()
            if cur.words_heard is not None and self._queue_ms:
                # bytes captured but still sitting in the source queue were not heard
                ahead = SAMPLE_RATE * BYTES_PER_SAMPLE * self._queue_ms // 1000
                if self._current is not None:
                    self._current.played_bytes = max(0, self._current.played_bytes - ahead)
                    cur = self._cursor_locked()
            self._drop_all_locked()
            self._gen += 1
        try:
            self._loop.call_soon_threadsafe(self._source.clear_queue)
        except Exception:  # noqa: BLE001
            pass
        return cur

    def close(self) -> None:
        self._closed = True
        self._wake.set()


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

class ScriptedPlayer(_QueueMixin):
    """No audio. Items sit in the queue until the test calls play_all() or
    play_words(n); `auto=True` plays each item instantly on enqueue."""

    def __init__(self, auto: bool = False) -> None:
        super().__init__()
        self.auto = auto
        self.enqueued: list[PcmItem] = []
        self.flushes: list[HeardCursor] = []

    def enqueue(self, item: PcmItem) -> None:
        self._push(item)
        self.enqueued.append(item)
        if self.auto:
            self.play_all()

    def start_next(self) -> PcmItem | None:
        with self._lock:
            if self._current is None and self._q:
                self._current = self._q.popleft()
            return self._current

    def finish_current(self) -> None:
        """Current item fully played; move on WITHOUT firing on_drained."""
        with self._lock:
            self._current = None

    def play_words(self, n: int) -> None:
        """Advance the current item so that n of its words count as heard."""
        cur = self.start_next()
        if cur is None:
            return
        with self._lock:
            cur.played_bytes = int(len(cur.pcm) * min(1.0, n / max(1, cur.n_words)))

    def play_all(self) -> None:
        with self._lock:
            self._q.clear()
            self._current = None
        self._fire_drained()

    def flush(self) -> HeardCursor:
        with self._lock:
            cur = self._cursor_locked()
            self._drop_all_locked()
        self.flushes.append(cur)
        return cur

    def close(self) -> None:
        pass


class TimedPlayer(_QueueMixin):
    """Silent, but takes real time: items 'play' at 16 kHz on a background
    thread. For dry runs where the tutor's cadence must be realistic but no
    sound card is wanted. `speed` > 1 plays faster than real time."""

    STEP_S = 0.02

    def __init__(self, speed: float = 1.0) -> None:
        super().__init__()
        self.speed = speed
        self._closed = False
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name="timed-player", daemon=True)
        self._thread.start()

    def enqueue(self, item: PcmItem) -> None:
        self._push(item)
        self._wake.set()

    def _run(self) -> None:
        step_bytes = int(SAMPLE_RATE * BYTES_PER_SAMPLE * self.STEP_S * self.speed)
        while not self._closed:
            with self._lock:
                if self._current is None and self._q:
                    self._current = self._q.popleft()
                cur = self._current
            if cur is None:
                self._wake.wait(timeout=0.1)
                self._wake.clear()
                continue
            time.sleep(self.STEP_S)
            drained = False
            with self._lock:
                if self._current is not cur:
                    continue                            # flushed meanwhile
                cur.played_bytes += step_bytes
                if cur.played_bytes >= len(cur.pcm):
                    self._current = None
                    drained = not self._q
            if drained:
                self._fire_drained()

    def flush(self) -> HeardCursor:
        with self._lock:
            cur = self._cursor_locked()
            self._drop_all_locked()
        return cur

    def close(self) -> None:
        self._closed = True
        self._wake.set()


def pcm_for_words(n_words: int, seconds_per_word: float = 0.4) -> bytes:
    """Silent PCM of a plausible length, for tests and dry runs."""
    n = int(SAMPLE_RATE * seconds_per_word * max(1, n_words))
    return bytes(n * BYTES_PER_SAMPLE)


__all__ = ["PcmItem", "HeardCursor", "Player", "LocalPlayer", "LiveKitPlayer", "ScriptedPlayer",
           "TimedPlayer", "pcm_for_words", "SAMPLE_RATE", "field"]
