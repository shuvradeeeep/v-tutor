"""
Dead-air watchdog. Arm it when entering something slow (web search, an LLM
call, lesson preparation); if nothing has been spoken by the deadline it says a
short filler ("Let me check that."); the next real speech disarms it.

Fenced like everything else: the filler carries the turn it was armed under and
stays silent if the learner has interrupted since. Otherwise the tutor would say
"let me check that" about a question the learner already abandoned.
"""
from __future__ import annotations

import threading
from typing import Callable


class GapFiller:
    def __init__(self, clock, deadline_ms: int = 700) -> None:
        self.clock = clock
        self.deadline = deadline_ms / 1000.0
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._armed_turn: int | None = None
        self.fired: list[int] = []            # turns on which a filler was actually spoken

    def arm(self, turn_id: int, say: Callable[[], None]) -> None:
        with self._lock:
            self._cancel()
            self._armed_turn = turn_id
            self._timer = threading.Timer(self.deadline, self._fire, args=(turn_id, say))
            self._timer.daemon = True
            self._timer.start()

    def disarm(self) -> None:
        with self._lock:
            self._cancel()
            self._armed_turn = None

    def _cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _fire(self, turn_id: int, say: Callable[[], None]) -> None:
        with self._lock:
            if self._armed_turn != turn_id or self.clock.current() != turn_id:
                return                        # disarmed, or the learner moved on: stay quiet
            self._armed_turn = None
        say()
        self.fired.append(turn_id)
