"""Streaming playback: an item can be enqueued before its last chunk exists.
Players must play what has arrived, wait (silence) for more, and only finish
the item once it is complete. Offline."""
from __future__ import annotations

import time

from voice.player import BYTES_PER_SAMPLE, SAMPLE_RATE, PcmItem, TimedPlayer

SEC = SAMPLE_RATE * BYTES_PER_SAMPLE


def test_incomplete_item_reports_availability_and_estimates_words():
    item = PcmItem(pcm=bytearray(), turn_id=1, text="a b c d e f", n_words=6, complete=False)
    assert item.available == 0 and not item.exhausted
    item.append(bytes(SEC))                          # 1 s arrived
    item.played_bytes = int(SEC * 0.76)              # 0.76 s played -> 2 words at 0.38 s/word
    assert item.words_heard == 2
    item.finish([0.3, 0.6, 0.9, 1.2, 1.5, 2.0])      # ends pinned to the real 1.0 s clip
    assert item.complete and item.word_ends[-1] == 1.0
    assert item.words_heard == 5                     # 0.76 s: ends <= 0.76 after rescale: 0.15,0.3,0.45,0.6,0.75


def test_finish_drops_odd_trailing_byte():
    item = PcmItem(pcm=bytearray(b"\x00" * 7), turn_id=1, text="x", n_words=1, complete=False)
    item.finish(None)
    assert len(item.pcm) == 6


def test_timed_player_waits_for_more_bytes_then_drains():
    p = TimedPlayer(speed=50.0)                      # fast clock so the test stays quick
    drained = []
    p.set_on_drained(lambda: drained.append(time.perf_counter()))
    item = PcmItem(pcm=bytearray(bytes(SEC // 4)), turn_id=1, text="one two three four", n_words=4, complete=False)
    p.enqueue(item)
    time.sleep(0.3)
    assert not p.is_idle() and not drained, "must not finish an item that is still streaming in"
    assert item.played_bytes == SEC // 4              # played everything that had arrived, no more
    cur = p.flush()                                   # a barge-in mid-stream still yields a cursor
    assert cur.turn_id == 1 and cur.playing and cur.words_heard is not None
    p.enqueue(item)                                   # (tests reuse the item) now complete it
    item.append(bytes(SEC // 4))
    item.finish(None)
    deadline = time.perf_counter() + 2
    while not drained and time.perf_counter() < deadline:
        time.sleep(0.02)
    assert drained and p.is_idle()
    p.close()
