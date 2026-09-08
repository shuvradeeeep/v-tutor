"""Phrase variety, the exact heard cursor from word timestamps, and the
coda speed mapping. Offline."""
from __future__ import annotations

import pytest

import config
from agents import strings
from agents.strings import t, variants
from voice.player import PcmItem


def test_every_variant_formats_and_variant_zero_is_canonical():
    for lang, table in strings.STRINGS.items():
        for key in table:
            vs = variants(lang, key)
            assert vs and all(isinstance(v, str) and v.strip() for v in vs), (lang, key)
            fmt = {k: "x" for k in ("title", "topic", "grade", "options", "langs", "minutes",
                                    "done", "total", "sections")}
            for i in range(len(vs) + 1):            # wraps past the end
                assert t(lang, key, variant=i, **fmt)
            assert t(lang, key, **fmt) == t(lang, key, variant=0, **fmt)


def test_fillers_and_bridges_have_alternatives_in_english():
    for key in ("filler_check", "filler_moment", "bridge", "ask_language", "goodbye"):
        assert len(variants("en", key)) >= 2, key


def test_variety_rotates_by_turn(monkeypatch):
    from agents.nodes import TutorNodes
    monkeypatch.setattr(config, "PHRASE_VARIETY", True)
    nodes = TutorNodes.__new__(TutorNodes)          # _T needs no deps
    said = {nodes._T({"turn_id": i, "active_lang": "en"}, "filler_check") for i in range(8)}
    assert len(said) >= 2
    monkeypatch.setattr(config, "PHRASE_VARIETY", False)
    assert nodes._T({"turn_id": 5, "active_lang": "en"}, "filler_check") == "Let me check that."


def test_words_heard_uses_word_timestamps_when_present():
    rate = 16000 * 2
    item = PcmItem(pcm=bytes(rate * 4), turn_id=1, text="one two three four", n_words=4,
                   word_ends=[0.5, 1.5, 2.5, 3.5])
    item.played_bytes = int(rate * 1.6)            # 1.6 s in: "one", "two" fully heard
    assert item.words_heard == 2
    item.played_bytes = int(rate * 0.2)
    assert item.words_heard == 0
    item.played_bytes = rate * 4
    assert item.words_heard == 4


def test_words_heard_falls_back_to_even_spread_without_timestamps():
    rate = 16000 * 2
    item = PcmItem(pcm=bytes(rate * 4), turn_id=1, text="one two three four", n_words=4)
    item.played_bytes = rate * 2
    assert item.words_heard == 2


@pytest.mark.parametrize("pace,expected", [(1.0, 1.0), (0.7, 1.0 / 0.7), (1.5, 1.0 / 1.5), (0.1, config.TIME_SCALE_MAX)])
def test_pace_maps_to_coda_time_scale_factor(pace, expected):
    from voice.tts import time_scale_factor
    assert time_scale_factor(pace) == pytest.approx(min(config.TIME_SCALE_MAX, expected), rel=1e-3)
