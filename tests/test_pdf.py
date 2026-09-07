"""The PDF path against generated fixtures (scripts/make_fixtures.py)."""
from __future__ import annotations

import re
from pathlib import Path

from agents.graph import TutorRunner
from agents.material import parse_pdf
from conftest import make_deps

FIX = Path(__file__).parent / "fixtures"
CHAPTER = str(FIX / "sample_chapter.pdf")
SCANNED = str(FIX / "scanned.pdf")


def test_sample_chapter_parses_into_sections():
    res = parse_pdf([CHAPTER])
    assert res is not None
    sections, lang, title = res
    titles = [s["title"] for s in sections]
    assert "Chambers" in titles and "History" in titles and lang == "en"
    assert title == "sample_chapter"
    body = "\n".join(s["text"] for s in sections)
    assert "Science for Class 6" not in body, "running header must be dropped"
    assert not any(re.fullmatch(r"\s*\d+\s*", line) for line in body.splitlines()), "bare page numbers must be dropped"
    assert "four chambers" in body


def test_scanned_pdf_has_no_text_layer():
    assert parse_pdf([SCANNED]) is None


def test_pdf_onboarding_skips_topic_question(speaker):
    r = TutorRunner(make_deps(speaker=speaker, pdf_parse=parse_pdf), "pdf", pdf_paths=[CHAPTER])
    r.start()
    r.barge_in("English")
    st = r.state
    assert st["source_kind"] == "pdf" and st["onboarding_step"] == "done"
    texts = [l.text for l in speaker.lines]
    assert any("Give me a moment" in t for t in texts)
    assert "sample_chapter" in texts[-1] or "muscular organ" in texts[-1]
    r.barge_in("How many chambers does the heart have?")
    assert any("four chambers" in l.text.lower() for l in speaker.lines)


def test_unreadable_pdf_falls_back_to_asking_for_a_topic(speaker):
    r = TutorRunner(make_deps(speaker=speaker, pdf_parse=parse_pdf), "scan", pdf_paths=[SCANNED])
    r.start()
    r.barge_in("English")
    assert "can't read that file" in speaker.lines[-1].text
    assert r.state["onboarding_step"] == "source" and r.state["pdf_paths"] == []
    r.barge_in("the heart for class six")
    assert r.state["onboarding_step"] == "done"
