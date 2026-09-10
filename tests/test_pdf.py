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


def test_speaking_upload_request_asks_for_document_then_teaches_it(speaker):
    r = TutorRunner(make_deps(speaker=speaker, pdf_parse=parse_pdf), "upload")
    r.start()
    r.barge_in("English")
    r.barge_in("I want to upload a document")

    assert r.state["awaiting_document"] is True
    assert r.state["onboarding_step"] == "source"
    assert "please send the document" in speaker.lines[-1].text.lower()

    r.load_pdf([CHAPTER])
    assert r.state["source_kind"] == "pdf"
    assert r.state["awaiting_document"] is False


def test_upload_request_mid_lesson_waits_without_erasing_topic(speaker):
    r = TutorRunner(make_deps(speaker=speaker, pdf_parse=parse_pdf), "upload-mid")
    r.start()
    r.barge_in("English")
    r.barge_in("the heart for class six")
    assert r.state["topic"] == "the heart"

    r.barge_in("please let me upload a file")

    assert r.state["awaiting_document"] is True
    assert r.state["onboarding_step"] == "source"
    assert r.state["topic"] == "the heart"
    assert "please send the document" in speaker.lines[-1].text.lower()


def test_topic_switch_from_pdf_does_not_reparse_old_document(speaker):
    from agents.material import sections_from_plaintext

    frost = sections_from_plaintext(
        "== Robert Frost ==\nRobert Frost was an American poet.\n",
        title="Robert Frost", source_url="fixture://frost")

    def wiki(topic: str, lang: str):
        return frost, "en", "fixture://frost"

    r = TutorRunner(make_deps(speaker=speaker, pdf_parse=parse_pdf, wiki_fetch=wiki), "pdf-switch",
                    pdf_paths=[CHAPTER])
    r.start()
    r.barge_in("English")
    assert r.state["source_kind"] == "pdf"

    r.barge_in("Let's switch the topic.")
    assert r.state["onboarding_step"] == "source"
    assert r.state["pdf_paths"] == []

    r.barge_in("The life of Robert Frost for class 9.")

    assert r.state["source_kind"] == "topic"
    assert r.state["topic"] == "The life of Robert Frost"
    assert r.state["grade"] == "class 9"
    assert r.state["source_title"] == "Robert Frost"


def test_unreadable_pdf_falls_back_to_asking_for_a_topic(speaker):
    r = TutorRunner(make_deps(speaker=speaker, pdf_parse=parse_pdf), "scan", pdf_paths=[SCANNED])
    r.start()
    r.barge_in("English")
    assert "can't read that file" in speaker.lines[-1].text
    assert r.state["onboarding_step"] == "source" and r.state["pdf_paths"] == []
    r.barge_in("the heart for class six")
    assert r.state["onboarding_step"] == "done"
