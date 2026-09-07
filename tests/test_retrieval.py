from __future__ import annotations

import pytest

import config
from agents.material import make_chunks
from agents.retrieval import Doc, HashEmbedder, HybridRetriever
from conftest import fixture_sections

IN_NOTES = [
    ("How many chambers does the heart have?", "Chambers"),
    ("What are the upper chambers called?", "Chambers"),
    ("Which is the strongest chamber?", "Chambers"),
    ("How many times does the heart beat per minute?", "The Heart"),
    ("What is the largest artery?", "Blood Flow"),
    ("What do valves do?", "Blood Flow"),
    ("Who described the circulation of blood?", "History"),
    ("What happened in 1628?", "History"),
    ("What did people believe the liver did?", "History"),
    ("What carries blood back to the heart?", "Blood Flow"),
]
OUT_OF_NOTES = [
    "How many moons does Mars have?",
    "Who wrote Hamlet?",
    "What is the boiling point of nitrogen?",
]


@pytest.fixture(scope="module")
def retriever():
    chunks = make_chunks(fixture_sections(), config.CHUNK_SENTENCES, config.CHUNK_OVERLAP)
    docs = [Doc(c["id"], c["text"], c["section_id"], c["section_title"]) for c in chunks]
    return HybridRetriever(docs, HashEmbedder())


@pytest.mark.parametrize("q,section", IN_NOTES)
def test_in_notes_questions_hit_the_right_section_above_tau(retriever, q, section):
    hit = retriever.best(q)
    assert hit is not None
    assert hit.doc.section_title == section, (q, hit.doc.section_title, round(hit.score, 3))
    assert hit.score >= config.RETRIEVAL_TAU, (q, round(hit.score, 3))


@pytest.mark.parametrize("q", OUT_OF_NOTES)
def test_out_of_notes_questions_fall_below_tau(retriever, q):
    hit = retriever.best(q)
    assert hit is None or hit.score < config.RETRIEVAL_TAU, (q, round(hit.score, 3))
