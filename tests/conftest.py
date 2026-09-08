"""
Shared fixtures. Everything runs offline: stub LLM, hash embedder, canned
material, canned web results. No API keys, no network, no audio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config                                            # noqa: E402
config.PHRASE_VARIETY = False        # tests assert canonical wording (variant 0)

from agents.graph import TutorRunner                     # noqa: E402
from agents.llm import StubLLM                           # noqa: E402
from agents.material import sections_from_plaintext      # noqa: E402
from agents.retrieval import HashEmbedder                # noqa: E402
from agents.session import Deps, TextSpeaker, TurnClock  # noqa: E402

# A small "textbook chapter" with names and numbers, so retrieval has teeth.
FIXTURE_TEXT = """\
== The Heart ==
The heart is a muscular organ about the size of a fist. It pumps blood through the whole body. An adult heart beats about 72 times per minute at rest.

== Chambers ==
The heart has four chambers. The two upper chambers are called atria. The two lower chambers are called ventricles. The left ventricle is the strongest chamber.

== Blood Flow ==
Blood leaves the heart through the aorta, the largest artery in the body. Veins carry blood back to the heart. Valves stop blood from flowing backwards.

== History ==
William Harvey described the circulation of blood in 1628. Before him, many people believed the liver made blood.
"""

MARS_SNIPPET = "Mars has two moons, Phobos and Deimos."


def fixture_sections():
    return sections_from_plaintext(FIXTURE_TEXT, title="The Heart", source_url="fixture://heart")


def fake_wiki(topic: str, lang: str):
    return (fixture_sections(), "en", "fixture://heart")


def fake_web(query: str):
    return [{"title": "Mars", "snippet": MARS_SNIPPET, "url": "https://example.test/mars"}]


def make_deps(**overrides) -> Deps:
    clock = overrides.pop("clock", None) or TurnClock()
    speaker = overrides.pop("speaker", None) or TextSpeaker()
    kw = dict(
        clock=clock, speaker=speaker,
        llm_fast=StubLLM(), llm_strong=StubLLM(),
        embedder=HashEmbedder(),
        web_search=fake_web, wiki_fetch=fake_wiki,
        pdf_parse=lambda paths: None,
        stress_delay_ms=0,
    )
    kw.update(overrides)
    return Deps(**kw)


def onboard(runner: TutorRunner, lang_reply: str = "English",
            topic_reply: str = "the heart for class six") -> None:
    """Drive the two spoken onboarding questions until the first beat is spoken."""
    runner.start()
    runner.barge_in(lang_reply)
    runner.barge_in(topic_reply)


@pytest.fixture
def speaker() -> TextSpeaker:
    return TextSpeaker()


@pytest.fixture
def clock() -> TurnClock:
    return TurnClock()


@pytest.fixture
def runner(speaker, clock) -> TutorRunner:
    return TutorRunner(make_deps(speaker=speaker, clock=clock), session_id="t")


@pytest.fixture
def lesson(runner) -> TutorRunner:
    onboard(runner)
    return runner
