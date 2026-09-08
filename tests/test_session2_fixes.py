"""Regressions from the 2026-09-09 browser session (evidence/demo-1788901600.csv):
a Whisper hallucination answered the language question, "switch to the
language again" became the lesson topic, a restated "photosynthesis for class
6" could not replace it, and "and, um," resumed the lesson over the learner."""
from __future__ import annotations

import time

import pytest

import config
from agents.intent import is_incomplete, wants_language_step
from conftest import make_deps
from agents.graph import TutorRunner


@pytest.mark.parametrize("utter", ["and, um,", "I want to", "what is the", "so, like", "can you", "aur"])
def test_cut_off_utterances_are_incomplete(utter):
    assert is_incomplete(utter)


@pytest.mark.parametrize("utter", ["pause", "yeah", "slower,", "the heart for class six", "English,",
                                   "what does that mean", "photosynthesis for class 6."])
def test_complete_utterances_are_not_held(utter):
    assert not is_incomplete(utter)


@pytest.mark.parametrize("utter", ["Can you ask the language again?", "Let's switch to the language again.",
                                   "I want you to speak in Henry.", "speak in Hindi please", "switch to English"])
def test_language_requests_during_onboarding(utter):
    assert wants_language_step(utter)


@pytest.mark.parametrize("utter", ["the heart for class six", "photosynthesis", "English grammar for class 7"])
def test_topics_are_not_language_requests(utter):
    assert not wants_language_step(utter)


def test_onboarding_can_go_back_to_the_language_question(speaker, clock):
    r = TutorRunner(make_deps(speaker=speaker, clock=clock), session_id="s2a")
    r.start()
    r.barge_in("English")
    assert r.state["onboarding_step"] == "source"
    r.barge_in("Let's switch to the language again.")
    assert r.state["onboarding_step"] == "language"
    assert "language" in speaker.lines[-1].text.lower()
    assert r.state.get("topic") is None
    r.barge_in("English")
    assert r.state["onboarding_step"] == "source"


def test_named_language_while_asked_for_topic_is_taken(speaker, clock):
    r = TutorRunner(make_deps(speaker=speaker, clock=clock), session_id="s2b")
    r.start()
    r.barge_in("English")
    r.barge_in("speak in Hindi please")
    assert r.state["active_lang"] == "hi" and r.state["onboarding_step"] == "source"


def test_restated_topic_and_class_replaces_a_misheard_topic(speaker, clock):
    r = TutorRunner(make_deps(speaker=speaker, clock=clock), session_id="s2c")
    r.start()
    r.barge_in("English")
    r.barge_in("the hard")                         # "the heart", misheard and accepted as the topic
    assert r.state["topic"] == "the hard" and not r.state.get("grade")
    r.barge_in("photosynthesis for class 6")       # the learner restates everything
    assert r.state["topic"] == "photosynthesis" and r.state["grade"] == "class 6"


def test_fragment_is_held_then_joined_with_the_rest(monkeypatch):
    from test_voice_bridge import build
    monkeypatch.setattr(config, "FRAGMENT_HOLD_S", 0.6)
    bridge, player, spoken, events = build()
    bridge.start()
    assert bridge.wait_idle(20)
    bridge.on_speech_start(); bridge.on_transcript("English", lang="en"); assert bridge.wait_idle(20)
    bridge.on_speech_start(); bridge.on_transcript("the heart for class six", lang="en"); assert bridge.wait_idle(30)
    n_spoken = len(spoken)
    bridge.on_speech_start()
    bridge.on_transcript("and, um,", lang="en")
    time.sleep(0.2)
    assert bridge.wait_idle(5)
    assert any(n == "fragment_hold" for n, _ in events)
    assert len(spoken) == n_spoken, "the tutor must stay quiet while the learner is mid-thought"
    bridge.on_speech_start()
    bridge.on_transcript("how many chambers does the heart have", lang="en")
    assert bridge.wait_idle(20)
    delivered = [p["text"] for n, p in events if n == "transcript" and not p.get("held")]
    assert delivered[-1] == "how many chambers does the heart have"   # leading fillers stripped
    # The joined utterance reached the graph as one turn (the instant player has
    # already finished the whole lesson, so the answer itself is not asserted).
    assert [n for n, _ in events].count("graph_turn_done") >= 3
    bridge.close()


def test_fragment_is_delivered_alone_when_nothing_follows(monkeypatch):
    from test_voice_bridge import build
    monkeypatch.setattr(config, "FRAGMENT_HOLD_S", 0.3)
    bridge, player, spoken, events = build()
    bridge.start()
    assert bridge.wait_idle(20)
    bridge.on_speech_start(); bridge.on_transcript("English", lang="en"); assert bridge.wait_idle(20)
    bridge.on_speech_start(); bridge.on_transcript("the heart for class six", lang="en"); assert bridge.wait_idle(30)
    bridge.on_speech_start()
    bridge.on_transcript("and, um,", lang="en")
    deadline = time.perf_counter() + 3
    while not any(n == "fragment_release" for n, _ in events) and time.perf_counter() < deadline:
        time.sleep(0.05)
    assert any(n == "fragment_release" for n, _ in events)
    assert bridge.wait_idle(20)
    bridge.close()


# --- second phone session (evidence/demo-1788902635.csv) ---------------------

def test_long_question_at_language_step_reasks_instead_of_guessing(speaker, clock):
    r = TutorRunner(make_deps(speaker=speaker, clock=clock), session_id="s2d")
    r.start()
    r.barge_in("Can you ask me again which language do you want me to study in?")
    assert r.state["onboarding_step"] == "language"
    assert "language" in speaker.lines[-1].text.lower()
    r.barge_in("English")
    assert r.state["onboarding_step"] == "source"


@pytest.mark.parametrize("utter,topic", [
    ("I want to learn about machines, electrical machines", "electrical machines"),
    ("the heart, the human heart for class 6", "the human heart"),
    ("photosynthesis, respiration", "photosynthesis respiration"),   # two topics, not a correction: unchanged
])
def test_self_corrected_topic_keeps_the_refinement(utter, topic):
    from agents.intent import parse_topic_grade
    assert parse_topic_grade(utter)[0] == topic
