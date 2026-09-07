"""Real-model pass (AGENT_GAPS step 4) and the direct-answer path: an
out-of-notes question goes to the strong model first and only to the web when
the model says LOOKUP. Everything here runs offline with fake models."""
from __future__ import annotations

import time

from agents.gap_filler import GapFiller
from agents.graph import TutorRunner
from agents.intent import classify_rules, is_ask_permission, parse_llm_json
from agents.llm import clean_output
from conftest import MARS_SNIPPET, TextSpeaker, TurnClock, make_deps, onboard


class DirectLLM:
    """Answers general-knowledge questions itself, asks for a lookup otherwise."""
    provider = model = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, system: str, user: str) -> str:
        if "lesson notes do not cover" not in system:
            return ""                                   # ingest / explain: stay on the fallback path
        self.calls.append(user)
        if "France" in user:
            return "The capital of France is Paris. It is also the country's largest city."
        return "LOOKUP"


def test_trivial_question_is_answered_by_the_model_without_web(speaker):
    web_calls: list[str] = []

    def web(q):
        web_calls.append(q)
        return [{"title": "x", "snippet": "irrelevant", "url": "u"}]

    llm = DirectLLM()
    r = TutorRunner(make_deps(speaker=speaker, llm_strong=llm, web_search=web), "direct")
    onboard(r)
    n = len(speaker.lines)
    r.barge_in("what is the capital of France", words_heard=6)
    new = [l.text for l in speaker.lines[n:]]
    assert any("Paris" in x for x in new)
    assert web_calls == [], "trivial question must not hit the web"
    assert r.state["answer_mode"] == "direct"
    assert r.state["recent_exchanges"][-1]["q"] == "what is the capital of France"
    assert any("back to where we were" in x for x in new) and "muscular organ" in new[-1]


def test_model_says_lookup_then_web_is_used(speaker):
    web_calls: list[str] = []

    def web(q):
        web_calls.append(q)
        return [{"title": "Mars", "snippet": MARS_SNIPPET, "url": "u"}]

    r = TutorRunner(make_deps(speaker=speaker, llm_strong=DirectLLM(), web_search=web), "lookup")
    onboard(r)
    n = len(speaker.lines)
    r.barge_in("who won the local election yesterday")
    assert web_calls == ["who won the local election yesterday"]
    assert r.state["answer_mode"] == "web"
    assert any(MARS_SNIPPET in l.text for l in speaker.lines[n:])   # extractive fallback from the web hit


def test_stub_model_still_goes_straight_to_web(lesson, speaker):
    n = len(speaker.lines)
    lesson.barge_in("How many moons does Mars have?")
    assert any("Phobos" in l.text for l in speaker.lines[n:])
    assert lesson.state["answer_mode"] == "web"


def test_in_notes_question_never_asks_the_model_directly(speaker):
    llm = DirectLLM()
    r = TutorRunner(make_deps(speaker=speaker, llm_strong=llm), "notes")
    onboard(r)
    r.barge_in("How many chambers does the heart have?")
    assert llm.calls == []
    assert r.state["answer_mode"] == "notes"


def test_stale_direct_answer_is_discarded():
    sp, clock = TextSpeaker(), TurnClock()

    class InterruptingLLM(DirectLLM):
        def complete(self, system, user):
            out = super().complete(system, user)
            if out:
                clock.bump()                             # learner speaks during the model call
            return out

    r = TutorRunner(make_deps(speaker=sp, clock=clock, llm_strong=InterruptingLLM()), "stale-direct")
    onboard(r)
    n = len(sp.lines)
    r.barge_in("what is the capital of France")
    assert not any("Paris" in l.text for l in sp.lines[n:])
    assert r.state["stale_drops"] == 1 and not r.finished


# ------------------------------------------------------------- gap filler once per turn
def test_gap_filler_fires_once_per_turn():
    sp, clock = TextSpeaker(), TurnClock()
    gf = GapFiller(clock, deadline_ms=40)
    fired: list[str] = []
    turn = clock.bump()
    gf.arm(turn, lambda: fired.append("a"))
    time.sleep(0.12)
    gf.arm(turn, lambda: fired.append("b"))              # second slow step, same turn
    time.sleep(0.12)
    assert fired == ["a"]
    turn2 = clock.bump()
    gf.arm(turn2, lambda: fired.append("c"))             # new turn: allowed again
    time.sleep(0.12)
    assert fired == ["a", "c"]


# ------------------------------------------------------------- JSON parser hardening
def test_parse_llm_json_tolerates_string_nulls_and_bad_fields():
    c = parse_llm_json('{"intent": "question", "command": "null", "session_cmd": "null", '
                       '"nav_target": "null", "reply_lang": "hi", "command_arg": "hi"}')
    assert c.intent == "question" and c.command is None and c.session_cmd is None
    assert c.nav_target is None and c.reply_lang is None and c.command_arg is None
    c = parse_llm_json('{"intent": "explain", "reply_lang": "hi"}')
    assert c.intent == "explain" and c.reply_lang == "hi"
    assert parse_llm_json('{"intent": "navigate", "nav_target": {"kind": "topic"}}') is None
    assert parse_llm_json('{"intent": "command", "command": "louder"}') is None
    c = parse_llm_json('Sure! {"intent": "navigate", "nav_target": {"kind": "topic", "value": "history"}}')
    assert c.nav_target == {"kind": "topic", "value": "history"}


def test_clean_output_strips_typography_and_markdown():
    assert clean_output("oxygen‑rich blood — the **aorta**") == 'oxygen-rich blood, the aorta'
    assert clean_output("1. First line\n2. Second line") == "1. First line\n2. Second line"
    assert clean_output("- a bullet\n# heading") == "a bullet\nheading"


# ------------------------------------------------------------- rule additions
def test_new_rules_from_the_probe():
    assert classify_rules("what").intent == "explain"
    assert classify_rules("wait what?").intent == "explain"
    assert classify_rules("huh").intent == "explain"
    assert classify_rules("how do you pronounce that").intent == "explain"
    assert classify_rules("spell that").intent == "explain"
    assert classify_rules("I don't get it").intent == "explain"
    assert classify_rules("I'm back", paused=True).session_cmd == "continue"
    assert classify_rules("what is the aorta").intent == "question"     # "what" + more is still a question
    assert is_ask_permission("I have a question")
    assert is_ask_permission("um, can I ask something?")
    assert not is_ask_permission("I have a question about valves")


def test_i_have_a_question_gets_go_ahead(lesson, speaker):
    lesson.barge_in("I have a question")
    assert speaker.lines[-1].text == "Sure, go ahead."
    assert lesson.state["clarify_count"] == 0
    assert not lesson.finished
    n = len(speaker.lines)
    lesson.barge_in("How many chambers does the heart have?")
    assert any("four chambers" in l.text.lower() for l in speaker.lines[n:])
