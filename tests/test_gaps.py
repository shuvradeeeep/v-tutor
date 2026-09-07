"""Tests for the gap-closing pass (docs/AGENT_GAPS.md): bugs, writing for the
ear, lesson cap + two-stage preparation, LLM JSON path, web provider, gap
filler, evidence writer, persistence."""
from __future__ import annotations

import time

import pytest

import config
from agents.evidence import EvidenceWriter
from agents.gap_filler import GapFiller
from agents.graph import TutorRunner, make_checkpointer
from agents.intent import classify, classify_rules, parse_grade_only
from agents.material import (DisambiguationError, clean_for_speech, disambiguation_options,
                             is_equation, make_beats, sections_from_plaintext)
from agents.web import duckduckgo_search, make_web_search, stub_search
from conftest import MARS_SNIPPET, TextSpeaker, TurnClock, fixture_sections, make_deps, onboard


# ---------------------------------------------------------------- step 1: bugs
def test_bare_stop_pauses(lesson, speaker):
    lesson.barge_in("stop")
    assert lesson.state["paused"] is True
    assert "pausing" in speaker.lines[-1].text.lower()


@pytest.mark.parametrize("utter,intent,extra", [
    ("dheere", "command", {"command": "slower"}),
    ("thoda dheere bolo", "command", {"command": "slower"}),
    ("dobara", "command", {"command": "repeat"}),
    ("ruko", "session", {"session_cmd": "pause"}),
    ("aage badho", "session", {"session_cmd": "continue"}),
    ("band karo", "session", {"session_cmd": "quit"}),
    ("wapas jao", "navigate", {"nav_target": {"kind": "prev"}}),
    ("agla", "navigate", {"nav_target": {"kind": "next"}}),
    ("iska matlab kya hai", "explain", {}),
    ("samjhao", "explain", {}),
    ("hindi me bolo", "explain", {"reply_lang": "hi"}),
    ("haan", "backchannel", {}),
    ("theek hai", "backchannel", {}),
    ("dil me kitne kaksh hote hain", "question", {}),
    ("pura lesson hindi me padhao", "command", {"command": "switch_lesson_lang", "command_arg": "hi"}),
])
def test_hinglish_rules(utter, intent, extra):
    c = classify_rules(utter)
    assert c is not None and c.intent == intent, (utter, c)
    for k, v in extra.items():
        got = getattr(c, k)
        if isinstance(v, dict):
            for kk, vv in v.items():
                assert got.get(kk) == vv, (utter, kk, got)
        else:
            assert got == v, (utter, k, got)


def test_parse_grade_only():
    assert parse_grade_only("six") == 6
    assert parse_grade_only("6") == 6
    assert parse_grade_only("class 6") == 6
    assert parse_grade_only("6th") == 6
    assert parse_grade_only("chhe") == 6
    assert parse_grade_only("the heart") is None
    assert parse_grade_only("six chambers") is None


def test_bare_number_answers_the_class_question(runner, speaker):
    runner.start()
    runner.barge_in("English")
    runner.barge_in("the heart")
    assert "which class" in speaker.lines[-1].text
    runner.barge_in("six")
    st = runner.state
    assert st["topic"] == "the heart" and st["grade"] == "class 6"
    assert st["onboarding_step"] == "done"


def test_disambiguation_asks_did_you_mean(speaker):
    calls = []

    def wiki(topic, lang):
        calls.append(topic)
        if "mercury" in topic.lower():
            raise DisambiguationError("Mercury", ["Mercury (planet)", "Mercury (element)", "Mercury (mythology)"])
        return (fixture_sections(), "en", "fixture://heart")

    r = TutorRunner(make_deps(speaker=speaker, wiki_fetch=wiki), "dis")
    r.start()
    r.barge_in("English")
    r.barge_in("mercury for class six")
    last = speaker.lines[-1].text
    assert "Did you mean" in last and "Mercury (planet)" in last and " or Mercury (mythology)" in last
    assert r.state["onboarding_step"] == "source" and r.state["topic"] is None
    r.barge_in("the heart")                     # grade already known from the first answer
    assert r.state["onboarding_step"] == "done" and r.state["grade"] == "class 6"


def test_disambiguation_options_parsing():
    extract = ("Mercury may refer to:\n\n== Science ==\nMercury (planet), the first planet from the Sun\n"
               "Mercury (element), a chemical element\n== Mythology ==\nMercury (mythology), a Roman god\n")
    assert disambiguation_options(extract) == ["Mercury (planet)", "Mercury (element)", "Mercury (mythology)"]


# ------------------------------------------------------ step 2: writing for the ear
@pytest.mark.parametrize("raw,spoken", [
    ("Plants convert 3–6% of light.[12]", "Plants convert 3 to 6 percent of light."),
    ("Adenosine triphosphate (ATP) stores energy.", "Adenosine triphosphate, ATP, stores energy."),
    ("Photosynthesis (/ˌfoʊtəˈsɪnθəsɪs/) is a process.", "Photosynthesis is a process."),
    ("Water boils at 100°C at sea level.", "Water boils at 100 degrees Celsius at sea level."),
    ("Light → chemical energy", "Light gives chemical energy."),
    ("Plants use CO2 and H2O.", "Plants use carbon dioxide and water."),
    ("This happens in leaves (mainly in the green parts of the plant, as we saw).", "This happens in leaves."),
    ("Some animals, e.g. cows, eat grass.", "Some animals, for example, cows, eat grass."),
])
def test_clean_for_speech(raw, spoken):
    assert clean_for_speech(raw) == spoken


def test_chemical_equation_is_not_spoken():
    eq = "6CO2 + 6H2O → C6H12O6 + 6O2"
    assert is_equation(eq)
    assert clean_for_speech(eq) == ""


def test_beats_are_cleaned_but_retrieval_text_is_not():
    secs = sections_from_plaintext("== A ==\nWater is H2O.[3] It covers 71% of Earth.\n")
    beats = make_beats(secs)
    joined = " ".join(b["text"] for b in beats)
    assert "[3]" not in joined and "H2O" not in joined and "71 percent" in joined
    assert "H2O" in secs[0]["text"]                       # original kept for search


def test_hindi_symbols():
    assert clean_for_speech("पानी 71% है।", "hi") == "पानी 71 प्रतिशत है।"


# -------------------------------------- step 3: lesson cap + two-stage preparation
def _many_sections(n: int) -> list[dict]:
    raw = "".join(f"== Topic {i} ==\nSentence one about topic {i}. Sentence two about topic {i}. "
                  f"Sentence three about topic {i}. Sentence four about topic {i}.\n\n" for i in range(1, n + 1))
    return sections_from_plaintext(raw)


def test_lesson_is_capped_but_search_covers_everything(speaker):
    r = TutorRunner(make_deps(speaker=speaker, wiki_fetch=lambda t, l: (_many_sections(14), "en", "u")), "cap")
    onboard(r, topic_reply="topics for class six")
    plan = r.state["lesson_plan"]
    assert len({b["section_id"] for b in plan}) <= config.MAX_SECTIONS
    assert len(plan) <= config.MAX_BEATS
    n = len(speaker.lines)
    r.barge_in("tell me about topic 14")            # not in the lesson, but in the notes
    assert "topic 14" in " ".join(l.text for l in speaker.lines[n:]).lower()


class SimplifyingLLM:
    """Returns 'SIMPLE k' per input line. `wrong_first` mis-counts on the first
    call to exercise the retry; `always_wrong` never gets it right."""
    provider = model = "stub"                            # keep _select_sections on its no-LLM path

    def __init__(self, wrong_first=False, always_wrong=False, delay=0.0):
        self.wrong_first, self.always_wrong, self.delay = wrong_first, always_wrong, delay
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        time.sleep(self.delay)
        lines = [l for l in user.splitlines() if l.strip() and l.strip()[0].isdigit()]
        n = len(lines) if not (self.always_wrong or (self.wrong_first and self.calls == 1)) else max(1, len(lines) - 1)
        return "\n".join(f"{k + 1}. SIMPLE {k + 1}" for k in range(n))


def test_localize_retries_once_then_uses_result(speaker):
    llm = SimplifyingLLM(wrong_first=True)
    events = []
    r = TutorRunner(make_deps(speaker=speaker, llm_strong=llm, on_event=lambda n, p: events.append(n)), "retry")
    onboard(r)                                            # grade set -> simplify pass runs
    assert "localize_retry" in events
    assert "SIMPLE" in speaker.lines[-1].text             # first beat uses the retried result


def test_localize_keeps_original_when_model_keeps_miscounting(speaker):
    r = TutorRunner(make_deps(speaker=speaker, llm_strong=SimplifyingLLM(always_wrong=True)), "keep")
    onboard(r)
    assert "muscular organ" in speaker.lines[-1].text    # original text, nothing lost


def test_background_preparation_swaps_in_later_sections(speaker):
    llm = SimplifyingLLM()
    r = TutorRunner(make_deps(speaker=speaker, llm_strong=llm), "bg")
    onboard(r)
    store = r.deps.stores["bg"]
    assert store["prepare_done"] is False or store.get("prepare_thread") is not None
    store["prepare_thread"].join(timeout=5)
    assert store["prepare_done"] is True
    assert store["prepared"], "later sections should have been prepared in the background"
    for _ in range(6):                                    # read into the 3rd/4th section
        r.confirm_playback()
    assert "SIMPLE" in speaker.lines[-1].text


def test_give_me_a_moment_is_said_before_preparing(runner, speaker):
    onboard(runner)
    texts = [l.text for l in speaker.lines]
    i_wait = next(i for i, t in enumerate(texts) if "Give me a moment" in t)
    i_begin = next(i for i, t in enumerate(texts) if "Let's begin" in t)
    assert i_wait < i_begin


# ------------------------------------------------------- step 4: LLM JSON intent path
def test_classify_uses_llm_json_when_rules_miss():
    class JsonLLM:
        def complete(self, system, user):
            return '{"intent": "explain", "reply_lang": null}'
    assert classify("the mitochondria thing", llm=JsonLLM()).intent == "explain"

    class GarbageLLM:
        def complete(self, system, user):
            return "sure thing!"
    assert classify("the mitochondria thing", llm=GarbageLLM()).intent == "question"   # heuristic


# ------------------------------------------------------------ step 5: web provider
def test_web_provider_selection():
    assert make_web_search("stub") is stub_search
    assert make_web_search("duckduckgo") is duckduckgo_search
    assert stub_search("anything") == []


# ------------------------------------------- step 7: gap filler + evidence writer
def test_gap_filler_speaks_during_a_slow_tool_and_not_when_stale():
    sp, clock = TextSpeaker(), TurnClock()
    gf = GapFiller(clock, deadline_ms=60)

    def slow_web(q):
        time.sleep(0.25)
        return [{"title": "Mars", "snippet": MARS_SNIPPET, "url": "u"}]

    r = TutorRunner(make_deps(speaker=sp, clock=clock, web_search=slow_web, gap_filler=gf), "gf")
    onboard(r)
    n = len(sp.lines)
    r.barge_in("How many moons does Mars have?")
    texts = [l.text for l in sp.lines[n:]]
    assert texts[0] == "Let me check that.", texts
    assert any(MARS_SNIPPET in t for t in texts)
    assert gf.fired == [r.state["turn_id"]]

    # stale: the learner interrupts during the tool call -> no filler, no answer
    sp2, clock2 = TextSpeaker(), TurnClock()
    gf2 = GapFiller(clock2, deadline_ms=60)

    def interrupted_web(q):
        clock2.bump()
        time.sleep(0.25)
        return [{"title": "Mars", "snippet": MARS_SNIPPET, "url": "u"}]

    r2 = TutorRunner(make_deps(speaker=sp2, clock=clock2, web_search=interrupted_web, gap_filler=gf2), "gf2")
    onboard(r2)
    n2 = len(sp2.lines)
    r2.barge_in("How many moons does Mars have?")
    assert sp2.lines[n2:] == [] and gf2.fired == []


def test_evidence_writer_records_events(tmp_path, speaker):
    path = tmp_path / "ev.csv"
    w = EvidenceWriter(path, "ev")
    r = TutorRunner(make_deps(speaker=speaker, on_event=w), "ev")
    onboard(r)
    r.barge_in("How many chambers does the heart have?")
    w.close()
    body = path.read_text(encoding="utf-8")
    assert body.startswith("timestamp,elapsed_ms,session,event,payload")
    assert ",ingest," in body and ",interrupt," in body and ",retrieve," in body
    assert w.rows >= 5


# ------------------------------------------------------------- step 8: persistence
def test_session_survives_a_restart(tmp_path):
    db = str(tmp_path / "sessions.db")
    sp1 = TextSpeaker()
    r1 = TutorRunner(make_deps(speaker=sp1), "persist", checkpointer=make_checkpointer(db))
    onboard(r1)
    beat_before = r1.state["beat_index"]

    sp2 = TextSpeaker()                                   # fresh process: new deps, empty stores
    r2 = TutorRunner(make_deps(speaker=sp2), "persist", checkpointer=make_checkpointer(db))
    r2.start()
    assert sp2.lines == [], "resume must not replay onboarding"
    assert r2.state["beat_index"] == beat_before and r2.state["onboarding_step"] == "done"
    r2.barge_in("How many chambers does the heart have?")
    assert any("four chambers" in l.text.lower() for l in sp2.lines)   # retriever rebuilt from state


def test_web_boilerplate_filtered_and_best_snippet_chosen(speaker):
    from agents.web import is_boilerplate
    assert is_boilerplate("This page was last edited on 17 August 2026, at 13:02 (UTC).")
    assert not is_boilerplate("Mars has two moons, Phobos and Deimos.")

    def web(q):
        return [{"title": "x", "snippet": "Weather on Mars is cold and dusty.", "url": "u1"},
                {"title": "y", "snippet": "Mars has two moons, Phobos and Deimos.", "url": "u2"}]
    r = TutorRunner(make_deps(speaker=speaker, web_search=web), "web")
    onboard(r)
    n = len(speaker.lines)
    r.barge_in("How many moons does Mars have?")
    assert any("Phobos" in l.text for l in speaker.lines[n:])
