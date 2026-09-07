"""
THE test. The PS full-duplex recipe: a slow tool call, an interruption while it
runs, and proof that the stale result is never spoken as current.

No API keys, no audio, no network. Judges can run this in under a second.
"""
from __future__ import annotations

import threading
import time

from conftest import MARS_SNIPPET, TutorRunner, TextSpeaker, TurnClock, make_deps, onboard


def spoken_after(speaker: TextSpeaker, n: int) -> str:
    return "\n".join(l.text for l in speaker.lines[n:])


def test_stale_web_result_is_never_spoken():
    sp, clock = TextSpeaker(), TurnClock()

    def web_that_gets_interrupted(query: str):
        # The learner speaks WHILE the tool is running: the VAD callback bumps
        # the live turn. This is exactly what happens in production.
        clock.bump()
        return [{"title": "Mars", "snippet": MARS_SNIPPET, "url": "u"}]

    r = TutorRunner(make_deps(speaker=sp, clock=clock, web_search=web_that_gets_interrupted), "fence")
    onboard(r)
    n0 = len(sp.lines)

    r.barge_in("How many moons does Mars have?")            # not in the notes -> web
    st = r.state
    assert MARS_SNIPPET not in spoken_after(sp, n0), "stale tool result was spoken"
    assert st["stale_drops"] >= 1
    assert st["answer"] is None
    assert not r.finished, "graph must be parked, ready for the learner's new utterance"

    # The utterance that caused the interruption is now delivered and answered normally.
    n1 = len(sp.lines)
    r.barge_in("How many chambers does the heart have?")
    assert "four chambers" in spoken_after(sp, n1).lower()
    assert MARS_SNIPPET not in spoken_after(sp, n0)


def test_control_web_result_is_spoken_when_not_interrupted():
    sp = TextSpeaker()
    r = TutorRunner(make_deps(speaker=sp), "ctrl")
    onboard(r)
    n0 = len(sp.lines)
    r.barge_in("How many moons does Mars have?")
    assert MARS_SNIPPET in spoken_after(sp, n0)
    assert r.state["stale_drops"] == 0


def test_stress_delay_aborts_slow_tool_when_turn_moves_on():
    """With the PS's injected delay, the tool call itself bails out early once
    the live turn changes, instead of finishing useless work."""
    sp, clock = TextSpeaker(), TurnClock()
    calls: list[str] = []

    def web(query: str):
        calls.append(query)
        return [{"title": "Mars", "snippet": MARS_SNIPPET, "url": "u"}]

    r = TutorRunner(make_deps(speaker=sp, clock=clock, web_search=web, stress_delay_ms=400), "stress")
    onboard(r)

    def interrupt_mid_delay():
        time.sleep(0.08)
        clock.bump()                    # VAD fires during the injected delay
    threading.Thread(target=interrupt_mid_delay).start()

    n0 = len(sp.lines)
    r.barge_in("How many moons does Mars have?")
    assert calls == [], "web search should have been aborted before running"
    assert r.state["web_aborted"] is True
    assert MARS_SNIPPET not in spoken_after(sp, n0)
    assert r.state["stale_drops"] >= 1


def test_stale_explain_is_discarded_too():
    """Fencing is structural: every speaking branch passes fence_check, not just
    the web path. Here the strong LLM call is where the interruption lands."""
    sp, clock = TextSpeaker(), TurnClock()

    class InterruptingLLM:
        provider = model = "test"
        def complete(self, system, user):
            if "student just heard" not in user:      # ingest's simplify pass: stay quiet
                return ""
            clock.bump()                              # the learner speaks mid-LLM-call
            return "Ventricles are the lower chambers."

    r = TutorRunner(make_deps(speaker=sp, clock=clock, llm_strong=InterruptingLLM()), "expl")
    onboard(r)
    n0 = len(sp.lines)
    r.barge_in("what does that mean", words_heard=12)
    assert "Ventricles" not in spoken_after(sp, n0)
    assert spoken_after(sp, n0) == ""            # nothing at all was said on the stale turn
    assert r.state["stale_drops"] == 1
    assert not r.finished
