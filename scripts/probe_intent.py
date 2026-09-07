"""
Run utterances the regex rules miss through the fast LLM's JSON intent prompt
and compare against what a human would expect. Needs LLM_FAST_* in .env.

    v-tutor/Scripts/python scripts/probe_intent.py            # the built-in set
    v-tutor/Scripts/python scripts/probe_intent.py "wait what" "so the blood goes where"

Prints one line per utterance: OK / MISS, the route taken (rules | llm |
heuristic), latency, and the classification. Ends with a summary. This is how
AGENT_GAPS step 4 ("adjust the prompt, not the rules") is measured.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))

from agents import intent as im, llm  # noqa: E402

# (utterance, expected intent, expected extras). Every one of these returns None
# from classify_rules, so the LLM (or the heuristic) has to decide.
ODD: list[tuple[str, str, dict]] = [
    ("wait what", "explain", {}),
    ("huh", "explain", {}),
    ("what", "explain", {}),
    ("I don't get it", "explain", {}),
    ("that went over my head", "explain", {}),
    ("sorry I wasn't listening", "command", {"command": "repeat"}),
    ("I missed that", "command", {"command": "repeat"}),
    ("spell that", "explain", {}),
    ("how do you pronounce that", "explain", {}),
    ("the ventricle thing", "explain", {}),
    ("in hindi", "explain", {"reply_lang": "hi"}),
    ("hindi mein", "explain", {"reply_lang": "hi"}),
    ("I think plants eat sunlight", "question", {}),
    ("is that the same as the thing before", "question", {}),
    ("so the blood goes where", "question", {}),
    ("that's wrong isn't it", "question", {}),
    ("you said 72 but I heard 70", "question", {}),
    ("my teacher said something different", "question", {}),
    ("tell me more about harvey", "question", {}),
    ("why though", "question", {}),
    ("no I meant the other one", "question", {}),
    ("so basically the heart is a pump", "question", {}),
    ("can you go over the part with the valves", "navigate", {"nav_kind": "topic"}),
    ("let's do the history bit", "navigate", {"nav_kind": "topic"}),
    ("I already know this part", "navigate", {"nav_kind": "next"}),
    ("let's take a break", "session", {"session_cmd": "pause"}),
    ("I'm back", "session", {"session_cmd": "continue"}),
    ("we're done here", "session", {"session_cmd": "quit"}),
    ("let's finish for now", "session", {"session_cmd": "quit"}),
    ("um okay so", "backchannel", {}),
    ("this is boring", "unknown", {}),
    ("I have a question", "unknown", {}),
]


def check(c: im.Classification, want: str, extra: dict) -> bool:
    if c.intent != want:
        return False
    for k, v in extra.items():
        if k == "nav_kind":
            if not c.nav_target or c.nav_target.get("kind") != v:
                return False
        elif getattr(c, k) != v:
            return False
    return True


def main() -> None:
    fast = llm.fast()
    print(f"fast LLM: {fast.provider}/{fast.model}\n")
    items = [(u, None, {}) for u in sys.argv[1:]] or ODD
    ok = miss = 0
    lat: list[float] = []
    for utter, want, extra in items:
        route = "rules" if im.classify_rules(utter) else "llm"
        t = time.perf_counter()
        c = im.classify(utter, llm=fast)
        dt = time.perf_counter() - t
        if route == "llm":
            lat.append(dt)
        raw = c.raw if hasattr(c, "raw") else ""
        if want is None:
            print(f"      {route:5} {dt:5.2f}s  {utter!r:45} -> {c}")
            continue
        good = check(c, want, extra)
        ok += good
        miss += not good
        mark = "OK  " if good else "MISS"
        print(f"{mark}  {route:5} {dt:5.2f}s  {utter!r:45} -> {c.intent}"
              f"{' ' + str({k: v for k, v in c.as_updates().items() if v and k != 'intent'}) if not good else ''}"
              f"{'   (wanted ' + want + ' ' + str(extra) + ')' if not good else ''}")
    if want is not None:
        print(f"\n{ok} ok, {miss} miss out of {ok + miss}")
    if lat:
        lat.sort()
        print(f"llm latency: median {lat[len(lat) // 2]:.2f}s  p90 {lat[int(len(lat) * 0.9)]:.2f}s  max {lat[-1]:.2f}s")


if __name__ == "__main__":
    main()
