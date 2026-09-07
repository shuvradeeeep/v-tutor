"""
Non-interactive end-to-end run with REAL services: Wikipedia, the configured
LLMs, DuckDuckGo. Builds a lesson, waits for background preparation, then asks
a few questions. Prints what the tutor would speak plus a summary of how the
model behaved (section pick, line-count retries, background sections).

    v-tutor/Scripts/python scripts/live_check.py                       # Photosynthesis, class 6
    v-tutor/Scripts/python scripts/live_check.py --topic "the heart" --grade 7 \
        --ask "what is the aorta" --ask "how many moons does mars have"
    v-tutor/Scripts/python scripts/live_check.py --pdf notes.pdf

This is the measurement AGENT_GAPS step 3 asks for ("how often does the model
return the wrong line count") and step 4 ("run the harness against real models").
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))

import config                                              # noqa: E402
from agents import llm                                     # noqa: E402
from agents.gap_filler import GapFiller                    # noqa: E402
from agents.graph import TutorRunner                       # noqa: E402
from agents.material import fetch_wikipedia, parse_pdf     # noqa: E402
from agents.retrieval import make_embedder                 # noqa: E402
from agents.session import Deps, TextSpeaker, TurnClock    # noqa: E402
from agents.web import make_web_search                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="photosynthesis")
    ap.add_argument("--grade", default="6")
    ap.add_argument("--pdf", nargs="*", default=None)
    ap.add_argument("--ask", action="append", default=[], help="question to ask after the first beat (repeatable)")
    ap.add_argument("--beats", type=int, default=3, help="how many beats to read before asking")
    ap.add_argument("--web", default=config.WEB_SEARCH_PROVIDER)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    events: list[tuple[str, dict]] = []
    t0 = time.perf_counter()

    def on_event(name: str, p: dict) -> None:
        events.append((name, p))
        if not args.quiet and name in {"ingest", "localize_retry", "localized_bg", "retrieve", "intent",
                                       "web_search", "fence_drop", "filler", "answer_mode"}:
            print(f"     . {time.perf_counter() - t0:6.2f}s {name} {p}")

    clock = TurnClock()
    speaker = TextSpeaker(echo=not args.quiet)
    deps = Deps(clock=clock, speaker=speaker, llm_fast=llm.fast(), llm_strong=llm.strong(),
                embedder=make_embedder(config.EMBEDDER), web_search=make_web_search(args.web),
                wiki_fetch=fetch_wikipedia, pdf_parse=parse_pdf, on_event=on_event,
                gap_filler=GapFiller(clock, config.GAP_FILLER_DEADLINE_MS))
    print(f"fast={deps.llm_fast.provider}/{deps.llm_fast.model}  strong={deps.llm_strong.provider}/"
          f"{deps.llm_strong.model}  web={args.web}  embedder={config.EMBEDDER}\n")

    r = TutorRunner(deps, f"live-{int(time.time())}", pdf_paths=args.pdf)
    r.start()
    r.barge_in("English")
    t = time.perf_counter()
    r.barge_in(f"{args.topic} for class {args.grade}" if not args.pdf else f"class {args.grade}")
    t_first_beat = time.perf_counter() - t
    st = r.state
    if st.get("onboarding_step") != "done":
        print(f"\nonboarding did not finish: last line = {speaker.lines[-1].text!r}")
        return

    store = deps.stores[r.session_id]
    th = store.get("prepare_thread")
    if th is not None:
        t = time.perf_counter()
        th.join(timeout=120)
        t_bg = time.perf_counter() - t
    else:
        t_bg = 0.0

    for _ in range(max(0, args.beats - 1)):
        r.confirm_playback()
    for q in args.ask:
        n = len(speaker.lines)
        t = time.perf_counter()
        r.barge_in(q, words_heard=6)
        print(f"     -> answered in {time.perf_counter() - t:.2f}s, {len(speaker.lines) - n} lines spoken")

    # ---- summary -----------------------------------------------------------
    names = Counter(n for n, _ in events)
    ingest = next((p for n, p in events if n == "ingest"), {})
    bg = [p for n, p in events if n == "localized_bg"]
    retries = [p for n, p in events if n == "localize_retry"]
    plan = st["lesson_plan"]
    print("\n==== summary ====")
    print(f"source: {st.get('source_title')}  {st.get('source_url') or ''}")
    print(f"sections fetched {ingest.get('sections')} -> lesson {ingest.get('lesson_sections')} "
          f"-> beats {ingest.get('beats')} (upfront {ingest.get('upfront')} sections, background {ingest.get('background')})")
    print(f"lesson sections: {[b['section_title'] for b in plan if b['section_title'] not in [x['section_title'] for x in plan[:plan.index(b)]]]}")
    print(f"time to first beat: {t_first_beat:.1f}s   background prep: {t_bg:.1f}s")
    print(f"localize calls: {sum(1 for _ in bg) + ingest.get('upfront', 0)}  line-count retries: {len(retries)}  "
          f"background sections ok: {sum(1 for p in bg if p['ok'])}/{len(bg)}")
    if retries:
        print(f"  retry details: {retries}")
    print(f"events: {dict(names)}")
    print(f"stale drops: {st.get('stale_drops')}   fence drops: {names.get('fence_drop', 0)}")


if __name__ == "__main__":
    main()
