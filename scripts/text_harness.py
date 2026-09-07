"""
Type what the student says; see what the tutor would speak.

    v-tutor/Scripts/python scripts/text_harness.py                 # topic mode, offline fixture
    v-tutor/Scripts/python scripts/text_harness.py --live          # real Wikipedia fetch
    v-tutor/Scripts/python scripts/text_harness.py --pdf notes.pdf
    v-tutor/Scripts/python scripts/text_harness.py --stress-ms 3000

At the prompt:
    <Enter>            the tutor finished speaking (playback confirmed)
    hello there        interrupt after hearing everything
    @4 what's that     interrupt after hearing 4 words
    /state             dump key state fields
    /quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _stream in (sys.stdout, sys.stderr):        # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import config                                             # noqa: E402
from agents import llm                                    # noqa: E402
from agents.evidence import EvidenceWriter                # noqa: E402
from agents.gap_filler import GapFiller                   # noqa: E402
from agents.graph import TutorRunner, make_checkpointer   # noqa: E402
from agents.material import fetch_wikipedia, parse_pdf    # noqa: E402
from agents.retrieval import make_embedder                # noqa: E402
from agents.session import Deps, TextSpeaker, TurnClock   # noqa: E402
from agents.web import make_web_search                    # noqa: E402

SHOW = {"fence_drop", "discard", "web_aborted", "interrupt", "navigate", "retrieve", "intent",
        "filler", "localize_retry", "localized_bg", "ingest"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", nargs="*", default=None)
    ap.add_argument("--live", action="store_true", help="real Wikipedia instead of the test fixture")
    ap.add_argument("--stress-ms", type=int, default=config.STRESS_DELAY_MS)
    ap.add_argument("--embedder", default=config.EMBEDDER)
    ap.add_argument("--quiet", action="store_true", help="hide internal events")
    ap.add_argument("--web", default=config.WEB_SEARCH_PROVIDER, help="stub | duckduckgo")
    ap.add_argument("--persist", default=None, metavar="DB", help="SQLite file; rerun with --session to resume")
    ap.add_argument("--session", default="harness")
    ap.add_argument("--evidence", default=None, metavar="CSV", help="append every event to this CSV")
    ap.add_argument("--no-filler", action="store_true", help="disable the dead-air filler")
    args = ap.parse_args()

    if args.live or args.pdf:
        wiki, web = fetch_wikipedia, make_web_search(args.web)
    else:
        from conftest import fake_web, fake_wiki
        wiki, web = fake_wiki, (make_web_search(args.web) if args.web != "stub" else fake_web)

    evidence = EvidenceWriter(args.evidence, args.session) if args.evidence else None

    def on_event(name: str, p: dict) -> None:
        if evidence:
            evidence(name, p)
        if not args.quiet and name in SHOW:
            print(f"     . {name} {p}")

    clock = TurnClock()
    deps = Deps(
        clock=clock, speaker=TextSpeaker(echo=True),
        llm_fast=llm.fast(), llm_strong=llm.strong(),
        embedder=make_embedder(args.embedder),
        web_search=web, wiki_fetch=wiki, pdf_parse=parse_pdf,
        stress_delay_ms=args.stress_ms, on_event=on_event,
        gap_filler=None if args.no_filler else GapFiller(clock, config.GAP_FILLER_DEADLINE_MS),
    )
    print(f"LLM fast={deps.llm_fast.provider} strong={deps.llm_strong.provider} | "
          f"embedder={args.embedder} | stress_delay={args.stress_ms}ms | "
          f"rime model={config.RIME_MODEL_ID} voices={config.LANG_SPEAKER}\n")

    r = TutorRunner(deps, args.session, pdf_paths=args.pdf, checkpointer=make_checkpointer(args.persist))
    r.start()
    while not r.finished:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line == "/quit":
            break
        if line == "/state":
            st = r.state
            for k in ("onboarding_step", "active_lang", "speaker", "beat_index", "beat_spoken", "paused",
                      "turn_id", "born_turn_id", "heard_cursor", "intent", "speed_alpha", "stale_drops"):
                print(f"  {k} = {st.get(k)!r}")
            continue
        if not line:
            r.confirm_playback()
            continue
        words = None
        if line.startswith("@"):
            head, _, rest = line.partition(" ")
            if head[1:].isdigit():
                words, line = int(head[1:]), rest
        r.barge_in(line, words_heard=words)
    print("session over.")


if __name__ == "__main__":
    main()
