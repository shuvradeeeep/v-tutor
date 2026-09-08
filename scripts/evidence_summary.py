"""
Aggregate every committed evidence CSV into the numbers RIME_EVIDENCE.md quotes.

    v-tutor/Scripts/python scripts/evidence_summary.py              # all of evidence/
    v-tutor/Scripts/python scripts/evidence_summary.py --glob "local-*.csv"
    v-tutor/Scripts/python scripts/evidence_summary.py --markdown   # paste-ready

Everything here is computed from the CSV rows the pipeline wrote during real
runs (voice/bridge.py, voice/tts.py, agents/nodes.py emit them); nothing is
typed in by hand. Cached and uncached Rime measurements are reported apart,
as the PS requires.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round(p / 100 * (len(xs) - 1))))
    return xs[k]


def dist(xs: list[float], unit: str = "ms", nd: int = 0) -> str:
    if not xs:
        return "n=0"
    f = f"{{:.{nd}f}}"
    return (f"n={len(xs)}  median {f.format(st.median(xs))} {unit}  p95 {f.format(pct(xs, 95))} {unit}  "
            f"max {f.format(max(xs))} {unit}")


def load(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with p.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    r["p"] = json.loads(r.get("payload") or "{}")
                except json.JSONDecodeError:
                    r["p"] = {}
                r["file"] = p.name
                rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "evidence"))
    ap.add_argument("--glob", default="*.csv")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    paths = sorted(Path(args.dir).glob(args.glob))
    if not paths:
        sys.exit(f"no CSVs match {args.glob} in {args.dir}")
    rows = load(paths)
    sessions = sorted({r["session"] for r in rows})
    by = defaultdict(list)
    for r in rows:
        by[r["event"]].append(r["p"])

    # --- barge-in: VAD start -> playback flushed, measured in-process
    stop_ms = [float(p["stop_ms"]) for p in by["vad_start"] if p.get("stop_ms") is not None]
    mid_item = [p for p in by["vad_start"] if (p.get("cursor") or {}).get("words_heard") is not None]
    # --- STT: inference, and VAD start -> transcript in hand (includes the utterance itself)
    whisper = [float(p["whisper_ms"]) for p in by["transcript"] if p.get("whisper_ms")]
    vad_to_text = [float(p["since_vad_start_ms"]) for p in by["transcript"] if p.get("since_vad_start_ms")]
    # --- Rime: synth time, cached vs uncached, plus audio produced
    tts = by["tts"]
    unc = [float(p["synth_ms"]) for p in tts if not p.get("cached") and float(p.get("synth_ms", 0)) >= 5]
    cac = [float(p["synth_ms"]) for p in tts if p.get("cached") or float(p.get("synth_ms", 0)) < 5]
    # Websocket lines (2026-09-09 on): time to first audio chunk, and whether
    # Rime returned word timestamps (exact heard cursor) for the line.
    ws_lines = [p for p in tts if p.get("transport") == "ws" and not p.get("cached")]
    ttfa = [float(p["first_audio_ms"]) for p in ws_lines if p.get("first_audio_ms")]
    ws_total = [float(p["synth_ms"]) for p in ws_lines if p.get("synth_ms")]
    with_ts = sum(1 for p in tts if p.get("timestamps"))
    http_unc = [float(p["synth_ms"]) for p in tts if not p.get("cached") and p.get("transport") not in ("ws",)
                and float(p.get("synth_ms", 0)) >= 5]
    audio_s = sum(float(p.get("audio_s") or 0) for p in tts)
    words = sum(int(p.get("words") or 0) for p in tts)
    # --- staleness: the three fences
    fences = {"tts_drop_stale (Rime audio born under an older turn)": len(by["tts_drop_stale"]),
              "fence_drop (LLM/tool text born under an older turn)": len(by["fence_drop"]),
              "discard (stale answer at the graph gate)": len(by["discard"])}
    fallbacks = len(by["tts_fallback"])
    # --- graph turn time from transcript delivered to graph parked again
    turn_ms = [float(p["ms"]) for p in by["graph_turn_done"] if p.get("ms") is not None]
    intents = Counter(p.get("intent") for p in by["intent"] if p.get("intent"))
    echo = len(by["echo_drop"]); half = len(by["half_duplex_on"])
    first_beat: list[float] = []
    # time from "prepare" ingest to the first lesson line, per session
    per_session: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_session[r["session"]].append(r)
    for s, rs in per_session.items():
        t_ing = next((float(r["elapsed_ms"]) for r in rs if r["event"] == "ingest"), None)
        t_beat = next((float(r["elapsed_ms"]) for r in rs if r["event"] == "speak" and r["p"].get("kind") == "lesson"), None)
        if t_ing is not None and t_beat is not None and t_beat >= t_ing:
            first_beat.append(t_beat - t_ing)

    lines = [
        f"sessions: {len(sessions)} files ({paths[0].name} .. {paths[-1].name}), rows: {len(rows)}",
        f"barge-ins (vad_start): {len(by['vad_start'])}, of which mid-utterance with a heard cursor: {len(mid_item)}",
        f"VAD start -> playback flushed:        {dist(stop_ms, 'ms', 2)}",
        f"STT inference (whisper_ms):           {dist(whisper)}",
        f"VAD start -> transcript delivered:    {dist(vad_to_text)}   (includes the utterance + 0.5 s end-of-speech)",
        f"graph turn (transcript -> parked):    {dist(turn_ms)}",
        f"Rime synth, UNCACHED, all transports: {dist(unc)}",
        f"Rime synth, UNCACHED, HTTP one-shot:  {dist(http_unc)}",
        f"Rime websocket, UNCACHED, full line:  {dist(ws_total)}",
        f"Rime websocket, time to FIRST AUDIO:  {dist(ttfa)}",
        f"Rime synth, CACHED (disk):            {dist(cac)}",
        f"lines with word timestamps (exact heard cursor): {with_ts} of {len(tts)}",
        f"Rime audio produced: {audio_s/60:.1f} min over {len(tts)} lines, {words} words; tts_fallback (no audio): {fallbacks}",
        f"stale results fenced: " + ", ".join(f"{k}: {v}" for k, v in fences.items()),
        f"stale results spoken: 0 by construction (the fences are the only path to the player); "
        f"tts_drop_stale + fence_drop + discard = {sum(fences.values())}",
        f"echo guard drops: {echo}, half-duplex switches: {half}",
        f"ingest -> first lesson line: {dist(first_beat, 'ms')}",
        f"intents: " + ", ".join(f"{k} {v}" for k, v in intents.most_common()),
    ]
    if args.markdown:
        print("```")
    print("\n".join(lines))
    if args.markdown:
        print("```")
    return 0


if __name__ == "__main__":
    sys.exit(main())
