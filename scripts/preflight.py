"""
Rime configuration and secret preflight.

    v-tutor/Scripts/python scripts/preflight.py            # everything
    v-tutor/Scripts/python scripts/preflight.py --offline  # no network: config + secret hygiene only

Checks, in order:
  1. The pinned model/voice/language combinations exist in Rime's LIVE catalog
     (https://users.rime.ai/data/voices/all-v2.json), not a copied list.
  2. One real synthesis on the exact shipped path: endpoint, model, speaker,
     language code, PCM format, sample rate. Verifies the body is s16le PCM of
     a plausible length for the text.
  3. Secret hygiene: .env is ignored by git, .env.example holds placeholders
     only, and no tracked file contains something that looks like a live key.

Exit code 0 = pass. Anything else prints what failed and why.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402  (loads .env)

OK, FAIL, WARN = "PASS", "FAIL", "WARN"
_results: list[tuple[str, str, str]] = []


def rec(status: str, check: str, detail: str = "") -> None:
    _results.append((status, check, detail))
    print(f"  [{status}] {check}{(' -- ' + detail) if detail else ''}")


# ---------------------------------------------------------------- 1. catalog
def check_catalog() -> dict | None:
    import requests
    try:
        r = requests.get(config.RIME_CATALOG_URL, timeout=15)
        r.raise_for_status()
        cat = r.json()
    except Exception as exc:  # noqa: BLE001
        rec(FAIL, "fetch live catalog", str(exc))
        return None
    rec(OK, "fetch live catalog", f"{config.RIME_CATALOG_URL} ({len(cat)} models)")
    model = config.RIME_MODEL_ID
    if model not in cat:
        rec(FAIL, f"model '{model}' in catalog", f"available: {', '.join(cat)}")
        return cat
    rec(OK, f"model '{model}' in catalog", f"languages: {', '.join(cat[model])}")
    for lang in config.SUPPORTED_LANGS:                         # the judged flow
        _check_voice(cat, model, lang, required=True)
    for lang in config.REPLY_LANGS:                             # one-off "explain in X"
        if lang not in config.SUPPORTED_LANGS:
            _check_voice(cat, model, lang, required=False)
    return cat


def _check_voice(cat: dict, model: str, lang: str, required: bool) -> None:
    ckey = config.LANG_TO_CATALOG.get(lang)
    speaker = config.LANG_SPEAKER.get(lang)
    label = f"voice '{speaker}' for {lang} ({ckey}) on {model}"
    if not ckey or ckey not in cat[model]:
        rec(FAIL if required else WARN, label, f"language key '{ckey}' not offered by {model}")
        return
    voices = cat[model][ckey]
    if speaker in voices:
        rec(OK, label, f"{len(voices)} voices for {ckey}")
    else:
        rec(FAIL if required else WARN, label, f"not in catalog; first few: {', '.join(voices[:5])}")


# ---------------------------------------------------------------- 2. synthesis
def check_synthesis(sample_text: str = "Preflight check. One, two, three.") -> None:
    key = os.getenv("RIME_API_KEY")
    if not key:
        rec(FAIL, "RIME_API_KEY present", "unset; the judged flow cannot speak")
        return
    from voice.tts import RimeHTTP
    synth = RimeHTTP(cache_dir=None)                # never a cache hit: this must hit the endpoint
    speaker = config.LANG_SPEAKER[config.SUPPORTED_LANGS[0]]
    t = time.perf_counter()
    pcm = synth.synth(sample_text, speaker=speaker, lang=config.SUPPORTED_LANGS[0])
    ms = (time.perf_counter() - t) * 1000
    label = (f"synthesize on shipped path: {config.RIME_HTTP_ENDPOINT} model={config.RIME_MODEL_ID} "
             f"speaker={speaker} lang={config.SUPPORTED_LANGS[0]} "
             f"format={config.RIME_AUDIO_FORMAT} rate={config.RIME_SAMPLE_RATE}")
    if not pcm:
        rec(FAIL, label, "no audio returned (see log: HTTP status / auth)")
        return
    seconds = len(pcm) / (config.RIME_SAMPLE_RATE * 2)
    words = len(sample_text.split())
    # s16le sanity: even byte count, and 6 words should be roughly 1-6 s of audio.
    if len(pcm) % 2 or not (0.15 * words <= seconds <= 1.0 * words):
        rec(FAIL, label, f"{len(pcm)} bytes = {seconds:.2f}s for {words} words: not s16le at {config.RIME_SAMPLE_RATE} Hz?")
        return
    import numpy as np
    peak = int(np.abs(np.frombuffer(pcm, dtype="<i2")).max())
    if peak < 500:
        rec(FAIL, label, f"audio is near-silent (peak {peak})")
        return
    rec(OK, label, f"{seconds:.2f}s of PCM in {ms:.0f} ms (uncached), peak {peak}")

    # The shipped transport is the websocket; HTTP above is its fallback.
    from voice.tts import RimeWS
    ws = RimeWS(cache_dir=None)
    try:
        t = time.perf_counter()
        pcm2 = ws.synth(sample_text, speaker=speaker, lang=config.SUPPORTED_LANGS[0])
        ms2 = (time.perf_counter() - t) * 1000
        label2 = f"synthesize over websocket: {config.RIME_WS_ENDPOINT} (segment=immediate, contextId, timestamps)"
        if not pcm2 or ws.ws_calls == 0:
            rec(FAIL, label2, f"no audio over the socket (failures {ws.ws_failures}, http fallbacks {ws.http_fallbacks})")
        else:
            n_words = len(ws.last_word_ends or [])
            rec(OK if n_words else WARN, label2,
                f"{len(pcm2) / (config.RIME_SAMPLE_RATE * 2):.2f}s in {ms2:.0f} ms, first audio "
                f"{ws.last_first_audio_ms:.0f} ms, word timestamps: {n_words} words"
                + ("" if n_words else " (none: timestamps are English/Spanish only)"))
    finally:
        ws.close()


# ---------------------------------------------------------------- 3. secrets
_SECRET_RX = re.compile(
    r"(?:gsk_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|"
    r"API[_-]?KEY\s*[=:]\s*['\"]?[A-Za-z0-9_-]{24,}|SECRET\s*[=:]\s*['\"]?[A-Za-z0-9_-]{24,})")


def check_secrets() -> None:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout
        except Exception:  # noqa: BLE001
            return ""
    tracked = git("ls-files").split()
    if not tracked:
        rec(WARN, "git tracked files", "not a git checkout or git unavailable; skipped")
        return
    if ".env" in tracked:
        rec(FAIL, ".env not tracked by git", ".env IS tracked -- remove it and rotate the keys")
    else:
        ignored = git("check-ignore", ".env").strip()
        rec(OK if ignored else WARN, ".env ignored by git", ignored or ".env not listed in .gitignore")
    ex = ROOT / ".env.example"
    if ex.exists():
        bad = [l.strip() for l in ex.read_text(encoding="utf-8", errors="replace").splitlines()
               if _SECRET_RX.search(l)]
        rec(FAIL if bad else OK, ".env.example holds placeholders only", "; ".join(bad)[:120])
    else:
        rec(FAIL, ".env.example exists", "missing")
    hits: list[str] = []
    for f in tracked:
        p = ROOT / f
        if not p.is_file() or p.suffix.lower() in {".pcm", ".png", ".jpg", ".wav", ".db", ".pdf"} or p.stat().st_size > 2_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in _SECRET_RX.finditer(text):
            frag = m.group(0)
            if "your_" in frag or "placeholder" in frag.lower() or "example" in f.lower():
                continue
            hits.append(f"{f}: {frag[:12]}...")
    rec(FAIL if hits else OK, "no live-looking secrets in tracked files", "; ".join(hits[:5]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip catalog and synthesis (no network)")
    args = ap.parse_args()
    print(f"\nRime preflight -- model={config.RIME_MODEL_ID} speakers={dict((k, config.LANG_SPEAKER[k]) for k in config.SUPPORTED_LANGS)} "
          f"format={config.RIME_AUDIO_FORMAT}@{config.RIME_SAMPLE_RATE}Hz\n  transport: {config.RIME_TRANSPORT}\n")
    if not args.offline:
        check_catalog()
        check_synthesis()
    check_secrets()
    fails = [c for s, c, _ in _results if s == FAIL]
    warns = [c for s, c, _ in _results if s == WARN]
    print(f"\n{'PREFLIGHT FAILED' if fails else 'PREFLIGHT PASSED'}: {len(_results) - len(fails) - len(warns)} pass, "
          f"{len(warns)} warn, {len(fails)} fail")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
