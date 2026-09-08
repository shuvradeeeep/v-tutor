"""
Controlled-delivery evidence for Rime coda: which way does `timeScaleFactor`
go, and does the documented `audio/L16` Accept header return the same PCM as
`audio/pcm`? Renders one sentence at several factors, measures the audio, and
writes the clips + a CSV so the PS's "render at least two variants, save the
clips" requirement is met with committed artefacts.

    v-tutor/Scripts/python scripts/speed_check.py
    -> evidence/speed/<speaker>-tsf-<factor>.wav, evidence/speed/speed-<ts>.csv
"""
from __future__ import annotations

import csv
import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

import requests  # noqa: E402

TEXT = "The heart has four chambers. Two upper atria, and two lower ventricles."
FACTORS = (0.7, 1.0, 1.3)


def render(text: str, speaker: str, lang: str, *, tsf: float | None = None, speed_alpha: float | None = None,
           accept: str = "audio/pcm") -> tuple[bytes, float]:
    body = {"speaker": speaker, "text": text, "modelId": config.RIME_MODEL_ID, "lang": lang,
            "samplingRate": config.RIME_SAMPLE_RATE}
    if tsf is not None:
        body["timeScaleFactor"] = tsf
    if speed_alpha is not None:
        body["speedAlpha"] = speed_alpha
    t = time.perf_counter()
    r = requests.post(config.RIME_HTTP_ENDPOINT, json=body, timeout=30,
                      headers={"Authorization": f"Bearer {os.getenv('RIME_API_KEY')}", "Accept": accept,
                               "Content-Type": "application/json"})
    r.raise_for_status()
    return r.content, (time.perf_counter() - t) * 1000


def main() -> int:
    out = ROOT / "evidence" / "speed"
    out.mkdir(parents=True, exist_ok=True)
    speaker, lang = config.LANG_SPEAKER["en"], "en"
    rows = []
    print(f"speaker={speaker} model={config.RIME_MODEL_ID} rate={config.RIME_SAMPLE_RATE}\n")
    print(f"{'variant':28} {'bytes':>8} {'audio s':>8} {'ms':>6}")
    for f in FACTORS:
        pcm, ms = render(TEXT, speaker, lang, tsf=f)
        secs = len(pcm) / (config.RIME_SAMPLE_RATE * 2)
        p = out / f"{speaker}-tsf-{f}.wav"
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(config.RIME_SAMPLE_RATE); w.writeframes(pcm)
        rows.append({"variant": f"timeScaleFactor={f}", "bytes": len(pcm), "audio_s": round(secs, 3),
                     "http_ms": round(ms), "clip": p.name})
        print(f"{'timeScaleFactor=' + str(f):28} {len(pcm):8d} {secs:8.2f} {ms:6.0f}")
    for a in (0.7, 1.3):
        pcm, ms = render(TEXT, speaker, lang, speed_alpha=a)
        secs = len(pcm) / (config.RIME_SAMPLE_RATE * 2)
        rows.append({"variant": f"speedAlpha={a}", "bytes": len(pcm), "audio_s": round(secs, 3), "http_ms": round(ms), "clip": ""})
        print(f"{'speedAlpha=' + str(a):28} {len(pcm):8d} {secs:8.2f} {ms:6.0f}")
    pcm, ms = render(TEXT, speaker, lang, accept="audio/L16")
    secs = len(pcm) / (config.RIME_SAMPLE_RATE * 2)
    rows.append({"variant": "Accept: audio/L16", "bytes": len(pcm), "audio_s": round(secs, 3), "http_ms": round(ms), "clip": ""})
    print(f"{'Accept: audio/L16':28} {len(pcm):8d} {secs:8.2f} {ms:6.0f}")
    pcm, ms = render(TEXT, speaker, "eng")
    rows.append({"variant": "lang=eng (3-letter)", "bytes": len(pcm), "audio_s": round(len(pcm) / 32000, 3), "http_ms": round(ms), "clip": ""})
    print(f"{'lang=eng (3-letter)':28} {len(pcm):8d} {len(pcm)/32000:8.2f} {ms:6.0f}")

    base = next(r["audio_s"] for r in rows if r["variant"] == "timeScaleFactor=1.0")
    slow = next(r["audio_s"] for r in rows if r["variant"] == "timeScaleFactor=1.3")
    verdict = "HIGHER timeScaleFactor = SLOWER (longer clip)" if slow > base * 1.1 else \
              "HIGHER timeScaleFactor = FASTER (shorter clip)" if slow < base * 0.9 else "no measurable effect"
    print(f"\nverdict: {verdict}")
    csv_path = out / f"speed-{int(time.time())}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "bytes", "audio_s", "http_ms", "clip"])
        w.writeheader(); w.writerows(rows)
        fh.write(f"# text: {TEXT}\n# verdict: {verdict}\n")
    print(f"clips + csv: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
