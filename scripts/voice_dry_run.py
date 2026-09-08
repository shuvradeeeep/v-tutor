"""
End-to-end check of stt -> agents -> tts WITHOUT a microphone.

Each scripted learner utterance is rendered to speech (Rime, cached), pushed
through the real Silero VAD and real Whisper exactly as microphone audio would
be, and whatever the tutor says comes back through the real graph and Rime.
By default the tutor's audio is not played (a ScriptedPlayer swallows it, so
the run is quiet and fast); pass --speakers to hear it.

    v-tutor/Scripts/python scripts/voice_dry_run.py
    v-tutor/Scripts/python scripts/voice_dry_run.py --speakers --say "English" \
        --say "the heart for class six" --say "how many chambers does the heart have" --say "stop for today"

Prints every transcript with Whisper latency, every tutor line, and a summary,
and appends each transcript to stt/transcripts.csv exactly as a live session
does.

Needs: RIME_API_KEY to render the learner's lines -- or `--sapi` to render them
with the Windows speech synthesiser instead, which needs no key and no network,
so the whole mic -> VAD -> Whisper -> graph -> tutor chain can be checked
offline. LLM keys are optional either way (without them the graph takes its
deterministic fallback paths).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))

import config                                              # noqa: E402
from voice.audio_io import SpeechInput                     # noqa: E402
from voice.bridge import make_bridge                       # noqa: E402
from voice.player import SAMPLE_RATE, TimedPlayer          # noqa: E402
from voice.tts import RimeHTTP                             # noqa: E402

DEFAULT_SCRIPT = ["English", "the heart for class six", "how many chambers does the heart have",
                  "what is the capital of France", "stop for today"]
LEARNER_VOICE = "ana"         # an English coda voice other than the tutor's (config.LANG_SPEAKER)


def sapi_pcm(lines: list[str]) -> list[bytes]:
    """
    Render the learner's lines with the Windows speech synthesiser: 16 kHz mono
    PCM, the same format Rime returns, so the rest of the run is identical.
    No API key, no network -- this is the offline path.
    """
    out = Path(tempfile.gettempdir()) / "v_tutor_sapi"
    out.mkdir(exist_ok=True)
    # Keyed by the text, never by position: --say lists change between runs and
    # an index-keyed cache would silently replay the previous run's audio.
    paths = [out / f"{hashlib.sha1(t.encode('utf-8')).hexdigest()[:16]}.wav" for t in lines]
    todo = [(p, t) for p, t in zip(paths, lines) if not p.exists()]
    if todo:
        ps = [
            "Add-Type -AssemblyName System.Speech",
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
            "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
            "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,"
            "[System.Speech.AudioFormat.AudioChannel]::Mono)",
        ]
        for path, text in todo:
            ps.append(f'$s.SetOutputToWaveFile("{path}",$f); $s.Speak("{text}")')
        ps.append("$s.Dispose()")
        subprocess.run(["powershell", "-NoProfile", "-Command", "; ".join(ps)], check=True)

    clips = []
    for path in paths:
        with wave.open(str(path), "rb") as w:
            clips.append(w.readframes(w.getnframes()))
    return clips


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--say", action="append", default=None, help="learner utterance (repeatable)")
    ap.add_argument("--speakers", action="store_true", help="play the tutor through the laptop speakers")
    ap.add_argument("--speed", type=float, default=3.0,
                    help="silent mode only: how much faster than real time the tutor's audio 'plays'")
    ap.add_argument("--gap", type=float, default=1.2, help="silence after each utterance, seconds")
    ap.add_argument("--settle", type=float, default=2.0, help="extra wait after the tutor answers, seconds")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--sapi", action="store_true",
                    help="render the learner with Windows TTS instead of Rime (no key, offline)")
    args = ap.parse_args()
    script = args.say or DEFAULT_SCRIPT

    from livekit import rtc

    synth = RimeHTTP()
    use_sapi = args.sapi or not synth.api_key
    if use_sapi and not args.sapi:
        print("no RIME_API_KEY: rendering the learner with Windows TTS (--sapi)")

    if args.speakers:
        from voice.player import LocalPlayer
        player = LocalPlayer()
    else:
        player = TimedPlayer(speed=args.speed)

    spoken: list[tuple[float, str]] = []
    t0 = time.perf_counter()

    def on_text(text: str, meta: dict) -> None:
        spoken.append((time.perf_counter() - t0, text))
        print(f"\n  [{time.perf_counter() - t0:6.2f}s tutor | turn {meta.get('turn')}]  {text}\n")

    def on_event(name: str, p: dict) -> None:
        if not args.quiet and name in {"vad_start", "transcript", "intent", "retrieve", "answer_mode",
                                       "tts_drop_stale", "fence_drop", "playback_confirmed",
                                       "graph_turn_done", "onboarding_reask"}:
            print(f"     . {time.perf_counter() - t0:6.2f}s {name} {p}")

    session = f"dryrun-{int(time.time())}"
    bridge = make_bridge(player, session_id=session, evidence_path=f"{config.EVIDENCE_DIR}/{session}.csv",
                         on_text=on_text, on_event=on_event)
    speech = SpeechInput()
    print("loading Silero + Whisper ...")
    speech.load()

    # ---- render the learner's lines up front (cached after the first run) ----
    print("rendering learner audio ...")
    if use_sapi:
        clips = sapi_pcm(script)
    else:
        clips = []
        for line in script:
            pcm = synth.synth(line, speaker=LEARNER_VOICE, lang="en")
            if pcm is None:
                sys.exit(f"could not render {line!r}")
            clips.append(pcm)

    frame_bytes = SAMPLE_RATE * 2 * 20 // 1000
    queue: asyncio.Queue = asyncio.Queue()

    async def frames():
        while True:
            data = await queue.get()
            if data is None:
                return
            yield rtc.AudioFrame(data=data, sample_rate=SAMPLE_RATE, num_channels=1,
                                 samples_per_channel=len(data) // 2)

    async def feed(pcm: bytes, realtime: bool = True) -> None:
        for i in range(0, len(pcm), frame_bytes):
            chunk = pcm[i:i + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk += bytes(frame_bytes - len(chunk))
            queue.put_nowait(chunk)
            if realtime:
                await asyncio.sleep(0.02)

    async def silence(seconds: float) -> None:
        await feed(bytes(int(SAMPLE_RATE * 2 * seconds)))

    stt_task = asyncio.create_task(speech.run_frames(frames(), bridge, label="dry-run"))
    bridge.start()
    await asyncio.to_thread(bridge.wait_idle, 60)
    await silence(0.5)

    for line, pcm in zip(script, clips):
        n_turns = len(bridge.turns)
        print(f"\n>>> learner says: {line!r}")
        await feed(pcm)
        await silence(args.gap)                            # lets the VAD close the utterance
        deadline = time.perf_counter() + 30
        while len(bridge.turns) == n_turns and time.perf_counter() < deadline:
            await silence(0.1)
        await asyncio.to_thread(bridge.wait_idle, 90)       # graph done with this turn
        await silence(args.settle)
        if bridge.finished:
            break

    queue.put_nowait(None)
    try:
        await asyncio.wait_for(stt_task, timeout=10)
    except (asyncio.TimeoutError, Exception):  # noqa: BLE001
        stt_task.cancel()
    bridge.wait_idle(10)
    bridge.close()
    speech.close()
    ev = getattr(bridge, "evidence", None)
    if ev:
        ev.close()

    print("\n==== summary ====")
    for t in bridge.turns:
        print(f"  heard {t['text']!r:50} whisper {t['whisper_ms']:>5} ms   vad->text {t['since_vad_start_ms']} ms")
    print(f"tutor lines: {len(spoken)}   rime calls: {synth.calls + bridge.speaker.synth.calls} "
          f"(cache hits {synth.cache_hits + bridge.speaker.synth.cache_hits})   "
          f"stale drops: {bridge.runner.state.get('stale_drops')}   finished: {bridge.finished}")
    if ev:
        print(f"evidence: {ev.path} ({ev.rows} rows)")
    from stt import settings as stt_settings
    print(f"transcripts: {stt_settings.TRANSCRIPTS_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
