"""
audio_check.py
Which speaker is the tutor talking to, and can you hear it?

The pipeline can be perfectly healthy and silent: on this laptop Windows
routes to the headphone jack by default, so with nothing plugged in (or that
device muted) every log line looks right and no sound arrives.

    python scripts/audio_check.py                 # list devices, no sound
    python scripts/audio_check.py --play          # beep on the default output
    python scripts/audio_check.py --play --all    # beep on every output in turn
    python scripts/audio_check.py --play --device 5
    python scripts/audio_check.py --say "hello"   # speak a line through Rime

Whichever device you hear, pass it to the tutor:
    python main.py local --lang en --output-device N
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLE_RATE = 16000


def tone(seconds: float = 0.7, hz: float = 440.0) -> bytes:
    """A quiet sine wave as 16 kHz mono int16 -- the tutor's audio format."""
    n = int(SAMPLE_RATE * seconds)
    return b"".join(
        struct.pack("<h", int(9000 * math.sin(2 * math.pi * hz * i / SAMPLE_RATE)))
        for i in range(n)
    )


def outputs(sd) -> list[tuple[int, str]]:
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
            if d["max_output_channels"] > 0 and d["hostapi"] == 0]


def play(sd, pcm: bytes, device: int | None) -> None:
    import numpy as np
    audio = np.frombuffer(pcm, dtype=np.int16)
    sd.play(audio, samplerate=SAMPLE_RATE, device=device, blocking=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true", help="make a sound (silent otherwise)")
    ap.add_argument("--all", action="store_true", help="try every output device in turn")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--say", default=None, help="speak this line with the tutor's own voice")
    args = ap.parse_args()

    import sounddevice as sd
    default_out = sd.default.device[1]
    print(f"default output: [{default_out}] {sd.query_devices(default_out)['name']}")
    print("outputs:")
    for i, name in outputs(sd):
        print(f"  {i:3d}  {name}{'   <== default' if i == default_out else ''}")

    if args.say:
        import config  # noqa: F401  loads .env
        from voice.tts import default_synth
        synth = default_synth()
        pcm = synth.synth(args.say, speaker=config.LANG_SPEAKER["en"], lang="en")
        if pcm is None:
            sys.exit("no speech provider available (set RIME_API_KEY, or TTS_PROVIDER=sapi)")
        print(f"\nspeaking {len(pcm) / (SAMPLE_RATE * 2):.1f}s via {synth.model_id} ...")
        play(sd, pcm, args.device)
        return

    if not args.play:
        print("\n(no sound played; add --play, or --play --all to find the one you can hear)")
        return

    pcm = tone()
    targets = [i for i, _ in outputs(sd)] if args.all else [args.device]
    for dev in targets:
        label = sd.query_devices(dev)["name"] if dev is not None else "default"
        print(f"\nbeep -> [{dev}] {label}")
        try:
            play(sd, pcm, dev)
        except Exception as exc:  # noqa: BLE001
            print(f"   failed: {exc}")


if __name__ == "__main__":
    main()
