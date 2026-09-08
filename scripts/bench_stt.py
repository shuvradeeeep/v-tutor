"""
bench_stt.py
Measures Whisper latency for the STT pipeline, so tuning is driven by numbers.

It compares the settings the pipeline shipped with (beam 5, CTranslate2's
default threading, two encoder passes) against the current stt/settings.py, and
checks that the fast path returns exactly what the plain faster-whisper API
returns.

    python scripts/bench_stt.py [fixture.wav ...]        # 16kHz mono s16le

With no wav arguments it synthesises three clips with Windows SAPI. Each phase
runs in its own process: two CTranslate2 models in one process fight over
threads and the numbers stop meaning anything.
"""

import statistics
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stt import settings  # noqa: E402
from stt.transcriber import WhisperTranscriber  # noqa: E402

REPEATS = 4
SENTENCES = [
    "Hello.",
    "My name is Prakhar Jha and I want to learn about data science.",
    "Can you explain what a neural network is, and how it learns from data "
    "during training? I would like a simple example as well.",
]


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), (
            f"{path}: need 16kHz mono 16-bit"
        )
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def sapi_fixtures() -> list[str]:
    """Three spoken clips via Windows SAPI, so the bench needs no assets."""
    out = Path(tempfile.gettempdir()) / "stt_bench"
    out.mkdir(exist_ok=True)
    paths = [str(out / f"bench{i}.wav") for i in range(len(SENTENCES))]
    if all(Path(p).exists() for p in paths):
        return paths
    lines = [
        "Add-Type -AssemblyName System.Speech",
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,"
        "[System.Speech.AudioFormat.AudioChannel]::Mono)",
    ]
    for path, text in zip(paths, SENTENCES):
        lines.append(f'$s.SetOutputToWaveFile("{path}",$f); $s.Speak("{text}")')
    lines.append("$s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", "; ".join(lines)], check=True)
    return paths


def run_phase(phase: str, paths: list[str]) -> None:
    clips = [(Path(p).name, load_wav(p)) for p in paths]

    if phase == "before":  # stt/transcriber.py as it was, before this change
        transcriber = WhisperTranscriber(
            model_size=settings.WHISPER_MODEL_SIZE, device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE, beam_size=5,
            cpu_threads=0, single_pass=False)
        label = "BEFORE  beam=5, cpu_threads=0 (ct2 default), two encoder passes"
    else:
        transcriber = WhisperTranscriber(
            model_size=settings.WHISPER_MODEL_SIZE, device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE, beam_size=settings.WHISPER_BEAM_SIZE,
            cpu_threads=settings.WHISPER_CPU_THREADS, single_pass=settings.WHISPER_SINGLE_PASS)
        label = (f"AFTER   beam={settings.WHISPER_BEAM_SIZE}, "
                 f"cpu_threads={settings.WHISPER_CPU_THREADS}, "
                 f"single_pass={settings.WHISPER_SINGLE_PASS}")

    transcriber.warmup()
    print(f"\n{label}   (model={settings.WHISPER_MODEL_SIZE}/{settings.WHISPER_COMPUTE_TYPE})")
    for name, audio in clips:
        runs = [transcriber.transcribe_array(audio) for _ in range(REPEATS)]
        ms = [r[3] for r in runs]
        text, lang, prob, _ = runs[-1]
        print(f"  {name:14s} {audio.size / 16000:5.2f}s audio -> min {min(ms):6.0f} ms  "
              f"median {statistics.median(ms):6.0f} ms  [{lang} {prob:.2f}] {text!r}")

    if phase == "after" and transcriber.single_pass:
        # The fast path must agree with the public API it replaces.
        print("\n  equivalence (single encode vs faster-whisper API, same beam):")
        ok = True
        for name, audio in clips:
            fast, plain = transcriber._single_encode(audio), transcriber._public_api(audio)
            same = fast[0] == plain[0] and fast[1] == plain[1] and abs(fast[2] - plain[2]) < 1e-6
            ok &= same
            print(f"    {name:14s} {'match' if same else 'DIFFER'}")
            if not same:
                print(f"      fast : {fast}\n      plain: {plain}")
        print("  all match" if ok else "  MISMATCH -- do not ship")


def main(argv: list[str]) -> None:
    if argv and argv[0] == "--phase":
        run_phase(argv[1], argv[2:])
        return
    paths = argv or sapi_fixtures()
    for phase in ("before", "after"):
        subprocess.run([sys.executable, __file__, "--phase", phase, *paths], check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
