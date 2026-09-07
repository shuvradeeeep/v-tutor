# Running the full voice pipeline (mic → STT → agents → Rime → speaker)

> Status 2026-09-07: **working end to end**, verified with
> `scripts/voice_dry_run.py` (synthesised learner speech through the real
> Silero VAD, real Whisper, real Groq models, real Rime coda). Not tuned.

```
 mic / LiveKit track ──► stt/vad.py (Silero) ──START_OF_SPEECH──► voice/bridge.py: clock.bump() + player.flush()   (fast path, <1 ms)
                              │
                              └──END_OF_SPEECH──► stt/transcriber.py (Whisper) ──text──► bridge.on_transcript()
                                                                                          │  user_barge_in {text, cursor}
                                                                                          ▼
                                                                                 agents/graph.py (LangGraph tutor)
                                                                                          │  _say(text)
                                                                                          ▼
                                                     voice/tts.py RimeSpeaker: Rime HTTP (coda, PCM 16 kHz, disk-cached) ── fence ──► voice/player.py
                                                                                          │
                                                       LocalPlayer (sounddevice) │ LiveKitPlayer (audio track) │ TimedPlayer (silent dry run)
                                                                                          │ drained + graph parked
                                                                                          └──► playback_confirmed ──► next beat
```

`voice/` is the only new code. `stt/` gained two additive things: an
`on_speech_start` callback and `process_frames()` (same segmenter over any
frame source). `agents/` is unchanged except that an empty transcript (a cough)
classifies as a backchannel so the lesson resumes instead of asking "pardon?".

---

## 0. One-time setup

```powershell
cd d:\v-tutor-folder\v-tutor
..\venv\Scripts\python -m pip install -r requirements.txt -r requirements-audio.txt
```

`.env` needs: `RIME_API_KEY`, `RIME_SPEAKER_EN` (a coda English voice; `beatty`
is set), `LLM_*` (Groq keys are set), and for room mode `LIVEKIT_URL`,
`LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`. Whisper `base` weights download on
first run (~150 MB) and load in ~10 s.

---

## 1. Dry run — no microphone, everything else real

The fastest way to see the whole chain work, and what CI would run if it had
keys. The learner's lines are rendered with Rime (a different voice), fed
through the real VAD + Whisper as if they came from a mic, and the tutor's
lines come back through the real graph and Rime.

```powershell
..\venv\Scripts\python scripts\voice_dry_run.py
..\venv\Scripts\python scripts\voice_dry_run.py --speakers        # hear the tutor
..\venv\Scripts\python scripts\voice_dry_run.py --say "English" --say "photosynthesis for class six" `
    --say "what does chlorophyll mean" --say "slower" --say "stop for today"
```

You see, in order: `vad_start` with the heard cursor (stop latency in ms),
`transcript` with Whisper latency, the graph's `intent` / `retrieve` /
`answer_mode` events, every tutor line, `tts_drop_stale` when a barge-in lands
during synthesis, and `playback_confirmed` when a beat finished. A summary
table and the evidence CSV path print at the end.

## 2. Local — laptop mic and speakers, no LiveKit

```powershell
..\venv\Scripts\python main.py local
..\venv\Scripts\python main.py local --lang en                    # skip the spoken language question
..\venv\Scripts\python main.py local --pdf my_chapter.pdf --lang en --session demo1
..\venv\Scripts\python main.py local --stress-ms 3000             # PS full-duplex test: 3 s tool delay
```

**Wear headphones.** There is no echo cancellation on this path; through
speakers the mic hears the tutor and interrupts it. Then just talk: answer the
two onboarding questions, listen, interrupt whenever you like ("wait, what does
that mean", "slower", "go back to the part about valves", "pause", "stop for
today"). `Ctrl+C` ends the session.

Pick devices with `--input-device N --output-device N`
(`..\venv\Scripts\python -c "import sounddevice; print(sounddevice.query_devices())"`).

## 3. LiveKit room — browser mic, tutor publishes an audio track

Terminal 1, the agent worker (joins any room created on `LIVEKIT_URL`):

```powershell
$env:TUTOR_LANG = "en"                                 # optional: skip the language question
$env:TUTOR_PDF = "d:\path\to\chapter.pdf"              # optional: PDF mode
..\venv\Scripts\python main.py dev
```

Terminal 2, a join link for the browser:

```powershell
..\venv\Scripts\python scripts\room_token.py --room demo --identity student
```

Open the printed `https://meet.livekit.io/custom?...` link, allow the mic. The
worker logs `received job request`, `joined room demo`, `listening to student`,
and the tutor's voice plays in the browser.

How dispatch works: the worker registers as agent `v-tutor` (`TUTOR_AGENT_NAME`)
and the token carries a `RoomAgentDispatch` for that name, so the tutor is
dispatched every time a participant joins with that token, even into a room
that already exists. (Auto-dispatch, the default for unnamed agents, only fires
when a room is *created*; a stale room from a crashed run once left the worker
sitting at `registered worker` forever.) If you ever see that: mint a new token
with this script, or use a new `--room` name. The browser does echo cancellation, so speakers are fine
here. Every tutor line and learner transcript is also sent as a LiveKit data
message on topic `tutor` (`{"role": "tutor"|"learner", "text": ...}`) for any
UI that wants captions.

---

## 4. Where the outputs are

| What | Where |
|---|---|
| Live console | tutor lines in brackets, `. event {payload}` lines underneath |
| Full log | `logs/voice.log` |
| Evidence CSV (every event, ms since start) | `evidence/<session>.csv` — `vad_start.stop_ms`, `transcript.whisper_ms`, `transcript.since_vad_start_ms`, `tts.synth_ms`, `fence_drop`, `tts_drop_stale`, `playback_confirmed` |
| Session state (resumable) | `sessions.db` (SQLite); rerun with the same `--session` to resume |
| Rime audio cache | `.cache/tts/*.pcm` — fixed phrases cost nothing after the first run |
| Offline proof, no keys | `..\venv\Scripts\python -m pytest tests -q` (152 tests; `tests/test_voice_bridge.py` covers the bridge) |

---

## 4b. Room-mode troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Worker stuck at `registered worker` after you join | old token without dispatch, or room pre-existed | re-mint with `scripts/room_token.py` (embeds dispatch) |
| `Plugins must be registered on the main thread` | Silero imported first inside the job thread | fixed: `main.py` imports the plugin before the worker starts |
| Tutor joins, never hears you | browser mic blocked or muted | allow mic in the Meet UI; worker subscribes to audio only |
| Tutor silent, console shows `tts_fallback` | Rime call failed | check `RIME_API_KEY`, network |
| Barge-ins with nobody talking | VAD picks up room noise | `VAD_MIN_SPEECH_DURATION=0.3`, better mic |

## 5. What the dry run measured (2026-09-07, this laptop, CPU)

| Stage | Measured | Knob |
|---|---|---|
| VAD start → playback stopped | 0.15–0.27 ms (in-process flush) | — |
| Whisper `base`, int8, beam 5 | 2.3–2.5 s per utterance | `WHISPER_BEAM_SIZE=1` (≈2× faster), `WHISPER_MODEL_SIZE=tiny` |
| VAD start → transcript delivered | 4–6.5 s (includes the utterance itself + 0.5 s end-of-speech silence) | `VAD_MIN_SILENCE_DURATION` |
| Rime coda HTTP, one line | 2–3 s (cached: 0) | model `arcana`/`mistv2` are faster; websocket streaming not built |
| Groq answer (direct) | 0.4–0.7 s | — |
| Lesson prep after "give me a moment" | ~12 s (Wikipedia + section pick + simplify) | `PREPARE_UPFRONT_SECTIONS`, see AGENT_GAPS |

Known behaviour to expect:

- **Whisper `base` mishears topics.** In the dry run "the heart for class six"
  came back as "The Hartford Class 6" and the tutor cheerfully taught the
  Hartford insurance article. Use `WHISPER_MODEL_SIZE=small` for the demo, or
  say the topic slowly; "start over" does not change the topic (restart the
  session for that).
- The tutor's first words after a question take ~3 s (LLM + Rime). The gap
  filler says "Let me check that" / "One moment" after 0.7 s, once per turn.
- A barge-in during Rime synthesis is logged as `tts_drop_stale` and never
  played. A barge-in during an LLM call or web search is dropped by the graph
  fence (`fence_drop` / `discard`). Both are the PS's "stale result never spoken".
- Heard cursor is word-approximate (words spread evenly over the clip, HTTP
  gives no timestamps); the graph rounds to a sentence boundary anyway.
