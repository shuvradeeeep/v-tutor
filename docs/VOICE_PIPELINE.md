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

`.env` needs: `RIME_API_KEY`, `RIME_SPEAKER_EN` (a coda English voice; `clementine`
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
.\.venv\Scripts\python scripts\voice_dry_run.py
.\.venv\Scripts\python scripts\voice_dry_run.py --speakers        # hear the tutor
.\.venv\Scripts\python scripts\voice_dry_run.py --sapi            # no RIME_API_KEY: Windows TTS renders the learner
.\.venv\Scripts\python scripts\voice_dry_run.py --say "English" --say "photosynthesis for class six" `
    --say "what does chlorophyll mean" --say "slower" --say "stop for today"
```

The PS full-duplex stress test, no microphone, repeatable (see
[RIME_EVIDENCE.md](../RIME_EVIDENCE.md)): a 3 s delay inside the web-search
tool, and a `!`-prefixed line that is spoken 1.5 s after the previous transcript
instead of waiting for the tutor, i.e. while the slow tool call is in flight.

```powershell
.\.venv\Scripts\python scripts\voice_dry_run.py --stress-ms 3000 --say "English" --say "the heart for class six" `
    --say "who won the football match yesterday" --say "!how many chambers does the heart have" --say "stop for today"
```

`--sapi` (also chosen automatically when `RIME_API_KEY` is unset) renders the
learner's lines with the Windows speech synthesiser instead of Rime, so the
whole chain — VAD, Whisper, graph, tutor text — is checkable with no keys and
no network beyond Wikipedia. Without `LLM_*` keys the graph still runs; it takes
its deterministic fallback paths, so the tutor reads the article rather than a
model-simplified version of it.

You see, in order: `vad_start` with the heard cursor (stop latency in ms),
`transcript` with Whisper latency, the graph's `intent` / `retrieve` /
`answer_mode` events, every tutor line, `tts_drop_stale` when a barge-in lands
during synthesis, and `playback_confirmed` when a beat finished. A summary
table and the evidence CSV path print at the end.

Every utterance is also appended to `stt/transcripts.csv` (`stt/transcripts.py`,
same columns whichever entrypoint is running), so a tutor session leaves the
same transcript trail as the standalone STT worker. Follow it live with
`Get-Content -Wait stt\transcripts.csv`. Set `LOG_TRANSCRIPTS=0` to turn it off.

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

## 3. Web UI — laptop or phone browser

The page in `web/` joins a LiveKit room, plays the tutor's track, and shows
captions, the lesson plan with progress, the heard cursor at every barge-in,
every stale result the fences dropped, and the active speech provider
(`RIME · coda · clementine`, or `FALLBACK`). Buttons (Pause, Slower, Repeat, End,
language chips) travel down the same path as spoken words: the worker feeds
them to the bridge as transcripts, so they hit the same intent rules and the
same evidence CSV.

Terminal 1, the worker; terminal 2, the page server:

```powershell
.\v-tutor\Scripts\python main.py dev
.\v-tutor\Scripts\python web\server.py            # http://localhost:8080  (laptop)
.\v-tutor\Scripts\python web\server.py --https    # https://<lan-ip>:8443  (phone on the same Wi-Fi)
```

Phones only allow the microphone on HTTPS; `--https` makes a self-signed
certificate under `.cache/certs` and you accept the warning once. Each visit
gets its own room (`lesson-xxxx`) because the worker listens to one learner per
room; a second microphone joining the same room is ignored and shown as a note.
Phones may route the tutor to the call earpiece once the microphone is open:
the page asks for the loudspeaker where the browser allows it (Android Chrome)
and plays through Web Audio, which iOS routes to the speaker more often.
Earphones are the sure fix and also give the best barge-in. The LiveKit
secret stays in the server: the page only receives a 3-hour room token from
`/token`. Speakers are fine, the browser does echo cancellation.

Without a browser, `scripts\ui_smoke.py` joins as the page would, presses the
buttons, and checks the tutor's replies, events and state snapshots arrive
(`PASS`/`FAIL`, exit code). Data topics the worker publishes: `tutor` (lines,
with `kind` lesson/answer/system), `events` (the bridge and graph events the
page reacts to), `state` (lesson, progress, provider, paused, finished);
it listens on `control` (`{"say": "pause"}`).

## 3b. LiveKit Meet — no UI, just the room

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
| Transcripts (both entrypoints) | `stt/transcripts.csv` — one row per utterance with `whisper_ms` and `total_latency_ms`; follow a live session with `Get-Content -Wait stt\transcripts.csv` |
| Offline proof, no keys | `.\.venv\Scripts\python -m pytest tests -q` (210 tests, ~7 s; `tests/test_voice_bridge.py` covers the bridge, `tests/test_llm_budget.py` the rate limiting) |

---

## 4b. Room-mode troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Worker stuck at `registered worker` after you join | old token without dispatch, or room pre-existed | re-mint with `scripts/room_token.py` (embeds dispatch) |
| `Plugins must be registered on the main thread` | Silero imported first inside the job thread | fixed: `main.py` imports the plugin before the worker starts |
| Tutor joins, never hears you | browser mic blocked or muted | allow mic in the Meet UI; worker subscribes to audio only |
| Tutor silent, console shows `tts_fallback` | Rime call failed | check `RIME_API_KEY`, network |
| Barge-ins with nobody talking | VAD picks up room noise | `VAD_MIN_SPEECH_DURATION=0.3`, better mic |

---

## 4c. Local-mode troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Everything logs correctly, `tts` shows real `audio_s`, but you hear nothing | audio went to a device you are not listening to — Windows routes to the headphone jack by default on this hardware | the startup banner prints `audio out: [N] <name>`; find the one you can hear with `python scriptsudio_check.py --play --all`, then `--output-device N` |
| Tutor answers its own voice, drifts onto strange topics | no headphones: the mic hears the tutor | wear headphones, or run `--no-headphones` (half duplex). After two multi-word echoes the session switches itself and logs a warning |
| Cannot interrupt any more, mid-session | half duplex switched on after echoes | headphones + restart, or `ECHO_AUTO_HALF_DUPLEX=0` to disable the automatic switch |
| Your one-word reply is ignored | it matched a run of words the tutor just said | instructions (continue/stop/pause/repeat) are exempt from the echo guard; anything else, say two or three words |
| Hindi comes out in Arabic script, answers are nonsense | Whisper detected Urdu | fixed: `WHISPER_ALLOWED_LANGUAGES=en,hi` re-decodes it as Hindi before the transcript exists |
| Topic silently becomes something else | a misheard reply to "which class?" | fixed: only a class answers that question; misheard forms (`classics`, `glass 6`) are accepted as classes |
| `LLM ... 429` / answers suddenly get worse | provider tokens-per-minute limit | already retried automatically; if persistent, `RECENT_EXCHANGES_KEEP=3` or a smaller `LLM_STRONG_MODEL`. `LLM_TPM_LIMIT` should match your tier |

## 5. What the dry run measured (2026-09-07, this laptop, CPU)

| Stage | Measured | Knob |
|---|---|---|
| VAD start → playback stopped | 0.15–0.27 ms (in-process flush) | — |
| Whisper `base`, int8, beam 5, two encoder passes (before 2026-09-08) | 2.3–2.5 s per utterance | — |
| Whisper `base`, int8, beam 1, one encoder pass, 8 threads (now) | ~2× faster: 0.43–0.56 s on 1.3–8.6 s clips on an idle laptop, ~1.0–1.3 s while the tutor is under load | `WHISPER_MODEL_SIZE=tiny`, `WHISPER_SINGLE_PASS=0` to revert the fast path |
| VAD start → transcript delivered | 4–6.5 s before, ~1.5 s faster now (still includes the utterance itself + 0.5 s end-of-speech silence) | `VAD_MIN_SILENCE_DURATION` |
| Rime coda, one line | websocket (default since 2026-09-09): first chunk 0.38–0.47 s, whole line 0.9–4 s by length; HTTP fallback 2.4–3.4 s; cached: 0 | `RIME_TRANSPORT_MODE=http` to force one-shot; playing chunks as they arrive is not built |
| Groq answer (direct) | 0.4–0.7 s | — |
| Lesson prep after "give me a moment" | ~12 s (Wikipedia + section pick + simplify) | `PREPARE_UPFRONT_SECTIONS`, see AGENT_GAPS |

Every STT knob above lives in `stt/settings.py` (env-overridable, same names),
which is what both `voice/audio_io.py` and the standalone `stt/agent.py` read --
so tuning one does not leave the other behind. Re-measure with
`..\venv\Scripts\python scripts\bench_stt.py`, which A/Bs the old and current
settings in separate processes and checks the fast path against plain
faster-whisper.

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
