# RIME_EVIDENCE.md — v-tutor

## The hard voice claim

**A learner can interrupt the tutor at any moment, including while a slow tool
call is running, and (1) the tutor's audio stops within a few milliseconds,
(2) no stale speech, model reply or tool result is ever spoken, and (3) the
lesson resumes from the sentence the learner actually heard.**

The interruption path never waits for a transcript: the VAD's start-of-speech
signal flushes playback in-process, the transcript arrives later and decides
what the learner wanted. Anything produced under an older turn (Rime audio
still rendering, a model answer, a web search) is fenced by turn id and
dropped. This is the PS "Interruption and recovery" problem, with the
"conversation continuity during tool work" stress case.

## Rime configuration used for every number below

| | |
|---|---|
| Model ID | `coda` (sent explicitly as `modelId`) |
| Speaker | `clementine` (English, judged flow; chosen 2026-09-09 over `beatty`, which Rime describes as "distinctly casual", for a warmer, livelier tutor); Hindi `nadi` |
| Language code sent | BCP-47 `en` / `hi` (catalog keys `eng`/`hin` are used only to look voices up) |
| Endpoint, primary | `wss://users-ws.rime.ai/ws3` — persistent websocket, `segment=immediate`, `contextId` per line, `{"operation":"clear"}` on barge-in, `timestamps` messages for the heard cursor |
| Endpoint, fallback | `https://users.rime.ai/v1/rime-tts`, HTTPS POST, `Accept: audio/pcm` (per line, when the socket fails) |
| Audio format | raw PCM s16le mono, `samplingRate: 16000` |
| Pace | `timeScaleFactor` 0.4–2.5, higher = slower (measured below) |
| Transport | one synthesis per spoken line, disk-cached by text+voice+pace; LiveKit WebRTC audio track (room) or sounddevice (local) |
| Verified by | `python scripts\preflight.py` — live catalog lookup for every pinned voice, one uncached synthesis on the HTTP path AND one over the websocket (checks first-audio time and that word timestamps arrive), secret hygiene. Passed 15/15 on 2026-09-09. |

Rime is the only product voice. Fallback (Windows SAPI when no key; a skipped
line on a failed call) is logged as `tts_fallback` and shown as an amber
FALLBACK badge in the web UI.

## Acceptance test (defined before the demo)

Run a normal lesson, then the stress case: a **fixed 3000 ms delay inside the
web-search tool**, and an interruption **while that tool is running** that
changes the request. Pass criteria, all measured on what the learner hears:

| # | Criterion | Threshold |
|---|---|---|
| A1 | VAD start → playback flushed | ≤ 150 ms (PS target); we measure in-process |
| A2 | Stale results spoken | 0. Every drop is logged (`tts_drop_stale`, `fence_drop`, `discard`, `web_aborted`) |
| A3 | The interrupting request is answered, and the lesson resumes from the interrupted sentence | observed in the transcript |
| A4 | The application keeps accepting audio while Rime plays and while the tool runs | the interruption in the stress case lands during the tool call |
| A5 | Fallback is visible | provider badge / `tts_fallback` event |

## Procedure (repeatable)

Offline, no keys, ~15 s — the fences as unit tests:

```powershell
..\venv\Scripts\python -m pytest tests\test_fencing.py tests\test_voice_bridge.py -q
```

`test_stale_web_result_is_never_spoken`, `test_stress_delay_aborts_slow_tool_when_turn_moves_on`,
`test_stale_explain_is_discarded_too`, `test_stale_direct_answer_is_discarded`, and the
bridge tests for the heard cursor; `tests\test_streaming_player.py` for playback
of a line that is still arriving. 269 tests in the full suite (`python -m pytest tests -q`).

Whole chain, no microphone, real VAD + real STT + real Groq + real Rime. The
learner's lines are rendered by Rime in a different voice and fed through the
VAD as audio. The `!` prefix speaks that line 1.5 s after the previous
transcript instead of waiting for the tutor, i.e. inside the delayed tool call:

```powershell
..\venv\Scripts\python scripts\voice_dry_run.py --stress-ms 3000 --say "English" --say "the heart for class six" `
    --say "who won the football match yesterday" --say "!how many chambers does the heart have" --say "stop for today"
```

Live, laptop mic, same stress knob: `..\venv\Scripts\python main.py local --lang en --stress-ms 3000`
(wear headphones), or the web UI (`..\venv\Scripts\python main.py dev` + `..\venv\Scripts\python web\server.py`,
with `STRESS_DELAY_MS=3000` in the environment; the badge shows "stress +3000 ms").

Every run writes `evidence\<session>.csv` (one row per event, ms since start).
Recompute every number below from those files:

```powershell
..\venv\Scripts\python scripts\evidence_summary.py
```

## Result

### The stress case (evidence/dryrun-1788898882.csv, 2026-09-09)

| ms | event | what happened |
|---|---|---|
| 40 782 | transcript | "Who won the football match yesterday?" — not in the notes |
| 41 421 | answer_mode lookup | the model asks for a web search; the tool is delayed 3000 ms |
| 42 131 | filler | "Let me check that." (gap filler after 0.7 s of silence) |
| 44 388 | **vad_start**, stop 0.13 ms | learner cuts in during the delayed tool: "How many chambers does the heart have?" |
| 44 388 | **web_aborted** born=3 live=4 | the tool loop notices the turn moved on and returns without searching |
| 44 392 | **discard** born=3 live=4 | the stale branch's answer is dropped at the graph gate, never synthesised |
| 47 855 | transcript | the new question is classified and answered from the notes |
| 48 260 | speak answer | "The heart has four chambers. There are two upper atria and two lower ventricles." |
| 51 300 | speak system | "So, back to where we were." then the lesson resumes at the interrupted sentence |

Summary line of that run: `tutor lines: 10  stale drops: 1  finished: True`.
The football answer was never spoken; the updated request was.

### Aggregate over every committed session

48 sessions (`evidence/*.csv`: local mic, LiveKit room, dry runs, UI smoke),
2 674 events, 222 barge-ins of which 113 landed mid-utterance with a heard
cursor. Output of `scripts\evidence_summary.py` on 2026-09-09:

| Measurement | n | median | p95 | max | Criterion |
|---|---|---|---|---|---|
| VAD start → playback flushed | 222 | **0.15 ms** | 8.55 ms | 17.75 ms | A1 ✔ (≤ 150 ms) |
| Stale results spoken | — | **0** | | | A2 ✔ |
| Stale results fenced: `tts_drop_stale` 38 · `fence_drop` 15 · `discard` 3 | 56 | | | | A2 |
| Rime synth, **uncached** HTTPS round trip | 216 | 3 151 ms | 7 531 ms | 11 138 ms | disclosed |
| Rime synth, **cached** (disk) | 244 | 0 ms | 4 ms | 16 ms | disclosed |
| Rime audio produced | 460 lines, 7 896 words, 49.8 min; 10 `tts_fallback` (line skipped, logged) | | | | A5 |
| STT inference (local `base`, most sessions) | 227 | 1 794 ms | 2 861 ms | 5 708 ms | see note |
| VAD start → transcript in hand (includes the utterance itself + 0.5 s end-of-speech) | 224 | 3 514 ms | 6 615 ms | 9 711 ms | |
| Transcript → graph parked again | 227 | 2 793 ms | 15 019 ms | 32 965 ms | includes lesson builds |
| Ingest → first lesson line | 33 | 6 765 ms | 10 345 ms | 24 119 ms | not optimised |

Cached and uncached Rime numbers are separated at the source: `voice/tts.py`
records whether each line came from disk, and the summary never mixes them.

**STT note.** Most committed sessions used local faster-whisper `base` (the
1.8 s median above). On 2026-09-09 the default ear became Whisper
large-v3-turbo on Groq (`stt/cloud.py`); in the stress run above it measured
446–750 ms per utterance. A/B on eight Rime-rendered learner lines: 290–560 ms
for both ears, the cloud model fixing the one line `base` misheard.

### Rime transport and controlled delivery (2026-09-09, added after the aggregate above)

Websocket vs HTTP, same sentence, same voice, uncached (`scripts\preflight.py`
and `evidence/dryrun-1788900117.csv`):

| Path | Time to first audio | Full line | Word timestamps |
|---|---|---|---|
| HTTPS one-shot | = full line | 2 393 ms (preflight), 2.7–3.4 s typical | none |
| Websocket `ws3` | **375–470 ms** (8 of 8 uncached lines in the dry run; 441 ms in preflight) | 0.9–4.0 s, proportional to clip length | on every English line |

Playback is streamed: the line is handed to the player on the first chunk and
grows while it plays, so the learner hears the tutor ~0.4 s after synthesis
starts instead of 2–4 s. Over all committed websocket lines
(`scripts\evidence_summary.py`, 2026-09-09): time to first audio n=16, median
430 ms, p95 470 ms, max 481 ms; every English line carried word timestamps.
In `evidence/dryrun-1788900721.csv` (real-time pace) a barge-in landed while a
27-word beat was still arriving: playback stopped in 0.12 ms, Rime received
`clear`, the item was closed with 1.69 s of audio, the question was answered
and the beat resumed.

Pace control, rendered and saved (`scripts\speed_check.py` → `evidence/speed/`;
one sentence, model held constant, clips committed for both voices tried):

| Variant | `beatty` | `clementine` |
|---|---|---|
| `timeScaleFactor=0.7` | 3.31 s | 3.53 s |
| `timeScaleFactor=1.0` | 5.21 s | 5.61 s |
| `timeScaleFactor=1.3` | 7.39 s | 7.49 s |
| `speedAlpha=0.7` | 8.43 s (slower) | 7.75 s (slower) |
| `speedAlpha=1.3` | 5.17 s (no change) | 4.11 s (faster) |

So on coda **higher `timeScaleFactor` is slower**, consistently, while
`speedAlpha` behaves differently per voice and is not a usable speed control;
"slower"/"faster" are implemented on `timeScaleFactor`.
Coda is generative, so two renders of the same text differ by up to ~25% in
length; the direction above is far outside that variance. The timestamps
message and the delivered audio drift by up to ~19% at non-default speeds, so
word ends are rescaled to the clip's real duration before use.

## Limitations

- **Stop latency is measured in-process** (VAD start → player flushed), not at
  the loudspeaker. Device buffers add up to the block size (20 ms local; the
  LiveKit track's 200 ms queue in room mode, which is also flushed).
- **The heard cursor is exact only where Rime gives timestamps** (English over
  the websocket). Hindi, HTTP-fallback lines and clips cached before 2026-09-09
  use an even spread over the clip. The graph resumes at a sentence boundary,
  so the error is within a sentence either way.
- **Mid-line heard cursor is estimated, not exact.** Rime's `timestamps`
  message arrives with the end of the line, so an interruption while a line is
  still streaming in uses coda's typical pace (0.38 s/word) to estimate the
  words heard; once the line has fully arrived the cursor is exact. The graph
  resumes at a sentence boundary either way.
- **A socket failure mid-line truncates that line** (what arrived is played;
  the line is not re-sent over HTTP, which would repeat it). Logged as
  `status: truncated` on the `tts` event.
- **No emotion controls on coda.** Rime documents no emotion or style tags for
  coda (those belong to other models), so expressiveness is written into the
  text following Rime's prompting guide; it is not a synthesis parameter.
- **No acoustic echo cancellation in local mode.** Without headphones the mic
  hears the tutor; an echo guard drops verbatim echoes (4 in the data) and the
  session drops to half duplex after two (1 in the data), where barge-in waits
  for the transcript (~2 s) instead of firing on the VAD. The browser path has AEC.
- **STT mishears short or noisy utterances**, especially the local `base`
  model; the tutor now says the topic back before building a lesson so a
  mishearing is caught early. The cloud ear needs a network connection; on
  failure it falls back to local `base` for that utterance.
- **Unsupported input:** topics Wikipedia cannot resolve get "could you try a
  different topic"; scanned PDFs get "I can't read that file"; languages other
  than English and Hindi are refused for the whole lesson (one-off explanations
  in six more are supported).
- Aggregate numbers include development sessions with misheard topics and
  deliberately broken runs; they are not a curated best case.
