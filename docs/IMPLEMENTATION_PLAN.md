# v-tutor — Implementation Plan

Build order is deliberately **inverted from the data flow**. The scored core is
Rime + barge-in (25% hard voice engineering + 20% Rime integration + 20%
evidence = 65%). The tutor content is the *setting* for that problem, not the
problem. So Layer 3 gets built and measured first, against a fake text source,
long before the LangGraph reasoning exists.

> Rule for the whole build: **nothing is "done" until it has been heard.**
> Every phase below ends with an artifact you can listen to or a number you
> measured, not a passing import.

---

## Phase 0 — Environment and preflight

**Goal:** a reproducible env, and proof our Rime config is valid *before* we
write code against it.

| File | Purpose |
|---|---|
| `requirements.txt` | Pinned deps |
| `.env.example` | Placeholders only — never real keys (PS eligibility rule) |
| `config.py` | Single source of truth for model / speaker / lang / endpoint / format |
| `scripts/preflight.py` | Validates config against the **live** catalog |

`scripts/preflight.py` must:
1. `GET https://users.rime.ai/data/voices/all-v2.json`.
2. Assert `config.SPEAKER` exists under `catalog[MODEL_ID][lang3]`.
3. Open the WebSocket, synthesise one short line, assert audio bytes returned.
4. Print the exact config block for pasting into the README.
5. Exit non-zero on any mismatch.

This directly satisfies "use the current catalog at submission time rather than
copying a stale speaker list" and gives us a one-command answer to the
organizer's preflight check.

**Environment note:** Python 3.13 removed the `audioop` module. All resampling
must be numpy-based — already fixed in `stt/agent.py`. Do not reintroduce
`audioop` or the `audioop-lts` shim.

**Exit criteria:** `python scripts/preflight.py` prints a green config block and
writes one `.wav` you can play.

---

## Phase 1 — Rime streaming client

**File:** `tts/rime_client.py`

The heart of the submission. A persistent WebSocket wrapper with cancellation as
a first-class operation.

```
class RimeStreamClient:
    async def connect()                  # one socket, reused across turns
    async def speak(text_iter, context_id)   # feed tokens as the LLM emits them
    async def clear(context_id)          # {"operation":"clear"} — barge-in
    async def close()
    # emits: AudioChunk(context_id, pcm), WordTimestamps(...), Done, Error
```

Requirements:
- URL params: `?speaker={S}&modelId=coda&audioFormat=pcm&samplingRate=16000`
- Auth: `Authorization: Bearer $RIME_API_KEY` header, server-side only.
- Send `{"text": ...}` incrementally; `{"operation":"eos"}` to close a batch.
- **Every emitted chunk is tagged with the `context_id` it was requested under.**
  This is fence layer 2 and is non-negotiable.
- Reconnect with backoff; surface state changes so the provider badge can show
  `RIME` vs `FALLBACK`.

**Exit criteria:** a script that streams a 3-sentence paragraph, calls `clear()`
1.2 s in, and shows that no audio bytes tagged with the cancelled context arrive
afterwards.

---

## Phase 2 — Playback engine and the heard-cursor

**File:** `tts/player.py`

```
class PlaybackEngine:
    async def enqueue(chunk)      # drops chunk if chunk.context_id != current
    async def flush()             # instant local kill — fence layer 3
    def heard_cursor() -> Cursor  # (word_index, char_offset, seconds)
```

The cursor is the interesting part. Two inputs are combined:

1. Rime's `word_timestamps` — when each word *would* be spoken.
2. Frames the engine actually **rendered to the track** — how far we truly got.

`heard_cursor = last word whose end-time ≤ frames_rendered / sample_rate`

Do not trust the synthesis timeline alone: if the socket stalls or the buffer
underruns, Rime's timestamps say word 20 while the learner heard word 11.
Measuring rendered frames is what makes the claim honest.

**Exit criteria:** interrupt at a known point 10×; the reported cursor matches
what a human transcriber says was audible, ±1 word.

---

## Phase 3 — Barge-in harness (measure before you build the tutor)

**File:** `scripts/bargein_bench.py`

Before any LangGraph node exists, wire VAD → kill path → player directly and
measure it. Owning the number early prevents discovering a 600 ms stop on demo
day.

Measures, over N≥30 trials:
- `t_stop` — VAD speech-start → last audio sample rendered.
- `stale_leak_count` — audio frames from a cancelled context that reached the
  track. **Must be 0.**
- `cursor_error` — words claimed heard minus words actually heard.

Writes `evidence/bargein_results.csv` + a summary. Report cold vs warm socket
separately (PS explicitly requires this).

**Exit criteria:** committed CSV, `stale_leak_count == 0`, documented p50/p95.

---

## Phase 4 — Agentic layer

See **[AGENTS_PLAN.md](AGENTS_PLAN.md)** for the full node-by-node design.

Files: `agents/state.py`, `agents/nodes.py`, `agents/graph.py`,
`agents/gap_filler.py`.

---

## Phase 5 — Realtime loop integration

**Files:** `stt/vad.py`, `stt/transcriber.py`, `main.py`

Refactor the existing `stt/agent.py` prototype into two reusable pieces
(`vad.py` emits events, `transcriber.py` does Whisper), then `main.py` becomes
the LiveKit entrypoint that owns the graph, the Rime client, and the player.

Critical wiring detail: the VAD callback must invoke the kill path **directly**,
not by scheduling graph work. If barge-in has to wait for a LangGraph node to be
scheduled, the ~120 ms budget is gone.

---

## Phase 6 — Evidence and docs

| Artifact | Contents |
|---|---|
| `RIME_EVIDENCE.md` | Claim, acceptance test, procedure, result, limitations |
| `evidence/*.csv` | Item-level results, not just summaries |
| `evidence/clips/` | Before/after audio, including the speed_alpha A/B |
| `README.md` | Setup, architecture, exact Rime config, failure behaviour, limits |

The `speed_alpha` A/B is mandatory given the inverted-direction trap: render the
same sentence at three alpha values on `coda`, commit the clips, and state which
direction is actually slower. Hold model and voice constant, per the PS.

---

## Acceptance test (define before the demo — PS requirement)

> **Claim.** When the learner interrupts, v-tutor stops audible speech within
> **150 ms (p95)**, never speaks a single frame from a superseded turn, and
> resumes within **one word** of where the learner actually stopped hearing.

**Normal case.** Ask an in-syllabus question mid-lesson; get an answer grounded
in the material; lesson resumes from the heard cursor.

**Stress case** (the PS's full-duplex recipe). Inject a fixed 3 s delay into
`web_search_node`. Interrupt while the tutor is speaking *and* the tool is
running, and change part of the request. Assert:
- queued Rime audio stops promptly,
- the updated instruction reaches the graph,
- the 3 s tool result from the old turn is **dropped at `fence_check`**, never
  spoken,
- the final spoken reply reflects what the learner actually heard and asked.

**Measured, not proxied.** `t_stop` is measured at frames rendered to the
outbound track, not at the moment we called `clear()`.

---

## Risk register

| Risk | Mitigation |
|---|---|
| `speed_alpha` direction on coda unknown | Verify by ear in Phase 0; commit clips |
| Local Whisper too slow on CPU, inflating latency | It is off the kill path by design; only affects intent, not stop time |
| Hindi has 3 coda voices → voice identity shifts on language switch | Scope the demo to one language for the judged flow; disclose |
| LiveKit Agents API drift vs the prototype's assumptions | Pin `livekit-agents~=1.5`; the plugin exposes `use_websocket=True` |
| Doing all of this in the time available | Phases 0–3 alone are a submittable product with a real measured claim |

**If time runs short, cut Phase 4 scope, not Phase 3.** A tutor with a scripted
lesson and a rigorously proven barge-in scores far better than a clever agent
with an unmeasured voice layer.

---

## Phase 7 — Hardening against real sessions (2026-09-08)

The plan above ends at "it works". This phase was not planned: it is what a
dozen sessions with a real microphone demanded. Each item was found by
listening, not by testing, and each one now has a regression test named after
the session that produced it. Detail lives in
[STT_AND_INTENTS.md](STT_AND_INTENTS.md) and [AGENTS_PLAN.md](AGENTS_PLAN.md) §10.

**Measured, before and after**

| | before | after |
|---|---|---|
| Whisper per utterance (`base`, int8, CPU) | 850–990 ms | **430–560 ms** |
| Whisper per utterance (`small`) | — | ~1.8 s (cheaper than `base` was) |
| Provider errors in a 15-min, 44-question session | 429s, silent quality loss | **0** |
| Learner-visible waiting for LLM quota | up to 5 s per answer | **0 s** |
| Offline tests | 152 | **210** |

**What was done, in the order it mattered**

1. **STT latency.** One encoder pass instead of two, `cpu_threads` = physical
   cores, greedy decoding, model warmed at startup. `scripts/bench_stt.py`
   A/Bs it in separate processes and asserts the fast path returns byte-identical
   text to the plain faster-whisper API.
2. **One config, two entrypoints.** `stt/agent.py` and `voice/audio_io.py` had
   separate copies of the same env knobs (with different defaults). Both now
   read `stt/settings.py`. Transcripts from either go to one
   `stt/transcripts.csv` via `stt/transcripts.py`.
3. **Offline demonstrability.** `scripts/voice_dry_run.py --sapi` renders the
   learner with Windows TTS, and `TTS_PROVIDER=sapi` renders the tutor, so the
   whole chain is checkable with no API keys at all.
4. **Acoustic feedback.** Self-echo detection plus a half-duplex mode, because
   without headphones the tutor answers its own voice and derails.
5. **Onboarding guards.** Confusion, greetings and quit are handled before an
   utterance is read as an answer; "which class?" only accepts a class.
6. **Session memory.** Lesson, progress, interrupted sentence and conversation
   history on every answer prompt; follow-ups resolved against it.
7. **A `meta` intent** for questions about the session, answered from the
   lesson plan with no model call.
8. **Topic switching** as a first-class outcome of `navigate`, in English,
   Hindi and Hinglish.
9. **Provider budgets.** `max_tokens` sized to measured work, real usage
   metered per model, 429 retried with the provider's own delay, background
   prep held below the learner's share of each minute.
10. **Language integrity.** Spoken Hindi detected as Urdu is re-decoded as
    Hindi before the transcript exists; text in a script the voice cannot speak
    is never read aloud.

**Acceptance claim from Phase 3 still holds**, and the numbers behind it are
unchanged: `vad_start.stop_ms` is 0.1–0.3 ms with headphones. In half-duplex
mode (speakers) the stop is deliberately deferred by one Whisper pass, and that
is the documented trade, not a regression.
