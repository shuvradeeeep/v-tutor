# DataForge Voice Tutor — Detailed Implementation Plan
### Rime x Pathway Hackathon Submission

---

## 1. Product Summary

**What it is:** A voice-native AI tutor. The user uploads material (or names a topic, class/standard, and depth), and the tutor teaches the chapter aloud, continuously, in the target language — while the user can interrupt at any point to ask a question, get an answer (from context or live web search), and have the lesson resume exactly where it paused.

**Why voice is essential:** The core interaction loop — being taught, interrupting mid-sentence, getting an answer, resuming without repetition or gaps — only exists as a voice problem. A text chatbot has no "interruption," no "resume point," and no latency-driven gap-filling need. Removing speech collapses the product to a static Q&A bot.

**Headline hard-voice claim (what we are proving):**
> When interrupted mid-sentence — including when the interrupting question requires a live web search — the tutor stops audio in under 200ms, fills the silence naturally if the answer takes time, answers correctly, and resumes the lesson from the correct point without repeating or skipping content.

This bundles three PS-recognized hard voice problems into one demoable story:
- **Interruption and recovery**
- **Conversation continuity during tool work**
- **Perceived response time** (via the gap-filler)

Multilinguality and pronunciation are supported and demonstrated, but positioned as secondary evidence, not the headline claim — per the PS's own guidance that a narrow, deeply-proven claim beats a broad shallow one.

---

## 2. High-Level Architecture

Three layers, each with a distinct responsibility and a distinct latency budget. The most important design rule in this entire plan:

> **Audio-level interruption (stopping sound) must be instant and lives OUTSIDE the reasoning layer. Content-level interruption (deciding what to say next) lives INSIDE LangGraph and is allowed to take longer, because it's reasoning, not sound.**

Conflating these two is the single most common way voice-agent builds fail the "interruption" bar — teams route the stop-audio signal through their LLM graph and get 1–2 second dead air instead of an instant cut.

```
┌──────────────────────────── OUTER REAL-TIME VOICE LOOP (LiveKit Agents) ────────────────────────────┐
│                                                                                                        │
│   Mic input ──► VAD (fast, ~100-150ms) ──► "user started talking" signal                             │
│                        │                                                                              │
│                        ├──► IMMEDIATE local audio-buffer flush (stop Rime playback, <200ms)           │
│                        │                                                                              │
│   Mic input ──► Streaming STT (multilingual) ──► finalized transcript (~300-500ms)                   │
│                        │                                                                              │
│                        ▼                                                                              │
│              inject event into LangGraph session  (keyed by thread_id = session_id)                  │
│                        │                                                                              │
│         ┌──────────────▼──────────────┐                                                              │
│         │   LangGraph "Brain" Graph    │   ◄── checkpointer (Postgres/SQLite) persists all state      │
│         │  (Master / QA / Gap-Filler / │                                                              │
│         │   Resume Controller nodes)   │                                                              │
│         └──────────────┬──────────────┘                                                              │
│                        │  tts_directive { turn_id, action, text, lang, voice_id }                    │
│                        ▼                                                                              │
│              Rime streaming TTS (sentence-chunked) ──► speaker / call leg                             │
│                                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Layer responsibilities

| Layer | Owns | Latency budget |
|---|---|---|
| **Real-time loop** (Part 3: STT + Part 4: TTS mechanics) | VAD, audio buffer flush, streaming STT, streaming TTS playback, crossfades | <200ms for stop; <500ms for transcript finalize |
| **Agentic layer** (Part 2: LangGraph) | All reasoning: what to teach next, whether input is a question/command, retrieval, web search, answer composition, resume logic, turn fencing | No hard ceiling — this is exactly what the gap-filler exists to mask |
| **TTS provider** (Rime) | Converts text → speech only | N/A — Rime is the primary spoken output for every word the tutor says |

Per the PS's build rules: **"Rime provides text-to-speech. Your application remains responsible for user input, speech recognition, reasoning, orchestration, state, transport, tools, safety, and evaluation."** This document is structured around that exact division.

---

## 3. Middle Agentic Portion — LangGraph Architecture

### 3.1 State Schema

```python
from typing import TypedDict, List, Dict, Optional, Literal

class Segment(TypedDict):
    id: str
    sentences: List[str]
    depth_tag: str          # matches requested class/std + depth

class Cursor(TypedDict):
    segment_idx: int
    sentence_idx: int
    status: Literal["not_started", "playing", "confirmed_heard"]

class TTSDirective(TypedDict):
    turn_id: int
    action: Literal["SPEAK", "STOP", "CROSSFADE_IN", "CROSSFADE_OUT"]
    text: str
    lang: str
    voice_id: str

class TutorState(TypedDict):
    session_id: str
    lesson_lang: str                     # language the chapter is being taught in
    active_lang: str                     # language of the CURRENT turn (may differ, e.g. Hindi question mid-English lesson)
    turn_id: int                         # monotonic counter — the fencing key for everything
    mode: Literal[
        "TEACHING", "INTERRUPTED", "RETRIEVING",
        "ANSWERING", "GAP_FILL", "RESUMING",
        "COMMAND", "IDLE", "DONE"
    ]
    chapter_plan: List[Segment]
    cursor: Cursor
    pending_user_input: Optional[str]
    pending_question_turn_id: Optional[int]
    qa_answer: Optional[str]
    qa_source: Optional[Literal["context", "web"]]
    context_store_id: str
    glossary: Dict[str, str]
    resume_strategy: Optional[Literal["replay_sentence", "continue_next"]]
    tts_directive: Optional[TTSDirective]
```

**Why `turn_id` is the single most important field:** every downstream artifact (a QA answer, a web search result, a gap-filler clip) is tagged with the `turn_id` active when it was requested. Before anything is allowed to reach the TTS layer, it passes through `fence_check`, which drops it if `turn_id != state.turn_id`. This is what prevents a stale search result — from a question the user already abandoned by interrupting again — from ever being spoken. It is the mechanism, not a side detail; build and stress-test it before anything else.

### 3.2 Nodes / Agents

| Node | Agent role | Description |
|---|---|---|
| `ingest_material` | **Material Fetch Agent** | Runs once at session setup. Parses the uploaded document or stated topic + class/std + depth. Builds a vector store (context), a glossary of key terms, and a sentence-level `chapter_plan`. May itself call web search to fill gaps in thin uploaded material. |
| `teach_step` | **Master Agent** | Reads `cursor`, emits the next single sentence as a `SPEAK` directive, marks `cursor.status = "playing"`. Deliberately one sentence at a time — not a paragraph — so barge-in granularity stays fine and resume logic stays precise. |
| `await_event` | **Pause point** | Uses LangGraph's `interrupt()` to pause the graph until one of two externally-injected signals arrives: `playback_confirmed` (sentence finished playing cleanly) or `user_barge_in` (VAD fired). This single mechanism is what implements the continuity requirement. |
| `handle_interrupt` | **Interrupt Handler** | Increments `turn_id`. Sets `mode = "INTERRUPTED"`. Freezes `cursor` exactly as-is — critically, `status` stays `"playing"`, meaning *not confirmed heard*. Immediately spawns the gap-filler watchdog (3.4). |
| `classify_intent` | **Intent Classifier** | Fast/cheap LLM call: is this a question, a command (repeat / slower / switch language), chit-chat, or off-topic? This sits on the critical latency path, so keep it small and fast. |
| `command_handler` | **Command Agent** | Handles non-QA commands by mutating state directly (e.g. re-emit last sentence, change `active_lang`, change speaking rate) — no retrieval needed, so it's fast. |
| `qa_retrieve` | **QA Agent, step 1** | Vector search over `context_store_id`. |
| `web_search_node` | **QA Agent, step 2 (conditional)** | Fires only if retrieval similarity is below threshold. Tags its output with the requesting `turn_id`. |
| `compose_answer` | **QA Agent, step 3** | LLM composes the final spoken answer, in `active_lang` (which may differ from `lesson_lang` — e.g. a Hindi question asked mid-English lesson). |
| `fence_check` | **Fencing gate** (function, not just a node — call it at every emission point) | `if directive.turn_id != state.turn_id: drop`. This is the concrete implementation of "cancel or fence obsolete model and tool results so they cannot re-enter the conversation." |
| `resume_controller` | **Resume Controller** | Decides `resume_strategy`: if `cursor.status == "playing"` (meaning it was never confirmed heard before the interrupt), choose `replay_sentence`; otherwise `continue_next`. Emits a short verbal bridge ("Okay, back to—") then loops to `teach_step`. |
| `end_session` | **Terminal node** | Chapter exhausted or user ends session. |

### 3.3 Graph Wiring

```
ingest_material
      │
      ▼
  teach_step ──────────────────────────────────► await_event
      ▲                                                │
      │                                    ┌───────────┴────────────┐
      │                          [playback_confirmed]      [user_barge_in]
      │                                    │                        │
      │                                    ▼                        ▼
      │                              teach_step            handle_interrupt
      │                          (advance cursor,                  │
      │                              loop)                         ▼
      │                                                     classify_intent
      │                                                     │            │
      │                                              [command]      [question]
      │                                                     │            │
      │                                                     ▼            ▼
      │                                          command_handler    qa_retrieve
      │                                                     │            │
      │                                                     │    ┌───────┴────────┐
      │                                                     │  [sim>=thresh]  [sim<thresh]
      │                                                     │    │                │
      │                                                     │    ▼                ▼
      │                                                     │ compose_answer  web_search_node
      │                                                     │    │                │
      │                                                     │    └───────┬────────┘
      │                                                     │            ▼
      │                                                     │      compose_answer
      │                                                     │            │
      │                                                     │            ▼
      │                                                     │      fence_check
      │                                                     │            │
      │                                                     └────────────┤
      │                                                                  ▼
      └───────────────────────────────────────────────────────── resume_controller
                                                                          │
                                                        [chapter exhausted]
                                                                          ▼
                                                                   end_session
```

### 3.4 The Gap-Filler: NOT a sequential graph node

The gap-filler must run concurrently with `qa_retrieve` / `web_search_node` / `compose_answer` — if it were a node in the main sequential path, it couldn't "watch" latency in real time. Implement it as a sibling async task launched the instant `handle_interrupt` fires:

```python
async def gap_filler_watchdog(turn_id: int, active_lang: str, threshold_ms: int = 600):
    await asyncio.sleep(threshold_ms / 1000)
    if state.turn_id == turn_id and state.mode in ("RETRIEVING", "ANSWERING"):
        emit_directive(action="CROSSFADE_IN",
                        text=cached_filler[active_lang],
                        turn_id=turn_id)
```

When `compose_answer` eventually succeeds and its directive passes `fence_check`, it triggers a `CROSSFADE_OUT` on the filler client-side. If a *second* interruption happens before the first resolves, the watchdog's own `turn_id` check fails and it silently no-ops — the same fencing discipline applies uniformly across the whole system, which is exactly what you want to be able to say in `RIME_EVIDENCE.md`.

### 3.5 Checkpointing = Continuity, For Free

Use LangGraph's built-in checkpointer (Postgres or SQLite) keyed by `thread_id = session_id`. Every state mutation is persisted automatically. Concrete benefits:

- Session survives a dropped connection or process restart — not just an in-conversation interruption. This is a much stronger "continuity" story than most teams will show.
- You get a free, literal audit log: replaying the checkpoint history for a session proves, mechanically, that no sentence was duplicated or skipped across an interruption. This becomes a reproducible artifact for the evidence rubric, not just an assertion.

### 3.6 Requirement → Mechanism Map

| Requirement | Concrete mechanism |
|---|---|
| **Interruption** | Outer-loop VAD flushes audio in <200ms (Part 4); `handle_interrupt` freezes state; `turn_id` fencing discards anything stale |
| **Responsiveness** (gap-filling) | `gap_filler_watchdog` + pre-cached per-language filler clips + crossfade |
| **Continuity** | `cursor.status` semantics + `resume_controller` decision logic + LangGraph checkpointer |
| **Multilinguality** | `active_lang` decoupled from `lesson_lang`; per-utterance language ID; Rime voice/model routed per `active_lang` |
| **Pronunciation** | Fixture-tested glossary terms fed into prompts; Rime delivery controls (Part 4); before/after evidence clips |

---

## 4. STT (Speech-to-Text) Plan

**Stack:** a dedicated fast VAD (Silero or webrtcvad) running independently from a streaming multilingual STT engine (e.g. Deepgram Nova or a Whisper-streaming variant). Do **not** rely on the STT engine's own endpointing to drive barge-in — it is not fast enough on its own.

### 4.1 Two-speed pipeline

| Path | Trigger | Latency target | Purpose |
|---|---|---|---|
| **Fast path** | VAD detects speech onset | ~100–150ms | Signals "user started talking" → immediately flush the TTS audio buffer. No transcript required yet. |
| **Slow path** | STT finalizes on silence-based endpointing | ~300–500ms | Produces the actual transcript that feeds `classify_intent` in the graph. |

This split is precisely what lets the system hit sub-200ms audio-stop while still giving the reasoning layer a clean, complete transcript to work from — conflating the two paths is the most common cause of either false interruptions or laggy stops.

### 4.2 Backchannel filtering

Short interim utterances — "hmm," "okay," "right" — must **not** trigger a full `handle_interrupt`. Classify these client-side, before dispatching to the graph, or the demo will show false interruptions that visibly hurt the "interruption and recovery" story.

### 4.3 Language identification

Run per-utterance language ID (either the STT engine's built-in multilingual mode, or a lightweight separate classifier) so that, e.g., a Hindi question asked mid-English lesson is decoded correctly and `active_lang` is set properly before it reaches the QA and TTS legs.

### 4.4 Code-switching

Use an STT model that decodes mixed-language utterances natively rather than hard-splitting audio by detected language boundary — this matters directly if the demo includes an English technical term embedded in a Hindi sentence, which is a realistic and worth-showing stress case.

### 4.5 Interim vs. final transcripts

Interim transcripts are barge-in *signal only*. Only the **final**, silence-confirmed transcript is allowed to reach `classify_intent` — a partial transcript that later gets corrected must never leak into the reasoning layer, or the QA agent will occasionally answer the wrong question.

---

## 5. TTS (Rime) Plan

### 5.1 Streaming, sentence-chunked synthesis

`teach_step` deliberately emits one sentence per invocation specifically so Rime can begin speaking sentence *n* while sentence *n+1* is still being generated or queued. This is the primary lever against perceived latency during normal teaching flow, independent of the gap-filler (which only covers *interruption*-induced latency).

### 5.2 Instant local cancellation

On barge-in, the **client-side audio buffer is flushed immediately** — the system does not wait on a round trip to Rime to confirm a stop. Rime's stream may continue generating momentarily (or be cancelled server-side), but the speaker output is muted/cleared the instant VAD fires. This is the literal mechanism the PS's "stop queued TTS input and local playback promptly" requirement is testing, and it must be demonstrably instantaneous in the demo.

### 5.3 Gap-filler caching

Pre-synthesize a small pool of filler phrases (2–3 per supported language — e.g. "Let me check that…", "One moment…") **at session start**, using the *same* voice/model as the lesson itself, so there is zero synthesis latency when the watchdog fires. Crossfade (~150–200ms) into the real answer the instant it lands.

### 5.4 Voice / model / language selection

Pull the exact model, voice, and language combination from **Rime's live catalog at submission time** — not a hardcoded speaker list copied earlier in development, which the PS explicitly warns against. Pick one model+voice pairing per supported language and test that exact combination in the exact flow used for the demo recording.

### 5.5 Transport

Stream via LiveKit's official Rime integration over WebRTC/Opus — not request-response file downloads. File-based TTS makes true low-latency interruption structurally impossible, since you can't "stop" a file that's already been fully generated and handed to the player.

### 5.6 Pronunciation

Build a fixture list from the chosen chapter's domain vocabulary — formulas, named entities, technical terms, numbers/codes if relevant. Render before/after variants using Rime's delivery/prompting controls, informed by the "Writing for the ear" guidance on punctuation and phrasing, and keep the resulting audio clips as evidence.

### 5.7 Fallback disclosure

If a fallback TTS provider is added for resilience, the active provider must be visibly observable (in UI or logs) at all times, and Rime must remain the default path in whatever flow is captured for judging.

---

## 6. Acceptance Test (Define Before Building the Demo)

This is the literal test to script, run, and record evidence for — not an afterthought written after the demo.

**Claim:** *When interrupted mid-sentence — including when the interrupting question requires a live web search — the tutor stops audio in under 200ms, fills silence naturally if the answer takes time, answers correctly, and resumes the lesson from the correct point without repeating or skipping content.*

**Procedure:**
1. Start teaching a chosen chapter segment.
2. Artificially inject a fixed delay (e.g. 2–3s) into the web-search tool call.
3. Mid-sentence, interrupt with a question that requires that web search.
4. Measure and log:
   - Time from user speech onset to audio playback stopping.
   - Time from question end to gap-filler audio starting.
   - Correctness of the final spoken answer.
   - Whether `resume_controller` chose the correct strategy and the lesson resumed without repetition or a skipped sentence.
5. **Stress variant:** interrupt a second time, with a different question, before the first QA answer resolves. Verify the first (now-stale) answer is never spoken — this is the direct test of `turn_id` fencing.
6. Repeat the full procedure once in the secondary language (e.g. Hindi) to demonstrate multilingual routing doesn't break any of the above.
7. Record cold-run and warm-run numbers **separately** — unverified or unlabeled performance numbers receive no credit per the PS's rules.

**Results table template:**

| Run | Audio-stop latency | Gap-filler trigger latency | Time to correct answer | Resume correctness (Y/N) |
|---|---|---|---|---|
| Cold | | | | |
| Warm | | | | |
| Stress (double-interrupt) | | | | |
| Hindi run | | | | |

---

## 7. README & Evidence Deliverables

### 7.1 `README.md` — required sections
1. Title + one-paragraph pitch (who, problem, why voice, which hard voice problem)
2. Demo (link + timestamp index, screenshot of active-provider indicator)
3. Architecture overview (the 3-layer diagram + one paragraph per layer)
4. The hard voice problem + acceptance test (claim, procedure, results table — cross-reference `RIME_EVIDENCE.md`, don't duplicate)
5. Feature list (each tagged with the responsible agent)
6. Setup instructions (prerequisites, env vars as placeholders only, run commands, how to run the acceptance test script)
7. **Exact Rime configuration used**: model ID, speaker(s), language(s), endpoint/region, audio format, transport — stated literally, plus the note "current as of live catalog on [submission date]"
8. Third-party services table (service → purpose → where credentials live)
9. Multilinguality & pronunciation (languages supported, detection/routing method, code-switching limits, fixture list link)
10. Known limitations & failure behavior (STT failure, search API failure, unhandled edge cases, untested language pairs, fallback TTS disclosure)
11. Repository structure
12. Credential/configuration hygiene statement (no secrets committed anywhere, including recordings/screenshots)

### 7.2 `RIME_EVIDENCE.md` — required companion file
- The hard voice claim, verbatim, matching README §4
- Full acceptance test procedure and exact reproduction commands/scripts
- Raw results (not just the summary table) — cold/warm labeled, sample size stated
- Pronunciation before/after evidence method
- Limitations specific to the evidence itself (e.g. small sample size caveats)
- If pursuing the benchmark path: corpus, exact configs, generated clips, item-level results, analysis code, and blinding method per the PS's benchmark rules

---

## 8. Repository Structure

```
/agents/            LangGraph nodes: master, qa, resume_controller, gap_filler, ingest
/realtime/          LiveKit workers, VAD wiring, streaming STT integration
/tts/               Rime client, streaming playback + crossfade + cancellation logic
/tests/             acceptance test scripts, double-interrupt stress test harness
/fixtures/          pronunciation test sentences, before/after audio clip pairs
/docs/              RIME_EVIDENCE.md, architecture diagram source
.env.example
README.md
```

---

## 9. Build Timeline (Time-Boxed)

| Phase | Work |
|---|---|
| **Day 1 AM** | LiveKit + Rime streaming "hello world" — single hardcoded sentence loop, no LangGraph yet. |
| **Day 1 PM** | Barge-in detection + instant audio-buffer flush + checkpoint save/restore. **This is the highest-risk, highest-weighted piece — do not proceed until it is solid.** |
| **Day 2 AM** | LangGraph skeleton: `ingest_material → teach_step → await_event`, with checkpointing. Prove clean resume-from-checkpoint before adding QA. |
| **Day 2 PM** | Add `handle_interrupt → classify_intent → qa_retrieve / web_search_node → compose_answer → fence_check → resume_controller`. |
| **Day 3 AM** | Add gap-filler watchdog; run the double-interrupt stress test until fencing is airtight. Add multilingual routing + pronunciation fixtures. |
| **Day 3 PM** | Run full acceptance test (cold + warm + stress + Hindi run), write `RIME_EVIDENCE.md` and `README.md`, record the 4–5 min demo, do a final credential/config hygiene pass. |

---

## 10. Eligibility Checklist (Self-Audit Before Submission)

- [ ] Rime integration is verifiable in the submitted code, not just referenced.
- [ ] Rime is used for the *primary* spoken output throughout — not just a welcome message or final confirmation.
- [ ] A working product path exists (not static screens or a scripted mock).
- [ ] The 4–5 minute demo is included and covers: user/problem, normal flow, the chosen hard voice problem, one deliberate stress case, the result/measurement, and which speech provider is active.
- [ ] No live credential or secret appears anywhere — code, docs, screenshots, or recordings.
- [ ] The exact model/voice/language combination used has been tested against the event's preflight check.
- [ ] Cold and warm performance numbers are labeled separately; no unverified numbers are claimed.
