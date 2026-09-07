# v-tutor — Agent Layer Plan

> **Status (2026-09-07):** built and passing — `agents/` (state, session, llm,
> intent, material, retrieval, nodes, graph), 72 offline tests, and
> `scripts/text_harness.py`. LLM providers are stubbed until the model choice
> is made; web search is stubbed until DuckDuckGo is wired. Audio layer next.

LangGraph, one thread per session, checkpointed. The graph's job is **not** to be
clever. It is to guarantee one thing: *what the learner hears always matches the
conversational state the system believes it is in.* Everything else — reading,
answering, navigating — is built on top of that guarantee, never around it.

---

## 0. How this layer answers the problem statement

The PS asks for one hard voice problem, proven under real conditions. Ours is
**interruption and recovery**, and the PS gives the exact test we must pass
(page 4, "Full-duplex test example"). Each clause maps to a specific piece of
this graph:

| PS requirement | Where it is solved |
|---|---|
| "Introduce a fixed delay into a tool call" | `web_search_node` has an injectable delay (`STRESS_DELAY_MS` env var) |
| "While the agent is speaking or waiting, interrupt it" | `await_event` accepts `user_barge_in` at any time, including mid-tool-call |
| "Queued Rime audio stops promptly" | VAD callback → `clear()` + `flush()` **before** the graph is even scheduled (see §3) |
| "The updated instruction reaches the application" | `handle_interrupt` → `classify_intent` on the new utterance |
| "Stale tool results are not spoken as current" | `fence_check` routing on `born_turn_id != turn_id` |
| "Background work is cancelled or reconciled" | Stale branch routes to `END`; result is never written to `answer` |
| "Final spoken response reflects what the user actually heard" | `heard_cursor` frozen at interrupt, consumed by `resume_controller` |
| "Continue accepting user audio while Rime speech is playing" | `await_event` is an `interrupt()` pause, not a blocking call; audio path never depends on the graph |
| "Select compatible models, voices, and language settings deliberately" (multilingual) | `choose_language` picks `speaker` from `config.LANG_SPEAKER`, which `scripts/preflight.py` validates against the live catalog for every supported language, not just the demo default |
| "Use synthetic or de-identified data" | Lessons come from public Wikipedia text or the learner's own PDF; no student records are involved |

The PS also scores *evidence*: the fencing test in §7 needs no API keys and
runs in under a second, so judges can re-run it themselves.

---

## 1. State

**File:** `agents/state.py`

```python
from typing import Annotated, Literal, TypedDict
import operator

Intent = Literal["question", "explain", "navigate", "command",
                 "session", "backchannel", "unknown"]

class Cursor(TypedDict):
    beat_index: int      # which lesson beat
    sentence_index: int  # which sentence within the beat
    word_index: int      # last word the learner actually heard
    char_offset: int     # for resuming mid-beat text
    seconds: float

class TutorState(TypedDict):
    session_id: str

    # ---------- onboarding ----------
    onboarding_step: Literal["language", "source", "done"]
    source_kind: Literal["pdf", "topic", None]
    pdf_paths: list[str]            # uploaded files, if any
    topic: str | None               # "photosynthesis"
    grade: str | None               # "class 6"
    source_lang: str | None         # language the PDF is written in (detected)

    # ---------- material ----------
    material_chunks: list[dict]     # {id, text, embedding, section_id}
    lesson_plan: list[dict]         # ordered beats: {id, title, text, section_id}
    beat_index: int

    # ---------- FENCING — the core invariant ----------
    turn_id: int                    # monotonic; += 1 on every barge-in
    born_turn_id: int               # turn_id the current branch started under
    heard_cursor: Cursor            # frozen at interrupt time
    heard_sentence: str             # sentence being spoken at interrupt
    pending_text: str               # what we last asked Rime to speak

    # ---------- current interrupt turn ----------
    user_utterance: str
    detected_lang: str
    intent: Intent
    command: Literal["repeat", "slower", "faster", "switch_lesson_lang", None]
    session_cmd: Literal["pause", "continue", "restart", "quit", None]
    nav_target: dict | None         # {"kind": "prev"|"next"|"topic"|"index", "value": ...}
    queued_request: str | None      # second clause of "do A and B"
    recent_exchanges: list[dict]    # last 2 {q, a}; context for "and why?"
    retrieved: list[dict]
    retrieval_score: float
    answer: str
    reply_lang: str | None          # one-off language for this reply only

    # ---------- session / delivery ----------
    paused: bool
    active_lang: str                # BCP-47; language of the lesson AND of Rime
    speed_alpha: float
    speaker: str                    # Rime voice, chosen from LANG_SPEAKER map
                                    # for active_lang, validated against the
                                    # live catalog at startup

    transcript: Annotated[list[dict], operator.add]   # append-only audit log
```

### The two fields that make fencing work

`turn_id` is bumped by `handle_interrupt`. `born_turn_id` is stamped onto a
branch when it starts. If a slow `web_search_node` from turn 7 finishes after
turn 8 has begun, the branch still carries `born_turn_id == 7`, and
`fence_check` sees `7 < 8` and discards it. One integer comparison is the whole
mechanism.

### One rule, held everywhere

**No node may speak. Only `resume_controller` emits text to Rime.** Every other
node returns state. This gives one choke point where fencing is enforced instead
of a dozen places to forget it.

---

## 2. Nodes

**File:** `agents/nodes.py`

### Onboarding — runs once, entirely by voice

The learner never touches a settings screen. Two spoken questions, then the
lesson starts. Both questions go through the same `await_event` → barge-in path
as everything else, so onboarding is not a special mode — it is the normal graph
with `onboarding_step != "done"`.

| Node | Reads | Writes | Behaviour |
|---|---|---|---|
| `choose_language` | `user_utterance`, `detected_lang` | `active_lang`, `speaker`, `onboarding_step` | Asks *"Which language shall we study in? You can say it, or tap it on the screen."* — spoken in **English only**; the screen shows language buttons alongside. A button press enters the graph as if the learner had said the language name (`TutorRunner.press_language_button`). If a button was tapped *before* the session started, `preset_lang` skips this question entirely. Speech answers work by name ("Hindi", "हिंदी") or by inference from `detected_lang`. Looks up `speaker` in `config.LANG_SPEAKER[active_lang]`. From here on every Rime call uses this language and voice — this is the **only** screen interaction in the product. |
| `choose_source` | `user_utterance`, `pdf_paths` | `source_kind`, `topic`, `grade`, `onboarding_step` | If files were uploaded before the session, confirms them aloud and sets `source_kind="pdf"`. Otherwise asks *"What shall we study, and which class?"* — parses topic + grade from the answer ("photosynthesis for class six"). Asks once more if either is missing. |
| `parse_pdf` | `pdf_paths` | `raw_text`, `source_lang` | Text extraction, headings preserved as section markers, tables flattened to sentences, figures dropped. Detects the document's language. |
| `fetch_material` | `topic`, `active_lang` | `raw_text`, `source_lang`, `source_url` | **Fetches real text; does not write it.** Wikipedia REST API on the `active_lang` edition (`hi.wikipedia.org` for Hindi), so no translation is needed when an article exists. Search → best-matching page → sections as plain text, headings kept, infoboxes and references dropped. If the target-language edition has no article, falls back to English and lets `ingest_material` translate. If nothing is found at all, asks aloud for a different topic. The URL is kept so answers can say where a fact came from. |
| `ingest_material` | `raw_text`, `source_lang`, `active_lang`, `grade` | `material_chunks`, `lesson_plan` | Split into sections → chunk (2–4 sentences, 1 overlap) → embed → beats of 1–2 sentences. **One batched LLM pass per section does two jobs at once:** translate if `source_lang != active_lang`, and simplify to `grade` level if set (Wikipedia is not written for class 6). Runs once at ingest, so there is zero per-sentence latency later. Retrieval chunks keep the *original* text for factual precision; beats carry the spoken version. Every chunk and beat keeps its `section_id`. |

### Lesson loop

| Node | Reads | Writes | Behaviour |
|---|---|---|---|
| `teach_step` | `lesson_plan`, `beat_index`, `heard_cursor`, `paused` | `pending_text`, `beat_index` | Emits the next beat. If `heard_cursor` is set, starts from the sentence boundary at or before it. If `paused`, emits nothing. |
| `await_event` | — | — | `interrupt()`. The graph parks here. Resumed by `main.py` with one of: `playback_confirmed`, `user_barge_in`, `lesson_complete`. |

### Interrupt path

| Node | Reads | Writes | Behaviour |
|---|---|---|---|
| `handle_interrupt` | `turn_id`, player cursor | `turn_id`, `born_turn_id`, `heard_cursor`, `heard_sentence` | **Pure and instant.** No I/O, no LLM. Records where the learner stopped hearing. Does *not* perform the stop — see §3. |
| `classify_intent` | `user_utterance`, `recent_exchanges`, `paused` | `intent`, `command`, `session_cmd`, `nav_target`, `queued_request` | Rules first (pause / continue / slower / again / mm-hmm are regex-cheap), small fast model for the rest. Compound utterances: classify the first clause, park the second in `queued_request`. Latency here delays the *answer*, never the *stop*. |
| `clarify` | — | `answer` | "Sorry — I didn't catch that. Say it again?" Asks once. A second consecutive `unknown` resumes the lesson instead of looping. |
| `session_handler` | `session_cmd` | `paused`, `beat_index`, `heard_cursor` | `pause` sets `paused=True`. `continue` clears it. `restart` zeroes `beat_index` and cursor. `quit` ends. Questions asked while paused are still answered; the lesson just doesn't auto-advance. |
| `command_handler` | `command` | `speed_alpha`, `active_lang`, `pending_text`, `heard_cursor` | No LLM. `repeat` re-emits `pending_text`. `slower`/`faster` step `speed_alpha` (direction verified per model in Phase 0 — see ARCHITECTURE §5). `switch_lesson_lang` changes `active_lang` and rewinds `heard_cursor` to the start of the current sentence so no sentence is half in each language. |
| `find_section` | `nav_target`, `lesson_plan` | `beat_index`, `heard_cursor` (reset) | `prev` / `next` / `index` move the pointer. `topic` embeds the phrase, picks the nearest section, jumps to its first beat. Flows straight into `teach_step`. |
| `explain` | `heard_sentence`, `user_utterance`, `recent_exchanges` | `answer`, `reply_lang` | Define / simplify / translate **the sentence the learner just heard**. Skips retrieval — context is already in hand. "In Spanish" sets `reply_lang` for **this reply only**: `resume_controller` speaks it with that language's coda voice (`config.LANG_SPEAKER`, all eight verified languages), then the bridge and the lesson continue in the study language and voice. Asking to teach the *whole lesson* in a non-study language gets a polite refusal. |
| `qa_retrieve` | `user_utterance`, `material_chunks` | `retrieved`, `retrieval_score` | Whole-document **hybrid** search: vector cosine + BM25, scores fused, top-k 4. Keyword side catches names, numbers, dates that embeddings blur. Below τ the question is not in the notes and goes to `direct_answer` (or straight to web when no real model is configured). |
| `direct_answer` | `user_utterance`, `recent_exchanges` | `answer`, `answer_mode` | Out-of-notes questions are usually trivial for the strong model ("capital of France", "what does chlorophyll mean"), so it is asked first, ~0.5 s. It replies the single token `LOOKUP` when the question needs current or very specific information; only then does `web_search_node` run. Fenced via `fence_direct`. |
| `web_search_node` | `user_utterance` | `retrieved` | Out-of-syllabus fallback after `LOOKUP`. **Honours `STRESS_DELAY_MS`** — this is the deliberately delayed tool call in the PS test. |
| `compose_answer` | `retrieved`, `recent_exchanges`, `active_lang`, `reply_lang` | `answer`, `recent_exchanges` (append) | Streams tokens. Writes for the ear: short sentences, no lists, no markdown. Sees the last two exchanges so "and why?" resolves. If the answer came from the notes, may end with "that's in the part about X — want me to go back there?" |
| `resume_controller` | `answer`, `heard_cursor`, `intent`, `queued_request` | text → Rime | **The only speaking node.** Speaks `answer` (if any), then a bridge, then resumes per §4. Afterwards drains `queued_request` if set. |

### `fence_check` is a router, not a node

```python
def fence_check(state) -> Literal["current", "stale"]:
    return "current" if state["born_turn_id"] == deps.clock.current() else "stale"
```

It compares against the **live** `TurnClock`, not a copy in state: the clock is
bumped by the realtime layer while a node may still be running, and graph state
cannot change mid-node. Making the check a routing function means staleness is a
structural property of the graph. A new node cannot be wired to speech without
passing through it. Stale branches go to a tiny `discard` node (counts the drop
for evidence) and then back to `await_event`, so the graph is always parked and
ready for the utterance that caused the interruption.

### `_say()` is the single choke point in code

`resume_controller` is the only node that speaks *answers*, but `teach_step`
(lesson text) and `session_handler` (pause/quit acknowledgements) also produce
speech. All three call one helper, `TutorNodes._say()`, which re-checks the
fence immediately before handing text to the speaker. So even a branch that
passed `fence_check` a few milliseconds earlier cannot speak if the learner has
interrupted since.

### `promote_queued` drains the pocket

"Go back to chambers *and explain it simpler*" is one utterance with two
intents. `classify_intent` handles the first and parks the second in
`queued_request`. After the first is done, whichever node finished
(`resume_controller` or `teach_step`) routes to `promote_queued`, which makes
the second clause the current utterance and refreshes `heard_sentence` to
*what was just spoken*, so "it" refers to the section the tutor just read.

---

## 3. The stop happens *outside* the graph

This is the design decision the whole submission rests on.

The PS says queued audio must stop *promptly*. LangGraph scheduling a node takes
tens of milliseconds at best and is not on any latency guarantee. So the kill
path does **not** go through the graph:

```
Silero VAD  ──speech_start──▶  main.py callback
                                  ├─▶ rime.clear(context_id)     # server-side queue
                                  ├─▶ player.flush()             # local buffer
                                  └─▶ graph.resume("user_barge_in", cursor=player.heard_cursor())
```

The first two lines are the stop and run in well under the ~120 ms VAD budget.
The third line *tells the graph what just happened*. `handle_interrupt` then
records the cursor into state. The graph reacts to the stop; it never causes it.

Consequence for testing: the fencing test in §7 can drive the graph directly
with fake events and never needs audio, VAD, or Rime.

---

## 4. `resume_controller` — replay or continue

After answering, the tutor has to get back into the lesson. Rule table:

| Situation | Behaviour |
|---|---|
| Interrupted within the first 3 words of a beat | Replay the whole beat, no bridge |
| Interrupted mid-sentence | Bridge, then replay from the start of that sentence |
| Interrupted exactly at a sentence boundary | Bridge, then continue with the next sentence |
| `intent == backchannel` ("mm-hmm") | Continue from `heard_cursor`'s sentence, **no bridge** |
| `intent == clarify` | Speak the clarify line, then wait — do not resume the lesson |
| `paused == True` | Speak the answer only. Do not resume. |

Bridges are short, fixed phrases rendered once at build time with the same voice
and model as the live path: *"So — back to where we were."* *"Right, carrying
on."* Never resume mid-word; always round back to a sentence boundary at or
before `heard_cursor`.

---

## 5. Gap filler

**File:** `agents/gap_filler.py`

Dead air during `web_search_node` is the second most visible voice failure after
a slow stop. The watchdog arms when a slow node is entered; if no audio has been
rendered within 700 ms it plays a pre-rendered clip ("Let me check that…"); it
disarms the instant real audio starts.

**Pre-rendered is the point.** Generating the filler through the same Rime call
you are already waiting on adds latency rather than hiding it.

**Fencing applies here too.** A filler carries the `turn_id` it was armed under
and is dropped if the learner has interrupted since. Otherwise the tutor says
"let me check that" about a question the learner abandoned two turns ago.

---

## 6. Graph

**File:** `agents/graph.py`

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

g = StateGraph(TutorState)

for name, fn in [
    ("ingest_material", ingest_material), ("teach_step", teach_step),
    ("await_event", await_event),         ("handle_interrupt", handle_interrupt),
    ("classify_intent", classify_intent), ("clarify", clarify),
    ("session_handler", session_handler), ("command_handler", command_handler),
    ("find_section", find_section),       ("explain", explain),
    ("qa_retrieve", qa_retrieve),         ("web_search_node", web_search_node),
    ("compose_answer", compose_answer),   ("resume_controller", resume_controller),
]:
    g.add_node(name, fn)

for name, fn in [
    ("choose_language", choose_language), ("choose_source", choose_source),
    ("parse_pdf", parse_pdf),             ("fetch_material", fetch_material),
]:
    g.add_node(name, fn)

# --- onboarding: two spoken questions, then material ---
g.set_entry_point("choose_language")
g.add_conditional_edges("choose_language", route_onboarding, {
    "ask":  "resume_controller",     # speak the question, then wait for the answer
    "next": "choose_source",
})
g.add_conditional_edges("choose_source", route_onboarding, {
    "ask":   "resume_controller",
    "pdf":   "parse_pdf",
    "topic": "fetch_material",
})
g.add_conditional_edges("parse_pdf", route_parse, {
    "ok":       "ingest_material",
    "unreadable": "choose_source",   # scanned PDF: ask for a topic instead
})
g.add_conditional_edges("fetch_material", route_fetch, {
    "ok":        "ingest_material",
    "not_found": "choose_source",    # ask for a different topic
})
g.add_edge("ingest_material", "teach_step")
g.add_edge("teach_step", "await_event")

# --- events from the realtime layer ---
g.add_conditional_edges("await_event", route_event, {
    "playback_confirmed": "teach_step",
    "user_barge_in":      "handle_interrupt",
    "stay_parked":        "await_event",     # paused: loop back and wait again
    "lesson_complete":    END,
})
# During onboarding the learner's reply is an answer, not an interruption.
g.add_conditional_edges("handle_interrupt", route_after_interrupt, {
    "onboarding_language": "choose_language",
    "onboarding_source":   "choose_source",
    "lesson":              "classify_intent",
})

# --- seven intents ---
g.add_conditional_edges("classify_intent", route_intent, {
    "unknown":     "clarify",
    "session":     "session_handler",
    "command":     "command_handler",
    "navigate":    "find_section",
    "explain":     "explain",
    "question":    "qa_retrieve",
    "backchannel": "resume_controller",
})

g.add_conditional_edges("session_handler", route_session, {
    "pause":    "await_event",
    "continue": "teach_step",
    "restart":  "teach_step",
    "quit":     END,
})
g.add_edge("find_section", "teach_step")

g.add_conditional_edges("qa_retrieve", route_retrieval, {
    "grounded":  "compose_answer",
    "needs_web": "web_search_node",
})
g.add_edge("web_search_node", "compose_answer")

# --- every path that produces speech passes the fence ---
FENCED = {"current": "resume_controller", "stale": END}
for node in ("compose_answer", "command_handler", "explain", "clarify"):
    g.add_conditional_edges(node, fence_check, FENCED)

# --- after speaking, drain a queued second request before waiting ---
g.add_conditional_edges("resume_controller", route_after_speak, {
    "queued": "classify_intent",
    "done":   "await_event",
})

app = g.compile(checkpointer=SqliteSaver.from_conn_string("sessions.db"))
```

### Routers

```python
def route_event(state):
    ev = state["_event"]                 # injected by main.py on resume
    if ev == "user_barge_in":   return "user_barge_in"
    if ev == "lesson_complete": return "lesson_complete"
    if state["paused"]:         return "stay_parked"   # ignore playback_confirmed
    return "playback_confirmed"

def route_after_interrupt(state):
    step = state["onboarding_step"]
    if step == "language": return "onboarding_language"
    if step == "source":   return "onboarding_source"
    return "lesson"

def route_onboarding(state):
    # Each onboarding node sets `answer` to a question when it still needs
    # input, and clears it when the step is complete.
    if state.get("answer"):            return "ask"
    if state["onboarding_step"] == "source":
        return "next"
    return state["source_kind"]        # "pdf" | "topic"

def route_parse(state):  return "ok" if state.get("raw_text") else "unreadable"
def route_fetch(state):  return "ok" if state.get("raw_text") else "not_found"

def route_intent(state):    return state["intent"]
def route_session(state):   return state["session_cmd"]

def route_retrieval(state):
    return "grounded" if state["retrieval_score"] >= TAU else "needs_web"

def route_after_speak(state):
    return "queued" if state.get("queued_request") else "done"
```

`route_event` reads `paused`: when paused, `playback_confirmed` does **not**
advance to `teach_step`; the graph stays parked. That single check is the entire
pause implementation, kept out of every other node.

---

## 7. Tests — what judges can re-run

**Directory:** `tests/` — 135 tests, all offline, ~4 s.

```
v-tutor/Scripts/python -m pytest tests -q
v-tutor/Scripts/python scripts/text_harness.py                       # offline fixture
v-tutor/Scripts/python scripts/text_harness.py --live --web duckduckgo \
    --persist sessions.db --session demo --evidence evidence/demo.csv  # everything real
```

Extra test files from the gap-closing pass: `test_gaps.py` (bugs, writing for
the ear, lesson cap, two-stage preparation, retry, gap filler, evidence,
persistence), `test_pdf.py` (generated fixtures in `tests/fixtures/`),
`test_retrieval_hindi.py`.

In the harness, an empty line means "the tutor finished speaking"; `@12 what's
that` means "I interrupted after hearing 12 words". Internal events (interrupt,
intent, retrieve score, fence drops) print inline so you can watch the fence work.

| Test | Needs | Asserts |
|---|---|---|
| `test_fencing.py` | nothing | The PS recipe. A web search that gets interrupted mid-call is never spoken; `stale_drops` increments; the graph stays parked; the interrupting question is then answered normally. A control run proves the same result *is* spoken when not interrupted. With `stress_delay_ms` set, the tool call aborts early once the turn moves on. The same holds when the interruption lands inside the explain LLM call. **The highest-value file in the repo.** |
| `test_routing.py` | nothing | 33 utterances (English and Hindi) → expected intent, command, session command, or navigation target. Compound splitting only when both halves classify. Topic/grade and language parsing. |
| `test_graph_flow.py` | nothing | Onboarding (language, topic, missing grade asked once, Hindi sets the Hindi voice); beat advance on `playback_confirmed`; pause blocks advance but answers questions; in-notes question answered from notes then lesson resumes from the interrupted sentence; out-of-notes goes to web; backchannel resumes without a bridge; slower changes `speed_alpha` in the configured direction; navigate by topic / prev / next; explain uses the heard sentence; compound request drained and refers to the section just read; unknown asks once then carries on; quit ends; lesson completes; lesson language switch changes voice. |
| `test_retrieval.py` | nothing | 10 in-notes questions (names, numbers, dates) hit the right section **and** score above τ; 3 out-of-notes questions score below τ. |

Still to write when the audio layer exists: a real-PDF ingest fixture, and an
English-PDF-taught-in-Hindi translation test (needs a non-stub LLM).

Build order: `state.py` → `test_fencing.py` (red) → stub `graph.py` (green) →
real nodes one at a time, retrieval first, LLM nodes last.

---

## 8. Decisions made so building can start

- **Two LLM roles, one adapter.** `agents/llm.py` exposes `fast()` for intent
  classification and `strong()` for answers. Provider is a single env setting so
  it can be swapped without touching nodes. Intent classification tries regex
  rules first and only calls `fast()` on a miss.
- **Retrieval store:** in-memory numpy cosine + `rank_bm25`. One document per
  session does not justify a vector database.
- **Chunking:** 2–4 sentences, 1 sentence overlap. Beats: 1–2 sentences.
- **τ:** 0.35 on the fused score. Fused = (0.7·BM25 + 0.3·dense) × (0.4 + 0.6·coverage)
  with the hash embedder (0.5/0.5 with fastembed). Query-side stopwords are
  removed and English is lightly stemmed; BM25 is normalised so that "every
  content word present once" scores ≈1.0; *coverage* is the fraction of the
  query's content words present in the chunk. On the fixture this puts in-notes
  questions at 0.5–0.9 and out-of-notes at 0.0–0.03. Re-tune against
  `test_retrieval.py` if the fixture or embedder changes.
- **Checkpointer:** SQLite. Postgres is a one-line swap if ever needed.
- **Supported study languages at launch:** English and Hindi. Both exist on
  `coda`. `config.LANG_SPEAKER` maps each to one pinned voice; adding a language
  is one line plus a preflight run. The onboarding question is spoken in both.
- **PDF parsing:** `pymupdf` for text + heading detection. Scanned PDFs (no text
  layer) are declined aloud: *"I can't read that file — could you tell me the
  topic instead?"* No OCR in scope.
- **Topic mode fetches, never writes.** Wikipedia REST API (`/page/summary` to
  confirm the match, `/page/mobile-sections` or `action=parse` for section text),
  target-language edition first. No API key, no rate-limit concerns at demo
  scale. Grade-level simplification happens in the ingest batch pass, not at
  fetch time.

---

## 9. Declared limitations

Written down deliberately — the PS rewards disclosed limits over hidden ones.

- Compound requests are handled two-deep. "A and B" works; "A, B and C" drops C.
- Follow-up memory is two exchanges. "And why?" works; "what did you say five
  questions ago?" does not.
- `clarify` asks once, then carries on. A tutor that keeps saying "pardon?" is
  worse than one that resumes.
- Resume is accurate to the sentence, not the word: we always round back to a
  sentence boundary because resuming mid-word sounds broken even when correct.
- One document per session. No memory between sessions. No quizzing.
- Two study languages at launch. Hindi has three `coda` voices, so the Hindi
  tutor sounds different from the English one — voice identity does not carry
  across a language switch.
- Translated lessons (English PDF taught in Hindi) are machine-translated at
  ingest. Technical terms may come out awkwardly; the learner can always ask
  "explain that word" to get the source-language term.
- Scanned or image-only PDFs are not read. Topic mode is offered instead.
- Topic mode is only as good as Wikipedia's coverage of the topic in the chosen
  language. Hindi Wikipedia is thinner than English; when it has no article we
  fall back to English and translate, and say so aloud.
