# v-tutor — Agent Layer Plan

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
    active_lang: str                # BCP-47; language of the lesson itself
    speed_alpha: float
    speaker: str

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

### Lesson loop

| Node | Reads | Writes | Behaviour |
|---|---|---|---|
| `ingest_material` | source document | `material_chunks`, `lesson_plan` | Runs once. Split into sections → chunk (2–4 sentences, 1 overlap) → embed → beats of 1–2 sentences each. Every chunk and beat keeps its `section_id` so navigation has targets. |
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
| `explain` | `heard_sentence`, `user_utterance`, `recent_exchanges` | `answer`, `reply_lang` | Define / simplify / translate **the sentence the learner just heard**. Skips retrieval — context is already in hand. "In Hindi" sets `reply_lang` for this reply only; the lesson language is untouched. |
| `qa_retrieve` | `user_utterance`, `material_chunks` | `retrieved`, `retrieval_score` | Whole-document **hybrid** search: vector cosine + BM25, scores fused, top-k 4. Keyword side catches names, numbers, dates that embeddings blur. Route to web only if *both* signals fall below τ. |
| `web_search_node` | `user_utterance` | `retrieved` | Out-of-syllabus fallback. **Honours `STRESS_DELAY_MS`** — this is the deliberately delayed tool call in the PS test. |
| `compose_answer` | `retrieved`, `recent_exchanges`, `active_lang`, `reply_lang` | `answer`, `recent_exchanges` (append) | Streams tokens. Writes for the ear: short sentences, no lists, no markdown. Sees the last two exchanges so "and why?" resolves. If the answer came from the notes, may end with "that's in the part about X — want me to go back there?" |
| `resume_controller` | `answer`, `heard_cursor`, `intent`, `queued_request` | text → Rime | **The only speaking node.** Speaks `answer` (if any), then a bridge, then resumes per §4. Afterwards drains `queued_request` if set. |

### `fence_check` is a router, not a node

```python
def fence_check(state) -> Literal["current", "stale"]:
    return "current" if state["born_turn_id"] == state["turn_id"] else "stale"
```

Making it a routing function means staleness is a structural property of the
graph. A new node cannot be wired to speech without passing through it.

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

g.set_entry_point("ingest_material")
g.add_edge("ingest_material", "teach_step")
g.add_edge("teach_step", "await_event")

# --- events from the realtime layer ---
g.add_conditional_edges("await_event", route_event, {
    "playback_confirmed": "teach_step",
    "user_barge_in":      "handle_interrupt",
    "stay_parked":        "await_event",     # paused: loop back and wait again
    "lesson_complete":    END,
})
g.add_edge("handle_interrupt", "classify_intent")

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

**Directory:** `tests/`

| Test | Needs | Asserts |
|---|---|---|
| `test_fencing.py` | nothing (stub nodes) | Turn 7's `web_search_node` result arriving after turn 8 started is routed to `END`, and `answer` is never written. **The highest-value test in the repo.** |
| `test_routing.py` | nothing | Every intent reaches the expected node; every speaking path passes `fence_check`; `paused` blocks auto-advance but not barge-ins. |
| `test_resume_rules.py` | nothing | Each row of the §4 table produces the expected resume text. |
| `test_retrieval.py` | sample document, embeddings | A fixture of 20 questions (10 factual with names/numbers, 10 conceptual) hits the right section; web fallback triggers only for the 3 out-of-syllabus questions. |
| `test_compound.py` | nothing | "Go back to cells and explain it simpler" → `navigate` then `explain`, in that order, on one interruption. |

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
- **τ:** start at 0.35 fused score; tune against `test_retrieval.py`, commit the
  fixture and the number.
- **Checkpointer:** SQLite. Postgres is a one-line swap if ever needed.

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
