# v-tutor — Agentic Layer Implementation Plan

LangGraph, one thread per session, checkpointed. The graph's job is **not** to be
clever. It is to guarantee that what the learner hears always matches the
conversational state the system believes it is in.

---

## 1. State schema

**File:** `agents/state.py`

```python
from typing import Annotated, Literal, TypedDict
import operator

class Cursor(TypedDict):
    beat_index: int      # which lesson beat
    word_index: int      # last word the learner actually heard
    char_offset: int     # for resuming mid-sentence text
    seconds: float

class TutorState(TypedDict):
    session_id: str

    # ---------- material ----------
    material_chunks: list[dict]     # {id, text, embedding, source}
    lesson_plan: list[dict]         # ordered lesson beats
    beat_index: int

    # ---------- FENCING (the core invariant) ----------
    turn_id: int                    # monotonic; ++ on every barge-in
    heard_cursor: Cursor            # frozen at interrupt time
    pending_text: str               # what we asked Rime to speak

    # ---------- current interrupt turn ----------
    born_turn_id: int               # turn_id this branch started under
    user_utterance: str
    detected_lang: str
    intent: Literal["question", "command", "backchannel", "unknown"]
    command: Literal["repeat", "slower", "faster", "switch_lang", None]
    retrieved: list[dict]
    retrieval_score: float
    answer: str

    # ---------- delivery ----------
    active_lang: str                # BCP-47
    speed_alpha: float
    speaker: str

    transcript: Annotated[list[dict], operator.add]   # append-only audit log
```

### Why `born_turn_id` exists

It is the whole fencing mechanism in one field. When `handle_interrupt` bumps
`turn_id` to 8, any branch still executing from turn 7 carries
`born_turn_id == 7`. `fence_check` compares the two and discards the branch.
Without it, a slow tool call from a superseded turn gets spoken as current —
precisely the failure the PS's full-duplex test hunts for.

**Invariant to hold everywhere:** *no node may speak. Only `resume_controller`
emits text to Rime.* Every other node returns state. This gives exactly one
choke point where fencing is enforced, instead of N places to get it wrong.

---

## 2. Node contracts

**File:** `agents/nodes.py`

| Node | Reads | Writes | Notes |
|---|---|---|---|
| `ingest_material` | source docs | `material_chunks`, `lesson_plan` | Runs once at session start. Chunk → embed → outline. |
| `teach_step` | `lesson_plan`, `beat_index`, `heard_cursor` | `pending_text`, `beat_index` | Emits the next beat. On resume, starts from `heard_cursor.char_offset`. |
| `await_event` | — | — | `interrupt()`. Blocks until `playback_confirmed` or `user_barge_in`. |
| `handle_interrupt` | `turn_id`, player cursor | `turn_id += 1`, `heard_cursor` | **Must be pure and instant.** No I/O, no LLM. |
| `classify_intent` | `user_utterance` | `intent`, `command` | Small/fast model or rules. Latency here delays the *answer*, not the *stop*. |
| `command_handler` | `command` | `speed_alpha`, `active_lang`, `pending_text` | No LLM needed — `repeat` replays `pending_text`. |
| `qa_retrieve` | `user_utterance`, `material_chunks` | `retrieved`, `retrieval_score` | Cosine similarity vs threshold τ. |
| `web_search_node` | `user_utterance` | `retrieved` | Out-of-syllabus fallback. **The deliberately-delayed node in the stress test.** |
| `compose_answer` | `retrieved`, `active_lang` | `answer` | Streams tokens. Must write for the ear: short sentences. |
| `fence_check` | `born_turn_id`, `turn_id` | routing only | `born < current` → discard. |
| `resume_controller` | `answer`, `heard_cursor` | text → Rime | Replay vs continue. The only speaking node. |

### `handle_interrupt` must not touch the network

It runs on the ~120 ms budget. Its entire job:

```python
def handle_interrupt(state):
    return {
        "turn_id": state["turn_id"] + 1,
        "heard_cursor": player.heard_cursor(),   # local read, already computed
        "born_turn_id": state["turn_id"] + 1,
    }
```

The Rime `clear()` and player `flush()` are fired by the **VAD callback
directly**, not from inside this node — see the wiring note in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md#phase-5--realtime-loop-integration).
The node records the consequence; it does not perform the stop.

---

## 3. Graph wiring

**File:** `agents/graph.py`

```python
g = StateGraph(TutorState)

g.add_node("ingest_material", ingest_material)
g.add_node("teach_step", teach_step)
g.add_node("await_event", await_event)
g.add_node("handle_interrupt", handle_interrupt)
g.add_node("classify_intent", classify_intent)
g.add_node("command_handler", command_handler)
g.add_node("qa_retrieve", qa_retrieve)
g.add_node("web_search_node", web_search_node)
g.add_node("compose_answer", compose_answer)
g.add_node("resume_controller", resume_controller)

g.set_entry_point("ingest_material")
g.add_edge("ingest_material", "teach_step")
g.add_edge("teach_step", "await_event")

g.add_conditional_edges("await_event", route_event, {
    "playback_confirmed": "teach_step",
    "user_barge_in": "handle_interrupt",
    "lesson_complete": END,
})
g.add_edge("handle_interrupt", "classify_intent")
g.add_conditional_edges("classify_intent", route_intent, {
    "command": "command_handler",
    "question": "qa_retrieve",
    "backchannel": "resume_controller",
})
g.add_conditional_edges("qa_retrieve", route_retrieval, {
    "grounded": "compose_answer",      # similarity >= tau
    "needs_web": "web_search_node",    # similarity < tau
})
g.add_edge("web_search_node", "compose_answer")

# Fencing as routing, applied to every path that produces speech.
g.add_conditional_edges("compose_answer", fence_check,
                        {"current": "resume_controller", "stale": END})
g.add_conditional_edges("command_handler", fence_check,
                        {"current": "resume_controller", "stale": END})

g.add_edge("resume_controller", "await_event")

app = g.compile(checkpointer=SqliteSaver.from_conn_string("sessions.db"))
```

`fence_check` is a routing function rather than a node — it makes staleness a
structural property of the graph rather than an `if` somebody can forget.

---

## 4. `resume_controller` — replay vs continue

The learner interrupted mid-sentence. After answering, do we re-read the
sentence or pick up mid-word? Rule:

| Condition | Behaviour |
|---|---|
| Interrupted within first 3 words of a beat | Replay the whole beat |
| Interrupted mid-sentence, answer was short | Bridge + replay from sentence start |
| Interrupted mid-sentence, answer was long | Bridge + replay from sentence start (context is lost after a long detour) |
| Interrupted at a sentence boundary | Continue to the next sentence |
| `intent == backchannel` | Continue exactly from `heard_cursor`, no bridge |

Bridge phrases are short and pre-cached as audio (see gap-filler): *"So — back
to where we were."* Resuming mid-word is never correct; always round back to the
last sentence boundary at or before `heard_cursor`.

---

## 5. `gap_filler.py` — the watchdog

**File:** `agents/gap_filler.py`

Dead air during `web_search_node` is the second-most-visible voice failure after
slow barge-in. The watchdog:

1. Arms when a node is entered that may exceed ~700 ms.
2. If no audio has been rendered by the deadline, plays a **pre-synthesised**
   clip ("Let me check that…").
3. Disarms the instant real audio starts.

Pre-synthesised is the point: generating a filler through the same Rime call you
are already waiting on adds latency instead of hiding it. Render a small set at
build time into `assets/fillers/`, same speaker and model as the live path so the
voice identity stays consistent.

**Fencing applies here too.** A filler clip carries the `turn_id` it was armed
under and is dropped if the learner has since interrupted — otherwise you get
the absurd case of the tutor saying "let me check that" about a question that was
abandoned two turns ago.

---

## 6. Build order within Phase 4

Each step is independently testable without audio:

1. `state.py` + a pure-Python fencing unit test (no LLM, no network): simulate
   turn 7 branch completing after turn 8 starts, assert it is discarded.
2. `graph.py` skeleton with stub nodes returning canned state — verify routing
   with `app.invoke()` over scripted event sequences.
3. `qa_retrieve` + `ingest_material` against a fixed sample document.
4. `compose_answer` with a real model.
5. `gap_filler`, last — it is polish.

Step 1 is the highest-value test in the repo and needs no API keys, which makes
it the thing judges can actually re-run. Make it `pytest tests/test_fencing.py`.

---

## 7. Open questions to settle before coding

- **Which LLM** for `compose_answer` and `classify_intent`? They have very
  different latency budgets; a small fast model for intent and a stronger one for
  answers is the obvious split.
- **Vector store**: in-memory numpy cosine is almost certainly enough for one
  document and removes a dependency.
- **`lesson_plan` granularity**: a "beat" should be one or two spoken sentences.
  Longer beats make the heard-cursor more valuable but resumption clumsier.
