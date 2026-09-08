# From sound to decision — the STT layer and what happens next

What the tutor does between "the learner made a noise" and "the tutor does
something about it". Two questions are answered here:

1. What `stt/` produces, and how fast.
2. Once a transcript exists, **what is decided about it, and in what order**.

Companion docs: [ARCHITECTURE.md](ARCHITECTURE.md) for the layering,
[AGENTS_PLAN.md](AGENTS_PLAN.md) for the graph, [VOICE_PIPELINE.md](VOICE_PIPELINE.md)
for how to run it.

---

## 1. The STT stage

```
mic / LiveKit track (48 kHz Opus)
   │
   ├─ stt/audio.py          frames -> 16 kHz mono float32 (LiveKit's SoX resampler)
   │
   ├─ stt/vad.py            Silero VAD
   │    ├── START_OF_SPEECH ──► bridge.on_speech_start()   fast path, ~0.1 ms
   │    └── END_OF_SPEECH   ──► SpeechUtterance(frames, duration, end_time)
   │
   └─ stt/transcriber.py    faster-whisper, one encoder pass
        └── (text, language, probability, whisper_ms)
```

Two paths leave the VAD and they are deliberately different speeds:

| path | trigger | latency | what it does |
|---|---|---|---|
| fast | START_OF_SPEECH | ~0.1 ms | stop the audio; no text exists yet |
| slow | END_OF_SPEECH | 0.5–2.5 s | transcribe, then decide what was meant |

### Latency, measured

`scripts/bench_stt.py` A/Bs the settings in separate processes and checks the
fast path against plain faster-whisper. On this laptop (Ryzen 7 5800H, 8
physical cores, `int8`):

| model | per utterance (1.3–8.6 s of speech) |
|---|---|
| `base`, beam 5, two encoder passes (before) | 850–990 ms |
| `base`, beam 1, one encoder pass, 8 threads | **430–560 ms** |
| `small`, same settings | ~1.8 s |

Three things bought that:

1. **One encoder pass.** `WhisperModel.transcribe(language=None)` runs the
   encoder twice — once inside `detect_language()`, then again in
   `generate_segments()`, discarding the first result. For a short utterance
   the encoder is ~60% of the call. `_single_encode()` encodes once, reads the
   language off that output, and hands the same output to the decoder.
   Language detection is unchanged: same model, same features, same
   probabilities. Any failure falls back permanently to the public API, which
   is only slower — never wrong. `WHISPER_SINGLE_PASS=0` disables it.
2. **`cpu_threads` = physical cores.** CTranslate2's default spreads over
   logical cores and loses ~25% to SMT contention.
3. **Greedy decoding, no timestamp tokens, no cross-window prompt.**

`base` is fast but mishears: "the heart" → "The Hard", "class six" →
"classics"/"Plastics", "hi" → "はい". Use `WHISPER_MODEL_SIZE=small` for
anything that matters; it is still faster than `base` was before this work.

### The default ear is now Whisper large-v3-turbo on Groq (2026-09-09)

A live session with a laptop mic showed what `base` does to the words a lesson
hangs on: "photosynthesis for class 6" came back as "Porto's & This is for
Class 6th" and "Autos and this is for last 6th", "English" as "Fuck.", and
"ask me the language again" as a topic that Wikipedia resolved to Yo-Yo Ma.
Local `small` fixes some of it at ~1.8 s. `whisper-large-v3-turbo` on Groq
fixes it at 0.3–0.6 s over a normal connection, so `stt/cloud.py` is the
default whenever `GROQ_API_KEY` is set (`STT_PROVIDER=auto`). Every request
carries a vocabulary hint (`STT_PROMPT`) with the words learners actually say
to a tutor. The local model stays loaded and answers any utterance whose
network call fails or times out (`STT_CLOUD_TIMEOUT_S`, 6 s), so a dropped
connection degrades to `base`, never to silence. `STT_PROVIDER=local` keeps
audio on the machine. A/B on eight Rime-rendered learner lines: same latency
band (290–560 ms both), cloud fixed the one line `base` misheard.

### Language handling

Spoken Hindi is acoustically almost identical to Urdu, and Whisper regularly
labels it `ur` and writes the transcript in Arabic script. That transcript then
goes to retrieval, to a web search, and comes back as a paragraph of Urdu prose
read out by an English voice.

`WHISPER_ALLOWED_LANGUAGES` (default `en,hi`) is the set the tutor can teach.
Anything else is decoded again as its alias (`ur`, `pa`, `ne`, `mr`, `sa` →
`hi`), or `WHISPER_LANGUAGE_FALLBACK`. Because the correction happens *before*
decoding and the encoder output is reused, it costs nothing.

### What the transcript is written to

Every utterance is appended to `stt/transcripts.csv` by `stt/transcripts.py` —
the same file, same columns, whichever entrypoint is running:

```
timestamp, participant, language, language_confidence,
vad_duration_sec, whisper_ms, total_latency_ms, transcript
```

`total_latency_ms` is what the learner waits: END_OF_SPEECH to transcript in
hand. `whisper_ms` is inference alone; the gap between them is buffering and
orchestration, normally a few ms. Follow a live session with
`Get-Content -Wait stt\transcripts.csv`. `LOG_TRANSCRIPTS=0` turns it off.

---

## 2. Between STT and the graph: the bridge filters

`voice/bridge.py` sits between the transcript and the tutor, and can drop or
delay an utterance before any classification happens.

### Self-echo (no headphones)

There is no acoustic echo cancellation on the local path. Without headphones
the mic hears the tutor, Whisper transcribes it, and the tutor answers its own
voice. One real session collapsed in four turns — "Let me check that." came
back as "Check that.", "I'm not sure." as "sure." — and ended up teaching an
article about a musician called Dean Blunt.

A transcript that is a **contiguous run of words the tutor said in the last 8
seconds** is treated as nothing heard. Exceptions, both learned the hard way:

- Utterances the rules recognise as an **instruction** (continue / stop /
  pause / repeat / slower) are never dropped. The tutor says "say continue when
  you're ready", the learner says "Continue." — that is an invited reply, not
  an echo. Dropping it left the lesson stuck.
- **Backchannels** ("sure", "okay") *may* be dropped, because an echoed "sure"
  and a real one both mean "carry on".

### Duplex mode

| mode | barge-in | when |
|---|---|---|
| full (default) | instant, ~0.2 ms | headphones |
| half | deferred, ~1 Whisper pass | `--no-headphones`, `HALF_DUPLEX=1`, or after 2 multi-word echoes |

In half duplex the VAD start does **not** stop playback — otherwise every beat
interrupts itself. The words still go to Whisper; if they survive the echo
guard the stop happens then. An interruption costs ~2 s instead of a
millisecond, which is the price of not wearing headphones. Utterances are
*deferred*, never discarded: the first version of this dropped them, which is
worse than the echo it was preventing.

Only multi-word echoes count towards the automatic switch. Disabling barge-in
for a whole session on the strength of one word is too aggressive.

---

## 3. What is decided about a transcript, and in what order

`agents/intent.py` classifies. **Rules first, model second, heuristic last** —
the common utterances ("slower", "again", "pause", "mm-hmm") are short,
closed-class, and must resolve in microseconds.

```
transcript
  │
  ├─ empty?  ────────────────────────────────► backchannel   (a cough; resume)
  ├─ "I have a question" (no question yet) ──► unknown        (say "go ahead", wait)
  │
  ├─ split_compound()      "continue. let's start with respiration."
  │                        -> first clause acts now, the rest is queued
  │
  ├─ classify_rules()      regex, microseconds, ~90% of real traffic
  │     └─ miss ─► fast LLM (gpt-oss-20b) with the tutor's last sentence
  │                and the last 2 exchanges as context
  │           └─ miss ─► heuristic: >=3 words -> question, else unknown
  ▼
one of eight intents
```

### The eight intents

| intent | means | routed to | examples |
|---|---|---|---|
| **session** | control the session | `session_handler` | pause, continue, restart, quit |
| **command** | change delivery | `command_handler` | repeat, slower, faster, switch the whole lesson to Hindi |
| **navigate** | move within the lesson **or** change subject | `find_section` | "go back", "skip this", "section 3", "the part about valves" |
| **explain** | didn't understand the last sentence | `explain` | "what does ventricle mean", "huh", "in simpler words", "say that in Hindi" |
| **question** | wants information | `qa_retrieve` | "how many chambers does the heart have", "who was William Harvey" |
| **meta** | about the **session**, not the subject | `session_status` | "how long will this take", "how much is left", "what are we studying", "who are you" |
| **backchannel** | "I'm following, carry on" | `resume_controller` | mm-hmm, okay, haan, theek hai |
| **unknown** | no request in it | `clarify` | "um", "this is boring" |

Sub-fields travel with the intent and are kept consistent with it, so a router
never sees a `question` carrying a `session_cmd`:

- `session_cmd`: pause | continue | restart | quit
- `command`: repeat | slower | faster | switch_lesson_lang (+ `command_arg`)
- `nav_target`: `{kind: prev|next|index|topic, value: …}`
- `reply_lang`: answer once in this language (explain only)

### Two distinctions that caused real bugs

**navigate-inside vs change-the-subject.** Both look like `navigate/topic`.
Only explicit wording changes the lesson:

| utterance | decision |
|---|---|
| "go to the part about valves" | jump inside this lesson |
| "tell me more about valves" | question inside this lesson |
| "I want to learn more about valves" | question inside this lesson |
| "I wanted respiration, **not** reproduction" | new lesson |
| "**change the topic to** volleyball" | new lesson |
| "let's **start with** football" | new lesson |
| "I want to learn football **now**" | new lesson |
| "topic change krte hai mujhe X ke bare me janna hai" | new lesson |

The topic name is taken from **after** the switch marker. Anchoring at the
start of the sentence produced "can we switch the volleyball now", which
Wikipedia resolved to *Dead or Alive Xtreme*. If the named topic is not in the
current lesson and the wording was not an explicit switch, the tutor says how
to switch rather than "let's carry on".

**subject question vs session question.** `meta` is checked *before*
`question`, because "how long will this take" looks exactly like a question and
retrieval has nothing useful to say about it. It went to a web search once and
answered "about three to four months" — the length of a teaching practicum.
`session_status` answers from the lesson plan: no model call, no tokens, cannot
be wrong. "How long does a heartbeat last" is still a subject question.

### Onboarding classifies differently

Before a lesson exists, an utterance is an **answer to the question just
asked** — which is why "hi" once became the topic, "I didn't understand the
question" became a Wikipedia lookup, and "just stop" became a lesson on *Just
Stop Oil*. Three guards run first (`agents/nodes.py::_onboarding_interrupt`):

- **quit** ("that's enough", "just stop") → end the session
- **confusion** ("what did you ask?", "samajh nahi aaya") → ask again
- **greeting** ("hi", "namaste") → ask again

All three match the **whole** utterance, so "explain photosynthesis" is still a
topic. Re-asking does not consume the turn or count as a failed attempt.
Confusion now also covers requests *about* the question ("ask me the language
again", "can you ask that again"), after one of them became a lesson on Yo-Yo
Ma. And the tutor says the topic back before building anything ("Okay,
photosynthesis, class 6. Give me a moment..."), so a misheard topic is caught
by the learner ten seconds before the wrong article is taught.

**Topic plus class, mid-lesson, is a new lesson.** "photosynthesis for class
6" said while the tutor is teaching something else is the onboarding answer
again; it now routes to `navigate/topic` (a whole new lesson) instead of being
answered as a question about the current one. Questions ("is this for class
6?") and in-lesson navigation ("the part about valves") are excluded.

While the tutor is asking *"which class?"*, only a class answers it. A non-class
reply used to overwrite the topic, so a misheard "class six" → "Plastics"
silently replaced "the heart". Misheard class words are accepted
(`class|clas|glass|grade|great`, plus squashed forms: `classics` → class 6).

---

## 4. After the intent: what the answer paths get

Every answer prompt receives the same context block (`_context_block`):

```
Lesson: Heart (for class 6)
Covered so far: Location and shape, Chambers
You were just saying: "<the sentence the learner interrupted>"
Asked earlier: what is an atrium; what is the aorta
Conversation so far:
Q: how many chambers does the heart have
A: The heart has four chambers...
```

Two tiers on purpose: the last **6 exchanges** carry full wording (answers
trimmed to 200 chars), and the **questions alone** are kept for the whole
session (12 of them), so an hour-long lesson can still answer "what did I ask
about first?" without paying for the full text every prompt. The intent
classifier gets only 2 exchanges — it runs on every rule miss and only needs
enough to resolve "that".

Follow-ups are stitched to the previous question before retrieval, because on
their own they retrieve nothing: "and which one is the strongest" scored 0.547
alone and was answered "I'm not sure". A grounded answer that the notes turn
out not to contain now gets one general-knowledge attempt (`answer_mode:
notes_miss`) before the extractive fallback. Web snippets are trimmed to two
sentences and are not read out at all if they are in an alphabet the voice
cannot speak.

---

## 5. Where each knob lives

| what | where |
|---|---|
| Whisper model, threads, VAD timings, language allowlist, transcript log | `stt/settings.py` (read by **both** entrypoints) |
| TTS provider, echo guard, duplex, LLM roles, token budgets, retrieval | `config.py` |
| Rules, markers, guards | `agents/intent.py` |
| Spoken lines | `agents/strings.py` (en + hi) |

---

## 6. What is still weak

- **ASR on short utterances.** One or two syllables are unreliable at any
  model size ("hi" → "はい"). Nothing downstream can fix a plausible wrong
  word: "the heart" → "The Hard" produces a correct disambiguation question
  about *Hardness / Hard water / Hard (band)*.
- **No echo cancellation.** The echo guard and half duplex make speakers
  usable, not good. The real fix is WebRTC AEC with the tutor's own PCM as the
  reference signal.
- **Section selection.** Wikipedia sections are taken in order, so a lesson can
  open on "Origins" or a cast list rather than an introduction.
- **A failed topic switch loses the old lesson.** Fetch failure resets to the
  topic question rather than resuming what was playing.
