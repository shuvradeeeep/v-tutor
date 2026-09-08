# Agent layer — gap-closing plan

## Where things stand (2026-09-08, after live sessions)

Everything below this block was written before the system had been driven by a
person with a microphone. It has now had a dozen live sessions, and this is the
delta. Detail: [AGENTS_PLAN.md](AGENTS_PLAN.md) §10 (graph) and
[STT_AND_INTENTS.md](STT_AND_INTENTS.md) (STT + classification).

**Closed since 2026-09-07**

| was open | now |
|---|---|
| ~2.4 s Whisper latency on CPU | **0.43–0.56 s** on `base`, ~1.8 s on `small`. One encoder pass instead of two, threads pinned to physical cores, greedy decoding. `scripts/bench_stt.py` re-measures it. |
| Whisper `base` mishears topics | Still true — but the damage is contained: misheard class words are accepted (`classics` → class 6), a non-class reply can no longer overwrite the topic, and `small` is now affordable. |
| Free Groq tier 429s → silent fallback to unsimplified text | `max_tokens` sized to measured work, real usage metered per model, 429 retried with the provider's own delay, background prep capped so it cannot eat the learner's minute. 15 min / 44 questions: **0 errors, 0 s learner-visible waiting**. |
| No echo cancellation in local mode | Still no AEC, but a speaker-only session survives: self-echo transcripts are dropped, and the session falls back to half duplex (deferred barge-in) rather than the tutor answering itself. |
| Follow-up memory two exchanges | Six with wording + every question asked this session. "And why is that?" and "what did I ask about first?" both work. |
| Hindi entirely on hold | Partly live: Whisper mis-detecting spoken Hindi as Urdu is corrected before decoding (`WHISPER_ALLOWED_LANGUAGES`), and Hindi/Hinglish topic-change and pause phrasings are in the rules. The Hindi *voice* is still untested end to end. |
| 152 offline tests | **210**, still offline, ~7 s. |

**Found by live use, fixed**

- Onboarding took any utterance as the answer: "hi" became a topic, "just stop"
  became a lesson on *Just Stop Oil*, "I didn't understand the question" became
  a Wikipedia lookup.
- `navigate` could not change subject: "I wanted respiration, not reproduction"
  answered "I couldn't find a part about that" and kept teaching reproduction.
- Questions about the session ("how long will this take") were web-searched and
  answered "about three to four months". Now a `meta` intent answered from the
  lesson plan with no model call.
- Polite pause requests ("can you pause for a while while I come?") were
  answered and then ignored — the lesson carried on.
- A grounded answer whose chunks did not contain the answer said "I'm not
  sure" about facts the model knew.
- A web result in Urdu was read aloud by the English voice.

**Still open**

- Section selection is positional, so a lesson can open on "Origins" or a cast
  list instead of an introduction.
- A failed topic switch loses the lesson that was playing.
- Short utterances remain unreliable at any model size ("hi" → "はい").
- Rime websocket streaming and word timestamps still not built; the heard
  cursor is word-approximate.
- Time to first beat is still ~9–12 s (Wikipedia + section pick + one localize).

---

## Where things stood (2026-09-07, late night)

**Voice pipeline connected (see `docs/VOICE_PIPELINE.md`):** mic/LiveKit →
Silero VAD → Whisper → graph → Rime coda → speakers/LiveKit track, in `voice/`
+ `main.py`. Verified end to end with `scripts/voice_dry_run.py` (real VAD,
real Whisper, real Groq, real Rime). 152 offline tests. Open on this layer:
Whisper `base` mishears topics (try `small`), ~2.4 s Whisper latency on CPU,
Rime HTTP 2–3 s per line (websocket streaming + word timestamps not built,
cursor is word-approximate), no echo cancellation in local mode (headphones),
Rime `clear` not needed on HTTP (nothing queued server-side).

**Working (verified live with Groq gpt-oss-20b / 120b, DuckDuckGo, Wikipedia):**
- Voice onboarding → Wikipedia fetch → model picks 8 sections → intro simplified
  before the first beat, the rest in a background thread → lesson read in beats.
- Barge-in at any point; stale results fenced (`test_fencing.py`), 145 offline tests.
- Questions: from the notes (hybrid retrieval) · trivial / general-knowledge
  straight from the model, no search (~0.5 s) · web only when the model says
  `LOOKUP` (~2 s).
- Explain / repeat / slower / faster / navigate / pause / continue / restart /
  quit / compound "A and B" / "I have a question" → "go ahead".
- Gap filler (once per turn), evidence CSV, SQLite resume, PDF path on the
  generated fixture.
- Tools: `scripts/text_harness.py --live --web duckduckgo` (interactive),
  `scripts/live_check.py` (non-interactive smoke + timings),
  `scripts/probe_intent.py` (intent prompt accuracy).

**Still open:**
- Time to first beat ~9–10 s after "give me a moment" (sequential: Wikipedia
  2.5 s, section pick 1–5 s, one localize call ~2 s). Not optimised on purpose.
- Free Groq tier: 8000 tokens/min per model → a second lesson build within a
  minute gets 429s and silently falls back to unsimplified text.
- Intent prompt 26/32 on odd utterances; misses are ambiguous phrasings.
- Step 6: real demo PDF not yet run through `parse_pdf` (needs the file).
- Hindi entirely on hold: voice, Hinglish rules, Whisper Devanagari check,
  Hindi Wikipedia fallback untested live.
- ~~Audio layer not connected~~ → connected 2026-09-07 (`voice/`, `main.py`).
  Remaining audio gaps are listed at the top of this section.

---

> **Status (2026-09-07, night):** steps 1–5, 7, 8 **done**; steps 3 and 4 now
> **verified with the real models** (Groq `gpt-oss-20b` fast / `gpt-oss-120b`
> strong). 145 offline tests. English pipeline works end to end: Wikipedia →
> model-picked 8 sections → simplified beats → questions from notes / model /
> web → fenced barge-in. Hindi is **on hold** until the English path is signed off.
>
> **Real-model pass (step 4) — what was found and fixed:**
> - Reasoning models think before answering: at default effort a one-word intent
>   took 2–3 s. `reasoning_effort=low` is sent automatically for known reasoning
>   models (0.5 s, same quality). `LLM_REASONING_EFFORT` overrides.
> - Model output carries typographic glyphs (non-breaking hyphens, curly quotes,
>   em dashes) and stray markdown. `llm.clean_output` strips them on every reply
>   so TTS and the Windows console never see them.
> - Intent JSON prompt rewritten against 32 odd utterances the rules miss
>   (`scripts/probe_intent.py`): 13/32 → 26/32. Parser hardened against string
>   `"null"`, invalid commands, and sub-fields inconsistent with the intent.
>   Small rule additions from the probe: bare "what"/"huh" → explain,
>   "pronounce"/"spell" → explain, "I'm back" when paused → continue,
>   "I have a question" → *"Sure, go ahead."* and wait.
> - Line-count retry rate on 3 live articles (Photosynthesis, Heart, Water
>   cycle): **0 retries in 11 localize calls**. Background sections 8/8 ok.
> - **New: direct answers.** Out-of-notes questions go to the strong model
>   first (`direct_answer` node); it replies `LOOKUP` only when it needs current
>   or very specific information, and only then does web search run. "Capital
>   of France" / "moons of Mars": 0.4–0.7 s, no search. "Who won yesterday's
>   match": LOOKUP → web. With the stub model the path is unchanged (straight
>   to web), so the fencing tests are untouched. Fenced like every other
>   speaking branch (`fence_direct`).
> - Wikipedia blocks **httpx by client fingerprint** (403 with the same UA that
>   gets 200 via `requests`/curl). Fetch switched to `requests`, one round trip
>   (`generator=search` + `extracts`).
> - Gap filler fired twice on web questions (armed for the search, re-armed for
>   the answer). Now at most once per turn.
> - DuckDuckGo via `ddgs` "auto" took 2.5–8 s; Brave backend ~1.1 s, tried first.
> - `PREPARE_UPFRONT_SECTIONS` 2 → 1 (one fewer strong-model call before the
>   first beat).
>
> **Known, deliberately not optimised yet (user's call, 2026-09-07):** time to
> first beat is ~9–10 s after "give me a moment" (Wikipedia 2.5 s + section pick
> ~1–5 s + one localize call ~2 s, all sequential). Free Groq tier is 8000
> tokens/minute per model; a second lesson build within a minute gets 429s and
> falls back to unsimplified text. `scripts/live_check.py` measures all of this.
>
> Still open and **needs you**: a real PDF to replace the generated fixture
> (step 6); ten minutes of spoken Hindi through Whisper (step 1's Hinglish list
> is a best guess until then) — deferred with the rest of Hindi.

Everything below is what stands between "works in the harness" and "works for a
real student on demo day", in the order it was done.

Each step says what it needs. Steps marked **[no key]** can be done right now.

---

## Step 1 — Small bugs [no key] · ~1 hour

| Gap | Fix | Test |
|---|---|---|
| Bare "stop" gets "Sorry?" | add `stop`, `stop please`, `रुक जाओ`, `bas` to the pause rule | routing table |
| "six" as the class answer becomes the topic | in `choose_source`, if `topic` is set and `grade` missing, a bare number or number word is the grade | onboarding flow |
| Hinglish commands unrecognised | romanised keyword lists beside every Hindi rule: `dheere`, `dobara`, `ruko`, `aage badho`, `samjhao`, `matlab`, `haan`, `theek hai`, `wapas`, `agla`, `band karo` | routing table, ~15 utterances |
| Disambiguation pages | if the extract's first section is mostly one-line bullets or the title ends in "(disambiguation)", speak "did you mean…" with the first three options and re-ask | fixture of a canned disambiguation extract |

## Step 2 — Writing for the ear [no key] · ~2 hours

A `clean_for_speech(text)` pass in `material.py`, applied to **beats only**
(retrieval keeps the original text so facts stay findable):

- drop `[12]`-style citations and `(…)` parentheticals longer than 3 words
- spell out `%` → "percent", `→` → "gives", `&` → "and", `°C` → "degrees Celsius"
- en-dash ranges `3–6%` → "3 to 6 percent"
- chemical formulas (`6CO2 + 6H2O`) → detected by regex, replaced with "the chemical equation" and the sentence kept only if something else survives; otherwise the sentence is dropped
- collapse double spaces, ensure every beat ends in terminal punctuation (Rime pauses on it)

Then render two variants of the same sentence through Rime once the client
exists (before/after cleanup), save both clips: that is the PS's
"pronunciation and controlled delivery" evidence.

## Step 3 — Lesson length and onboarding wait [needs strong LLM]

- **Cap:** take at most 8 sections / 30 beats for a class-level lesson. First
  choice: ask the strong model, given the section titles and the grade, which
  sections to keep and in what order (one call, returns a list). Fallback with
  no model: first 8 sections.
- **Prepare in two stages:** simplify only the first 2 sections before
  speaking; speak *"Give me a moment while I get the lesson ready"* (pre-rendered
  clip); simplify the rest in a background thread and swap beats in as they
  finish. `teach_step` reads whatever is ready; if it reaches an unprepared
  beat it reads the original text rather than waiting.
- **Robust line parsing:** if the model returns a different line count, retry
  once with "return exactly N lines"; on second failure keep the original for
  that section and log it. Measure how often this happens on 3 articles.

## Step 4 — Real models in [needs 2 API keys] · DONE 2026-09-07

- Set `LLM_FAST_*` and `LLM_STRONG_*` in `.env`. No code changes. ✔ Groq gpt-oss.
- Run the harness against Photosynthesis and The Heart with the real models.
  ✔ `scripts/live_check.py` (non-interactive) + `scripts/text_harness.py --live`.
- Test the intent JSON prompt on 30 odd utterances the rules miss. ✔
  `scripts/probe_intent.py`, 26/32; remaining misses are genuinely ambiguous
  ("no I meant the other one", "um okay so").
- Replace the `is_follow_up` regex with the model's judgement only if the regex
  proves too crude; keep the regex as the fallback. — regex kept, was fine.
- Add the Rime "writing for the ear" guidance to the `compose_answer` and
  `explain` prompts. ✔ `TutorNodes.EAR_RULES`, also in `direct_answer`.
- **Trivial questions skip the web** (added on request): `direct_answer` node,
  see status block above.

## Step 5 — Retrieval for real [needs ~120 MB download, no key]

- Set `EMBEDDER=fastembed`, run `test_retrieval.py`. Re-tune τ if in-notes
  scores shift (they will move up; out-of-notes should stay near 0).
- Add a Hindi fixture (a short Hindi Wikipedia extract) with 6 questions in
  Hindi and 2 in romanised Hindi. Expect BM25 to carry most of it.
- Wire DuckDuckGo behind `WEB_SEARCH_PROVIDER=duckduckgo` (`duckduckgo_search`
  library, no key). Keep the stub as default for tests.

## Step 6 — PDF path [needs one real PDF]

- Run `parse_pdf` on a real NCERT-style chapter. Check: headings detected,
  running headers/footers and page numbers not read aloud, tables flattened
  sanely.
- Add the file as `tests/fixtures/sample_chapter.pdf` and a test that asserts
  section count and that no line is a bare page number.
- Scanned PDF fixture (image only) → asserts the "I can't read that file" path.

## Step 7 — Gap filler and evidence writer [no key]

- `agents/gap_filler.py`: arm on entering `web_search_node`, `compose_answer`,
  `explain`, `ingest_material`; after 700 ms of no speech play the matching
  pre-rendered clip ("Let me check that…", "One moment…"); disarm on first
  speech; carries `turn_id`, dropped if stale. With `TextSpeaker` it just
  prints — testable now.
- `agents/evidence.py`: subscribe to `Deps.on_event`, append every event with a
  timestamp to `evidence/session_<id>.csv`. This is the raw material for
  `RIME_EVIDENCE.md`; the audio benchmark adds `t_stop` and `stale_leak_count`
  columns later.

## Step 8 — Persistence [no key] · 15 minutes

- `SqliteSaver` for `main.py`, `InMemorySaver` stays for tests. Reconnecting a
  browser session resumes where the lesson was.

---

## What I need from you

| Need | For which step | Why |
|---|---|---|
| **Two LLM API keys + model names** (fast, strong) | 3, 4 | Everything about lesson quality is blind until then. Any provider in `llm.py` works: Anthropic, OpenAI, Groq. |
| **One real PDF** you'd actually demo with | 6 | The heading heuristic has never met a real file. |
| **A Hindi speaker for 10 minutes** (you?) | 1, 5 | To say the commands aloud through Whisper and see how they come out (Devanagari vs romanised), and to sanity-check the Hindi phrases in `strings.py`. |
| **Rime API key** | 2 (before/after clips), then the whole audio layer | Not needed for anything in the agent layer itself. |
| **Decision: lesson length** | 3 | 8 sections / 30 beats is my default. A class 6 lesson is 10–15 minutes of speech at that size. |
| **Decision: DuckDuckGo** | 5 | You said "later"; say when. |

Out of scope unless you say otherwise: quizzing, multiple documents, memory
across sessions, Arabic voice.
