# Agent layer — gap-closing plan

> **Status (2026-09-07, evening):** steps 1, 2, 5, 6, 7, 8 are **done** and
> tested (135 offline tests). Steps 3 and 4 are **implemented and tested against
> fake models** — section cap, model-picked outline, two-stage preparation with
> a background thread, line-count retry, ear rules in every prompt, JSON intent
> path — and only need real API keys in `.env` to be exercised for real.
> Verified live: Wikipedia (23 sections → 8/30 beats), DuckDuckGo, gap filler,
> evidence CSV, SQLite resume in a fresh process.
>
> Still open and **needs you**: LLM keys + model names (step 4), a real PDF to
> replace the generated fixture (step 6), ten minutes of spoken Hindi through
> Whisper to check Devanagari vs romanised output (step 1's Hinglish list is a
> best guess until then).

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

## Step 4 — Real models in [needs 2 API keys]

- Set `LLM_FAST_*` and `LLM_STRONG_*` in `.env`. No code changes.
- Run the harness against Photosynthesis and The Heart with the real models.
- Test the intent JSON prompt on 30 odd utterances the rules miss ("I think
  plants eat sunlight", "wait what", "is that the same as the thing before").
  Adjust the prompt, not the rules.
- Replace the `is_follow_up` regex with the model's judgement only if the regex
  proves too crude; keep the regex as the fallback.
- Add the Rime "writing for the ear" system-prompt guidance to the
  `compose_answer` and `explain` prompts (short sentences, no lists, spell out
  numbers under 10, no symbols).

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
