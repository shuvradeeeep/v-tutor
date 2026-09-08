"""
Node implementations for the tutor graph. See docs/AGENTS_PLAN.md.

Two invariants are enforced here and nowhere else:

  1. All speech goes through TutorNodes._say(), which refuses to speak if the
     branch's birth turn is not the live turn. That is the fence.
  2. handle_interrupt performs no I/O. The stop (Rime clear + local flush) is
     done by the realtime layer / runner BEFORE the graph is resumed; this node
     only records where the learner stopped hearing.

Every node returns a partial state update. No node mutates `state`.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

from langgraph.types import interrupt

import config
from agents import intent as intent_mod
from agents.material import DisambiguationError, make_beats, make_chunks, split_sentences, word_count
from agents.retrieval import Doc, HybridRetriever, best_sentences
from agents.session import Deps
from agents.strings import t

logger = logging.getLogger("v-tutor.nodes")


def _merge(*updates: dict) -> dict:
    """Merge partial updates; concatenate the append-only transcript."""
    out: dict = {}
    for u in updates:
        for k, v in u.items():
            if k == "transcript":
                out[k] = out.get(k, []) + list(v)
            else:
                out[k] = v
    return out


class TutorNodes:
    def __init__(self, deps: Deps) -> None:
        self.d = deps

    # ------------------------------------------------------------------ helpers
    def _store(self, state: dict) -> dict:
        store = self.d.stores.setdefault(state["session_id"], {})
        if "retriever" not in store and state.get("sections"):
            self._build_indexes(state, store)          # e.g. session resumed after a restart
        return store

    def _build_indexes(self, state: dict, store: dict) -> None:
        """Retrieval indexes over ALL fetched sections (the lesson may read fewer)."""
        sections = state.get("sections") or []
        chunks = make_chunks(sections, config.CHUNK_SENTENCES, config.CHUNK_OVERLAP)
        store["retriever"] = HybridRetriever(
            [Doc(c["id"], c["text"], c["section_id"], c["section_title"], {"source": c["source"]})
             for c in chunks], self.d.embedder)
        store["section_retriever"] = HybridRetriever(
            [Doc(sec["id"], f"{sec['title']}. {sec['text'][:800]}", sec["id"], sec["title"])
             for sec in sections], self.d.embedder)
        store["n_chunks"] = len(chunks)

    def _arm(self, state: dict, key: str = "filler_check") -> None:
        """Start the dead-air watchdog for a slow step. Disarmed by the next _say()."""
        gf = self.d.gap_filler
        if gf is None:
            return
        lang = self._lang(state)
        text, speaker = t(lang, key), config.LANG_SPEAKER.get(lang) or state.get("speaker") or ""
        speed, turn = state.get("speed_alpha", config.SPEED_ALPHA_DEFAULT), state.get("born_turn_id", 0)

        def say() -> None:
            self.d.speaker.speak(text, turn_id=turn, lang=lang, speaker=speaker, speed_alpha=speed)
            self.d.emit("filler", turn=turn, text=text)
        gf.arm(turn, say)

    def _lang(self, state: dict) -> str:
        return state.get("active_lang") or "en"

    def _T(self, state: dict, key: str, **fmt: Any) -> str:
        return t(self._lang(state), key, **fmt)

    @staticmethod
    def _lang_name(code: str) -> str:
        return config.LANG_NAMES.get(code, code)

    def _say(self, state: dict, text: str, *, lang: str | None = None,
             kind: str = "lesson") -> dict:
        """Fenced speak. Returns the state updates to merge ({} if nothing said)."""
        text = (text or "").strip()
        if not text:
            return {}
        if self.d.gap_filler is not None:
            self.d.gap_filler.disarm()                    # real speech is about to start
        born = state.get("born_turn_id", 0)
        live = self.d.clock.current()
        if born != live:
            self.d.emit("fence_drop", born=born, live=live, kind=kind, text=text[:80])
            logger.info("FENCE: dropped %s text from turn %d (live turn %d)", kind, born, live)
            return {"stale_drops": state.get("stale_drops", 0) + 1}
        lang = lang or self._lang(state)
        speaker = config.LANG_SPEAKER.get(lang) or state.get("speaker") or ""
        self.d.speaker.speak(text, turn_id=live, lang=lang, speaker=speaker,
                             speed_alpha=state.get("speed_alpha", config.SPEED_ALPHA_DEFAULT))
        self.d.emit("speak", turn=live, kind=kind, lang=lang, text=text)
        return {
            "pending_text": text,
            "spoken_sentences": split_sentences(text),
            "spoken_kind": kind,
            "transcript": [{"role": "tutor", "kind": kind, "turn": live, "lang": lang, "text": text}],
        }

    # --------------------------------------------------------------- onboarding
    def _onboarding_interrupt(self, state: dict, utter: str, reask: str) -> dict | None:
        """
        Handle anything that is not an answer to the question just asked.

        Onboarding has no intent classification -- it takes what it hears as
        the answer. That made "hi" a topic, "I didn't understand the question"
        a Wikipedia lookup, and "just stop" a lesson about Just Stop Oil. So
        before an utterance is read as an answer:

          quit ("that's enough", "just stop")   -> end the session
          confusion ("what did you ask?")       -> ask the question again
          a greeting ("hi")                     -> ask the question again

        Re-asking does not consume the turn or count as a failed attempt, so
        the learner can be confused twice and still be understood the third
        time. Returns None when the utterance really is an answer.
        """
        c = intent_mod.classify_rules(utter)
        if (c and c.session_cmd == "quit") or intent_mod.is_stop_request(utter):
            return _merge(self._say(state, self._T(state, "goodbye"), kind="system"),
                          {"lesson_done": True, "session_cmd": "quit", "answer": None})
        if intent_mod.is_confusion(utter) or intent_mod.is_greeting(utter):
            self.d.emit("onboarding_reask", utterance=utter[:80],
                        step=state.get("onboarding_step"))
            return {"answer": reask, "user_utterance": None}
        return None

    def choose_language(self, state: dict) -> dict:
        if state.get("onboarding_step") != "language":
            return {}                                    # preset via screen button: skip the question
        utter = (state.get("user_utterance") or "").strip()
        if not utter:
            return {"answer": t("en", "ask_language")}
        if (q := self._onboarding_interrupt(state, utter, t("en", "ask_language"))):
            return q
        lang = intent_mod.parse_language(utter, state.get("detected_lang"), config.SUPPORTED_LANGS)
        if not lang:
            named = intent_mod._lang_mentioned(utter.lower())
            ask = t("en", "ask_language")
            if named and named not in config.SUPPORTED_LANGS:
                names = " or ".join(config.LANG_NAMES[l] for l in config.SUPPORTED_LANGS)
                ask = f"{t('en', 'lesson_lang_unsupported', langs=names)} {ask}"
            return {"answer": ask, "user_utterance": None}
        return {
            "active_lang": lang, "speaker": config.LANG_SPEAKER[lang],
            "onboarding_step": "source", "answer": None, "user_utterance": None,
            "transcript": [{"role": "system", "text": f"language={lang}"}],
        }

    def choose_source(self, state: dict) -> dict:
        if state.get("pdf_paths"):
            return _merge(self._say(state, self._T(state, "prepare_wait"), kind="system"),
                          {"source_kind": "pdf", "answer": None, "user_utterance": None,
                           "onboarding_step": "done"})
        utter = (state.get("user_utterance") or "").strip()
        if not utter:
            return {"answer": self._T(state, "ask_source")}
        # Re-ask whichever question is actually outstanding: the topic, or the
        # class if the topic is already known.
        pending = "ask_grade" if (state.get("topic") and not state.get("grade")) else "ask_source"
        if (q := self._onboarding_interrupt(state, utter, self._T(state, pending))):
            return q
        topic, grade = intent_mod.parse_topic_grade(utter)
        # "no, I said respiration, not reproduction" while still being onboarded:
        # take the topic they are correcting TO, not the whole sentence.
        if (corrected := intent_mod.parse_topic_switch(utter)):
            topic = corrected
        if state.get("topic") and not state.get("grade"):
            g = intent_mod.parse_grade_only(utter)     # tutor asked "which class?"; "six" is the grade
            if g:
                topic, grade = None, f"class {g}"
        topic = topic or state.get("topic")
        grade = grade or state.get("grade")
        asks = state.get("source_asks", 0)
        if not topic:
            return {"answer": self._T(state, "ask_source_again"), "source_asks": asks + 1,
                    "user_utterance": None}
        if not grade and asks < 1:
            return {"topic": topic, "answer": self._T(state, "ask_grade"), "source_asks": asks + 1,
                    "user_utterance": None}
        return _merge(self._say(state, self._T(state, "prepare_wait"), kind="system"),
                      {"source_kind": "topic", "topic": topic, "grade": grade, "answer": None,
                       "user_utterance": None, "onboarding_step": "done"})

    def parse_pdf(self, state: dict) -> dict:
        res = self.d.pdf_parse(list(state.get("pdf_paths") or []))
        if not res:
            return {"answer": self._T(state, "cant_read_pdf"), "pdf_paths": [], "sections": [],
                    "source_kind": None, "onboarding_step": "source"}
        sections, lang, title = res
        return {"sections": sections, "source_lang": lang, "source_title": title, "source_url": None}

    def fetch_material(self, state: dict) -> dict:
        topic = state.get("topic") or ""
        lang = self._lang(state)
        preface = None
        try:
            res = self.d.wiki_fetch(topic, lang)
            if not res and lang != "en":
                res = self.d.wiki_fetch(topic, "en")
                if res:
                    preface = self._T(state, "fallback_english")
        except DisambiguationError as e:
            opts = e.options[:3]
            joined = (", ".join(opts[:-1]) + " or " + opts[-1]) if len(opts) > 1 else (opts[0] if opts else "")
            return {"answer": self._T(state, "disambiguation", title=e.title, options=joined),
                    "topic": None, "sections": [], "source_kind": None, "onboarding_step": "source"}
        if not res:
            return {"answer": self._T(state, "not_found_topic"), "topic": None, "sections": [],
                    "source_kind": None, "onboarding_step": "source"}
        sections, src_lang, url = res
        return {"sections": sections, "source_lang": src_lang, "source_url": url,
                "source_title": sections[0]["title"] if sections else topic, "preface": preface}

    # ---- localisation: translate to the study language and/or simplify to grade
    EAR_RULES = ("Write for the ear: short sentences, plain words, no lists, no markdown, "
                 "no symbols or abbreviations, spell out units, one idea per sentence.")

    def _needs_localize(self, state: dict) -> bool:
        return (state.get("source_lang") or "en") != self._lang(state) or bool(state.get("grade"))

    def _localize_sentences(self, sents: list[str], dst: str, grade: str | None) -> list[str] | None:
        """One model call for one section. Returns None if the model is unavailable
        or still returns the wrong line count after a retry -- callers then keep
        the original text, so a bad model answer never loses lesson content."""
        system = (f"You rewrite lesson sentences to be read aloud to a {grade or 'school'} student. "
                  f"Output language: {config.LANG_NAMES.get(dst, dst)}. Keep every fact and number. "
                  f"{self.EAR_RULES} Return exactly one numbered line per input line, same numbering, "
                  "nothing else.")
        numbered = "\n".join(f"{k + 1}. {x}" for k, x in enumerate(sents))
        user = numbered
        for attempt in range(config.LOCALIZE_RETRIES + 1):
            reply = self.d.llm_strong.complete(system, user)
            if not reply:
                return None
            lines = [re.sub(r"^\s*\d+[.)]\s*", "", l).strip() for l in reply.splitlines() if l.strip()]
            if len(lines) == len(sents):
                return lines
            self.d.emit("localize_retry", attempt=attempt + 1, got=len(lines), want=len(sents))
            user = (f"You returned {len(lines)} lines but there were {len(sents)} inputs. Return EXACTLY "
                    f"{len(sents)} numbered lines, one per input. Do not merge or split sentences.\n\n{numbered}")
        return None

    @staticmethod
    def _apply_lines(beats: list[dict], idxs: list[int], lines: list[str]) -> None:
        k = 0
        for i in idxs:
            n = len(beats[i]["sentences"])
            beats[i]["sentences"] = lines[k:k + n]
            beats[i]["text"] = " ".join(lines[k:k + n])
            k += n

    def _localize_beats(self, state: dict, beats: list[dict],
                        only_sections: set[str] | None = None) -> list[dict]:
        """Synchronous pass over the given sections (all if None). Returns copies."""
        if not self._needs_localize(state):
            return beats
        dst, grade = self._lang(state), state.get("grade")
        by_section: dict[str, list[int]] = {}
        for i, b in enumerate(beats):
            by_section.setdefault(b["section_id"], []).append(i)
        out = [dict(b) for b in beats]
        for sec, idxs in by_section.items():
            if only_sections is not None and sec not in only_sections:
                continue
            lines = self._localize_sentences([x for i in idxs for x in beats[i]["sentences"]], dst, grade)
            if lines:
                self._apply_lines(out, idxs, lines)
        return out

    def _localize_in_background(self, state: dict, beats: list[dict], sections: list[str],
                                store: dict) -> None:
        """Remaining sections are prepared while the tutor is already teaching.
        Results land in store['prepared'][beat_id]; teach_step swaps them in."""
        dst, grade = self._lang(state), state.get("grade")
        prepared: dict[str, list[str]] = store.setdefault("prepared", {})

        def work() -> None:
            for sec in sections:
                idxs = [i for i, b in enumerate(beats) if b["section_id"] == sec]
                lines = self._localize_sentences([x for i in idxs for x in beats[i]["sentences"]], dst, grade)
                if lines:
                    k = 0
                    for i in idxs:
                        n = len(beats[i]["sentences"])
                        prepared[beats[i]["id"]] = lines[k:k + n]
                        k += n
                self.d.emit("localized_bg", section=sec, ok=bool(lines))
            store["prepare_done"] = True

        th = threading.Thread(target=work, name="localize-bg", daemon=True)
        store["prepare_thread"] = th
        th.start()

    def _select_sections(self, state: dict, sections: list[dict]) -> list[dict]:
        """Cap the lesson to a class-sized read. With a real model, let it pick
        which sections suit the grade; otherwise take the first N."""
        if len(sections) <= config.MAX_SECTIONS:
            return sections
        if getattr(self.d.llm_strong, "provider", "stub") != "stub":
            titles = "\n".join(f"{i + 1}. {sec['title']}" for i, sec in enumerate(sections))
            reply = self.d.llm_strong.complete(
                f"You plan a spoken lesson for a {state.get('grade') or 'school'} student on "
                f"'{state.get('topic') or state.get('source_title')}'. Pick at most {config.MAX_SECTIONS} "
                "section numbers, in teaching order. Reply with the numbers separated by commas, nothing else.",
                titles)
            nums = [int(x) for x in re.findall(r"\d+", reply or "")]
            picked = [sections[n - 1] for n in dict.fromkeys(nums) if 1 <= n <= len(sections)]
            if picked:
                return picked[:config.MAX_SECTIONS]
        return sections[:config.MAX_SECTIONS]

    def ingest_material(self, state: dict) -> dict:
        sections = state.get("sections") or []
        store = self.d.stores.setdefault(state["session_id"], {})
        self._build_indexes(state, store)
        lesson_sections = self._select_sections(state, sections)
        beats = make_beats(lesson_sections, config.BEAT_SENTENCES,
                           state.get("source_lang") or "en")[:config.MAX_BEATS]
        sec_order = list(dict.fromkeys(b["section_id"] for b in beats))
        upfront = set(sec_order[:config.PREPARE_UPFRONT_SECTIONS])
        later = sec_order[config.PREPARE_UPFRONT_SECTIONS:]
        self._arm(state, "filler_moment")
        beats = self._localize_beats(state, beats, only_sections=upfront)
        store.pop("prepared", None)
        store["prepare_done"] = not (later and self._needs_localize(state))
        if not store["prepare_done"]:
            self._localize_in_background(state, beats, later, store)
        title = state.get("source_title") or state.get("topic") or ""
        preface = self._T(state, "lets_begin", title=title)
        if state.get("preface"):
            preface = f"{state['preface']} {preface}"
        self.d.emit("ingest", sections=len(sections), lesson_sections=len(lesson_sections),
                    beats=len(beats), upfront=len(upfront), background=len(later))
        return {"lesson_plan": beats, "beat_index": 0, "beat_spoken": False, "lesson_done": False,
                "heard_cursor": None, "heard_sentence": None, "onboarding_step": "done",
                "preface": preface, "answer": None, "user_utterance": None,
                "topic_switch": False, "intent": None}

    # -------------------------------------------------------------- lesson loop
    def _resume_start(self, state: dict, idx: int, beat: dict) -> int:
        """Sentence index to resume from, per AGENTS_PLAN section 4."""
        cur = state.get("heard_cursor")
        if not cur or cur.get("beat_index") != idx:
            return 0
        sents = beat["sentences"]
        s = max(0, min(cur.get("sentence_index", 0), len(sents) - 1))
        w = cur.get("word_index", 0)
        if s == 0 and w < 3:
            return 0                                      # barely started: replay the beat
        if w >= word_count(sents[s]):
            return s + 1                                  # heard that sentence fully: next one
        return s                                          # mid-sentence: replay the sentence

    def _beat(self, state: dict, plan: list[dict], idx: int) -> dict:
        """The beat to read: background-prepared text if it has arrived, else the original."""
        beat = dict(plan[idx])
        prepared = self._store(state).get("prepared", {}).get(beat["id"])
        if prepared and len(prepared) == len(beat["sentences"]):
            beat["sentences"] = prepared
            beat["text"] = " ".join(prepared)
        return beat

    def teach_step(self, state: dict) -> dict:
        if state.get("paused"):
            return {}
        plan = state.get("lesson_plan") or []
        idx = state.get("beat_index", 0) + (1 if state.get("beat_spoken") else 0)
        start = 0
        if idx < len(plan):
            start = self._resume_start(state, idx, self._beat(state, plan, idx))
            if start >= len(plan[idx]["sentences"]):
                idx, start = idx + 1, 0                   # cursor was at the end: move on
        if idx >= len(plan):
            said = self._say(state, self._T(state, "lesson_done"), kind="system")
            return _merge(said, {"lesson_done": True, "beat_index": len(plan), "beat_spoken": True,
                                 "heard_cursor": None})
        beat = self._beat(state, plan, idx)
        prefix_parts: list[str] = []
        if state.get("preface"):
            prefix_parts.append(state["preface"])
        new_section = idx == 0 or plan[idx - 1]["section_id"] != beat["section_id"]
        if new_section and start == 0 and beat.get("section_title"):
            prefix_parts.append(f"{beat['section_title'].rstrip('.')}.")
        prefix = " ".join(prefix_parts)
        body = " ".join(beat["sentences"][start:])
        said = self._say(state, f"{prefix} {body}".strip(), kind="lesson")
        spoke = "pending_text" in said
        return _merge(said, {
            "beat_index": idx, "beat_spoken": spoke, "preface": None,
            "heard_cursor": None if spoke else state.get("heard_cursor"),
            "spoken_offset": len(split_sentences(prefix)) if prefix else 0,
            "spoken_start_sentence": start,
        })

    def await_event(self, state: dict) -> dict:
        ev = interrupt({"waiting_for": "event", "turn_id": self.d.clock.current()})
        if isinstance(ev, str):
            ev = {"type": ev}
        ev = dict(ev or {})
        return {"event": ev, "user_utterance": ev.get("text"),
                "detected_lang": ev.get("detected_lang")}

    def promote_queued(self, state: dict) -> dict:
        """Second clause of a compound request becomes the current utterance.

        "Go to chambers AND explain it simpler": by the time the second clause
        runs, the tutor has just read the new section, so "it" means that text,
        not the sentence that was playing when the learner first spoke."""
        spoken = state.get("spoken_sentences") or []
        heard = state.get("heard_sentence")
        if spoken:
            if state.get("spoken_kind") == "lesson":
                heard = " ".join(spoken[state.get("spoken_offset", 0):]) or spoken[-1]
            else:
                heard = spoken[-1]
        return {"user_utterance": state.get("queued_request"), "queued_request": None,
                "heard_sentence": heard}

    # ----------------------------------------------------------------- interrupt
    def _normalize_cursor(self, state: dict, words_heard: int | None) -> tuple[dict | None, str | None]:
        spoken = state.get("spoken_sentences") or []
        if not spoken:
            return state.get("heard_cursor"), state.get("heard_sentence")
        counts = [word_count(s) for s in spoken]
        total = sum(counts)
        if words_heard is None or words_heard > total:
            words_heard = total
        acc, k, w = 0, len(spoken) - 1, counts[-1]
        for i, c in enumerate(counts):
            if words_heard < acc + c:
                k, w = i, words_heard - acc
                break
            acc += c
        heard_sentence = spoken[k]
        if state.get("spoken_kind") != "lesson":
            return state.get("heard_cursor"), heard_sentence      # answer/system: lesson cursor unchanged
        off = state.get("spoken_offset", 0)
        start = state.get("spoken_start_sentence", 0)
        if k < off:
            beat_sent, w = start, 0                               # interrupted during the title/preface
        else:
            beat_sent = start + (k - off)
        cursor = {"beat_index": state.get("beat_index", 0), "sentence_index": beat_sent,
                  "word_index": w, "seconds": 0.0}
        return cursor, heard_sentence

    def handle_interrupt(self, state: dict) -> dict:
        live = self.d.clock.current()
        while live <= state.get("turn_id", 0):
            live = self.d.clock.bump()                # nobody bumped for us, or a fresh clock after a restart
        ev = state.get("event") or {}
        cur_in = ev.get("cursor") or {}
        cursor, heard_sentence = self._normalize_cursor(state, cur_in.get("word_index"))
        self.d.emit("interrupt", turn=live, cursor=cursor, heard=heard_sentence)
        return {
            "turn_id": live, "born_turn_id": live,
            "heard_cursor": cursor, "heard_sentence": heard_sentence,
            "beat_spoken": False,
            "answer": None, "reply_lang": None, "intent": None, "command": None,
            "command_arg": None, "session_cmd": None, "nav_target": None,
            "queued_request": None, "web_aborted": False, "answer_mode": None,
            "transcript": [{"role": "learner", "turn": live, "text": state.get("user_utterance") or ""}],
        }

    def classify_intent(self, state: dict) -> dict:
        utter = (state.get("user_utterance") or "").strip()
        parts = intent_mod.split_compound(utter)
        c = intent_mod.classify(parts[0], paused=bool(state.get("paused")), llm=self.d.llm_fast)
        upd = c.as_updates()
        upd["user_utterance"] = parts[0]
        upd["queued_request"] = parts[1] if len(parts) > 1 else None
        if c.intent != "unknown":
            upd["clarify_count"] = 0
        self.d.emit("intent", **{k: v for k, v in upd.items() if v is not None})
        return upd

    def clarify(self, state: dict) -> dict:
        if intent_mod.is_ask_permission(state.get("user_utterance") or ""):
            # "I have a question" -- invite it and wait; this is not a miss.
            return {"answer": self._T(state, "go_ahead"), "clarify_count": 0}
        n = state.get("clarify_count", 0)
        if n >= config.CLARIFY_MAX_ASKS:
            # Second miss in a row: don't loop on "pardon?" -- carry on with the lesson.
            return {"answer": None, "intent": "backchannel", "clarify_count": 0}
        return {"answer": self._T(state, "clarify"), "clarify_count": n + 1}

    def session_handler(self, state: dict) -> dict:
        cmd = state.get("session_cmd")
        if cmd == "pause":
            return _merge(self._say(state, self._T(state, "paused"), kind="system"),
                          {"paused": True, "answer": None})
        if cmd == "continue":
            return {"paused": False, "answer": None}
        if cmd == "restart":
            return _merge(self._say(state, self._T(state, "restart"), kind="system"),
                          {"beat_index": 0, "beat_spoken": False, "heard_cursor": None,
                           "paused": False, "answer": None})
        if cmd == "quit":
            return _merge(self._say(state, self._T(state, "goodbye"), kind="system"),
                          {"lesson_done": True, "answer": None})
        return {}

    def command_handler(self, state: dict) -> dict:
        cmd = state.get("command")
        cur = dict(state.get("heard_cursor") or {})
        if cur:
            cur["word_index"] = 0                                  # replay from the sentence start
        upd: dict = {"answer": None, "heard_cursor": cur or None}
        if cmd == "repeat":
            return upd
        if cmd in ("slower", "faster"):
            step = config.SPEED_ALPHA_STEP
            slower = cmd == "slower"
            delta = -step if (slower == config.SPEED_LOWER_IS_SLOWER) else step
            alpha = min(config.SPEED_ALPHA_MAX,
                        max(config.SPEED_ALPHA_MIN, state.get("speed_alpha", 1.0) + delta))
            upd["speed_alpha"] = round(alpha, 3)
            return upd
        if cmd == "switch_lesson_lang":
            target = state.get("command_arg") or self._lang(state)
            if target == self._lang(state):
                return upd
            if target not in config.SUPPORTED_LANGS:
                # Can explain a sentence in Spanish; cannot teach the whole lesson in it.
                names = " or ".join(config.LANG_NAMES[l] for l in config.SUPPORTED_LANGS)
                return _merge(self._say(state, self._T(state, "lesson_lang_unsupported", langs=names),
                                        kind="system"), upd)
            new_state = dict(state, active_lang=target)
            beats = self._localize_beats(new_state, state.get("lesson_plan") or [])
            upd.update({"active_lang": target, "speaker": config.LANG_SPEAKER[target],
                        "lesson_plan": beats})
            said = self._say(new_state, t(target, "lang_switched"), lang=target, kind="system")
            return _merge(said, upd)
        return upd

    def find_section(self, state: dict) -> dict:
        nav = state.get("nav_target") or {}
        plan = state.get("lesson_plan") or []
        if (switch := self._maybe_switch_topic(state, nav)):
            return switch
        if not plan:
            return {"answer": self._T(state, "nav_not_found")}
        idx = state.get("beat_index", 0)
        sec_order = list(dict.fromkeys(b["section_id"] for b in plan))
        first_beat = {sid: next(i for i, b in enumerate(plan) if b["section_id"] == sid) for sid in sec_order}
        cur_sec = plan[min(idx, len(plan) - 1)]["section_id"]
        cur_pos = sec_order.index(cur_sec)
        target: int | None = None
        kind = nav.get("kind")
        if kind == "prev":
            target = first_beat[sec_order[max(0, cur_pos - 1)]] if idx == first_beat[cur_sec] else first_beat[cur_sec]
        elif kind == "next":
            target = first_beat[sec_order[cur_pos + 1]] if cur_pos + 1 < len(sec_order) else len(plan)
        elif kind == "index":
            n = int(nav.get("value") or 0)
            if 1 <= n <= len(sec_order):
                target = first_beat[sec_order[n - 1]]
        elif kind == "topic":
            sr = self._store(state).get("section_retriever")
            hit = sr.best(str(nav.get("value") or "")) if sr else None
            if hit and hit.score >= config.RETRIEVAL_TAU * 0.6:
                target = first_beat.get(hit.doc.section_id)
        if target is None:
            return {"answer": self._T(state, "nav_not_found"), "nav_target": None}
        self.d.emit("navigate", kind=kind, to_beat=target)
        return {"beat_index": target, "beat_spoken": False, "heard_cursor": None,
                "answer": None, "nav_target": None}

    def _maybe_switch_topic(self, state: dict, nav: dict) -> dict | None:
        """
        "I wanted respiration, not reproduction" is a request for a different
        LESSON, not a different section of this one. Navigation only ever
        searched inside the current lesson, so the tutor answered "I couldn't
        find a part about that" and carried on with the wrong subject.

        Only explicit switch wording gets here ("instead", "I wanted X",
        "change the topic"); "go to the part about valves" still navigates
        inside the lesson. The old material is left in place until the new
        fetch succeeds, so a topic that does not exist costs nothing.
        """
        topic = str(nav.get("value") or "").strip()
        if nav.get("kind") != "topic" or not topic:
            return None
        if not intent_mod.is_topic_switch(state.get("user_utterance") or ""):
            return None
        self.d.emit("switch_topic", to=topic, from_topic=state.get("topic"))
        return _merge(self._say(state, self._T(state, "switch_topic", title=topic), kind="system"),
                      {"topic": topic, "topic_switch": True, "source_kind": "topic",
                       "source_url": None, "source_title": None, "preface": None,
                       "nav_target": None, "answer": None, "user_utterance": None,
                       "onboarding_step": "done"})

    def explain(self, state: dict) -> dict:
        self._arm(state, "filler_moment")
        sentence = state.get("heard_sentence") or state.get("pending_text") or ""
        utter = state.get("user_utterance") or ""
        reply_lang = state.get("reply_lang")
        if reply_lang and reply_lang not in config.REPLY_LANGS:
            reply_lang = None                                      # no voice for it: answer in the study language
        lang = reply_lang or self._lang(state)
        system = (
            f"You are a patient tutor speaking aloud to a {state.get('grade') or 'school'} student. "
            f"Respond in {self._lang_name(lang)} in at most two short sentences. {self.EAR_RULES} "
            "If asked to translate, translate faithfully. If asked what a word means, define it simply."
        )
        user = f"The student just heard this sentence: \"{sentence}\"\nThe student said: \"{utter}\""
        answer = self.d.llm_strong.complete(system, user).strip()
        if not answer:
            key = "explain_fallback_lang" if reply_lang else "explain_fallback"
            answer = f"{self._T(state, key)} {sentence}".strip()
            lang = self._lang(state)
        ex = (state.get("recent_exchanges") or []) + [{"q": utter, "a": answer}]
        return {"answer": answer, "reply_lang": lang if lang != self._lang(state) else None,
                "recent_exchanges": ex[-config.RECENT_EXCHANGES_KEEP:]}

    def qa_retrieve(self, state: dict) -> dict:
        utter = state.get("user_utterance") or ""
        query = utter
        recent = state.get("recent_exchanges") or []
        # "and why?" leans on the previous question; "What is chlorophyll?" must
        # not be polluted with it. See intent.is_follow_up.
        if recent and intent_mod.is_follow_up(utter):
            query = f"{recent[-1]['q']} {utter}"
        retriever = self._store(state).get("retriever")
        hits = retriever.search(query, k=config.RETRIEVAL_TOP_K) if retriever else []
        score = hits[0].score if hits else 0.0
        self.d.emit("retrieve", query=query, score=round(score, 3), n=len(hits))
        return {"retrieved": [h.as_dict() for h in hits], "retrieval_score": score, "answer_mode": None}

    LOOKUP_TOKEN = "LOOKUP"

    def direct_answer(self, state: dict) -> dict:
        """Not in the notes. Most such questions are trivial for the model
        ("what is the capital of France", "what does chlorophyll mean"), so ask
        it first; a web search is only worth its latency when the model itself
        says it needs one. Retrieval hits are passed along as context in case
        the notes were partially relevant."""
        self._arm(state, "filler_check")
        utter = state.get("user_utterance") or ""
        lang = state.get("reply_lang") or self._lang(state)
        recent = state.get("recent_exchanges") or []
        system = (
            f"You are a friendly tutor speaking aloud to a {state.get('grade') or 'school'} student. "
            f"The lesson notes do not cover this question. If you can answer it confidently from general "
            f"knowledge (definitions, everyday facts, school-level science, maths, history, geography), "
            f"answer in {self._lang_name(lang)} in at most two short sentences. {self.EAR_RULES} "
            f"If it needs current or very specific information you are not sure about (recent events, "
            f"prices, schedules, statistics, local details, little-known people), reply with exactly the "
            f"single word {self.LOOKUP_TOKEN} and nothing else."
        )
        history = "\n".join(f"Q: {e['q']}\nA: {e['a']}" for e in recent)
        user = f"Recent exchanges:\n{history or '(none)'}\n\nQuestion: {utter}"
        reply = self.d.llm_strong.complete(system, user).strip()
        if not reply or self.LOOKUP_TOKEN in reply.upper().split()[:3] or len(reply) < 4:
            self.d.emit("answer_mode", mode="lookup", question=utter)
            return {"answer_mode": "web"}
        self.d.emit("answer_mode", mode="direct", question=utter)
        ex = recent + [{"q": utter, "a": reply}]
        return {"answer": reply, "answer_mode": "direct",
                "recent_exchanges": ex[-config.RECENT_EXCHANGES_KEEP:]}

    def web_search_node(self, state: dict) -> dict:
        self._arm(state, "filler_check")
        born = state.get("born_turn_id", 0)
        delay = self.d.stress_delay_ms
        if delay > 0:                                              # PS stress test: fixed tool delay
            deadline = time.perf_counter() + delay / 1000.0
            while time.perf_counter() < deadline:
                if self.d.clock.current() != born:
                    self.d.emit("web_aborted", born=born, live=self.d.clock.current())
                    return {"retrieved": [], "web_aborted": True}   # learner moved on mid-tool
                time.sleep(0.02)
        results = self.d.web_search(state.get("user_utterance") or "") or []
        retrieved = [{"text": r.get("snippet", ""), "section_id": "", "section_title": r.get("title", ""),
                      "score": 0.0, "source": r.get("url", "web")} for r in results]
        self.d.emit("web_search", n=len(retrieved))
        return {"retrieved": retrieved}

    def compose_answer(self, state: dict) -> dict:
        utter = state.get("user_utterance") or ""
        retrieved = state.get("retrieved") or []
        lang = state.get("reply_lang") or self._lang(state)
        recent = state.get("recent_exchanges") or []
        answer = ""
        if retrieved and not state.get("web_aborted"):
            self._arm(state, "filler_check")
            system = (
                f"You are a friendly tutor speaking aloud to a {state.get('grade') or 'school'} student. "
                f"Answer in {self._lang_name(lang)} in at most two short sentences. {self.EAR_RULES} "
                "Use only the material given. If it does not answer the question, say you're not sure."
            )
            material = "\n".join(f"- ({r.get('section_title') or r.get('source')}) {r['text']}" for r in retrieved[:4])
            history = "\n".join(f"Q: {e['q']}\nA: {e['a']}" for e in recent)
            user = f"Material:\n{material}\n\nRecent exchanges:\n{history or '(none)'}\n\nQuestion: {utter}"
            answer = self.d.llm_strong.complete(system, user).strip()
        if not answer:                                             # deterministic extractive fallback
            top = retrieved[0] if retrieved else None
            if top and not top.get("section_id"):                  # web: pick the snippet that overlaps the question most
                from agents.retrieval import content_tokens
                q = set(content_tokens(utter))
                top = max(retrieved, key=lambda r: len(q & set(content_tokens(r["text"]))))
            if top and top.get("section_id"):
                sents = best_sentences(top["text"], utter, n=2)
                offer = self._T(state, "go_back_offer", title=top.get("section_title") or "")
                answer = f"{self._T(state, 'from_notes')} {sents} {offer}"
            elif top:
                answer = f"{self._T(state, 'from_web')} {top['text']}"
            else:
                answer = self._T(state, "not_found_answer")
        ex = recent + [{"q": utter, "a": answer}]
        mode = "notes" if state.get("retrieval_score", 0.0) >= config.RETRIEVAL_TAU else "web"
        return {"answer": answer, "answer_mode": mode, "recent_exchanges": ex[-config.RECENT_EXCHANGES_KEEP:]}

    def discard(self, state: dict) -> dict:
        self.d.emit("discard", born=state.get("born_turn_id"), live=self.d.clock.current(),
                    answer=(state.get("answer") or "")[:80])
        return {"stale_drops": state.get("stale_drops", 0) + 1, "answer": None,
                "queued_request": None, "retrieved": []}

    def resume_controller(self, state: dict) -> dict:
        """The only node that speaks the learner's answers. Speaks, then hands off."""
        updates: list[dict] = []
        answer = state.get("answer")
        intent = state.get("intent")
        onboarding = state.get("onboarding_step") != "done"
        if answer:
            kind = "system" if onboarding or intent in (None, "unknown", "session", "navigate") else "answer"
            updates.append(self._say(state, answer, lang=state.get("reply_lang") or None, kind=kind))
            if not onboarding and not state.get("paused") and intent in ("question", "explain") \
                    and not state.get("queued_request"):
                updates.append(self._say(state, self._T(state, "bridge"), kind="system"))
        return _merge(*updates, {"answer": None, "reply_lang": None, "event": None,
                                 "user_utterance": None})

    # ------------------------------------------------------------------ routers
    def route_event(self, state: dict) -> str:
        ev = (state.get("event") or {}).get("type")
        if ev == "user_barge_in":
            return "user_barge_in"
        if ev == "lesson_complete" or state.get("lesson_done"):
            return "lesson_complete"
        if state.get("paused") or state.get("onboarding_step") != "done":
            return "stay_parked"
        return "playback_confirmed"

    def route_after_interrupt(self, state: dict) -> str:
        step = state.get("onboarding_step")
        if step == "language":
            return "onboarding_language"
        if step == "source":
            return "onboarding_source"
        return "lesson"

    def route_onboarding(self, state: dict) -> str:
        if state.get("lesson_done"):
            return "quit"
        if state.get("answer"):
            return "ask"
        if state.get("onboarding_step") == "source":
            return "next"
        return state.get("source_kind") or "ask"

    def route_parse(self, state: dict) -> str:
        return "ok" if state.get("sections") else "unreadable"

    def route_fetch(self, state: dict) -> str:
        return "ok" if state.get("sections") else "not_found"

    def route_intent(self, state: dict) -> str:
        return state.get("intent") or "unknown"

    def route_session(self, state: dict) -> str:
        return state.get("session_cmd") or "continue"

    def route_nav(self, state: dict) -> str:
        if state.get("topic_switch"):
            return "switch"                       # fetch a whole new lesson
        return "not_found" if state.get("answer") else "found"

    def route_retrieval(self, state: dict) -> str:
        if state.get("retrieval_score", 0.0) >= config.RETRIEVAL_TAU:
            return "grounded"
        # Without a real model there is nothing to ask; go straight to the web.
        if getattr(self.d.llm_strong, "provider", "stub") == "stub":
            return "needs_web"
        return "direct"

    def route_direct(self, state: dict) -> str:
        return "answered" if state.get("answer") else "lookup"

    def fence_check(self, state: dict) -> str:
        return "current" if state.get("born_turn_id", 0) == self.d.clock.current() else "stale"

    def route_after_teach(self, state: dict) -> str:
        if state.get("lesson_done"):
            return "finished"
        if state.get("queued_request"):
            return "queued"
        return "wait"

    def route_after_speak(self, state: dict) -> str:
        if state.get("queued_request") and state.get("onboarding_step") == "done":
            return "queued"
        if state.get("onboarding_step") != "done" or state.get("paused") or state.get("lesson_done"):
            return "done"
        if state.get("intent") in (None, "unknown"):
            return "done"
        return "resume_lesson"
