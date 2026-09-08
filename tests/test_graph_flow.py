"""End-to-end graph behaviour through the runner, all offline."""
from __future__ import annotations


def texts(speaker, n=0):
    return [l.text for l in speaker.lines[n:]]


def test_onboarding_asks_language_then_topic_then_teaches(runner, speaker):
    runner.start()
    assert "Which language" in speaker.lines[0].text
    runner.barge_in("English")
    assert runner.state["active_lang"] == "en"
    assert "What shall we study" in speaker.lines[-1].text
    runner.barge_in("the heart for class six")
    st = runner.state
    assert st["onboarding_step"] == "done"
    assert st["topic"] == "the heart" and st["grade"] == "class 6"
    assert len(st["lesson_plan"]) > 3
    last = speaker.lines[-1]
    assert "Let's begin" in last.text and "muscular organ" in last.text
    assert last.lang == "en" and last.speaker == runner.state["speaker"]


def test_missing_grade_is_asked_once(runner, speaker):
    runner.start()
    runner.barge_in("English")
    runner.barge_in("the heart")
    assert "which class" in speaker.lines[-1].text
    runner.barge_in("class six")
    assert runner.state["grade"] == "class 6" and runner.state["topic"] == "the heart"
    assert runner.state["onboarding_step"] == "done"


def test_hindi_choice_sets_hindi_voice(runner, speaker):
    from config import LANG_SPEAKER
    runner.start()
    runner.barge_in("hindi")
    assert runner.state["active_lang"] == "hi"
    assert runner.state["speaker"] == LANG_SPEAKER["hi"]
    assert speaker.lines[-1].lang == "hi"


def test_playback_confirmed_advances_beats(lesson, speaker):
    b0 = lesson.state["beat_index"]
    lesson.confirm_playback()
    assert lesson.state["beat_index"] == b0 + 1
    assert "72 times per minute" in speaker.lines[-1].text


def test_pause_blocks_advance_but_answers_questions(lesson, speaker):
    lesson.barge_in("pause")
    assert lesson.state["paused"] is True
    n = len(speaker.lines)
    lesson.confirm_playback()
    assert len(speaker.lines) == n, "paused: playback_confirmed must not advance"
    lesson.barge_in("How many chambers does the heart have?")
    assert any("four chambers" in x.lower() for x in texts(speaker, n))
    assert lesson.state["paused"] is True
    lesson.barge_in("continue")
    assert lesson.state["paused"] is False
    assert speaker.lines[-1].text


def test_in_notes_question_answered_then_lesson_resumes_from_sentence(lesson, speaker):
    n = len(speaker.lines)
    lesson.barge_in("How many chambers does the heart have?", words_heard=12)
    new = texts(speaker, n)
    assert any("four chambers" in x.lower() for x in new)
    assert any("back to where we were" in x for x in new)          # bridge
    assert "muscular organ" in new[-1]                             # replays the interrupted sentence
    assert lesson.state["recent_exchanges"][-1]["q"].startswith("How many")


def test_out_of_notes_question_goes_to_web(lesson, speaker):
    n = len(speaker.lines)
    lesson.barge_in("How many moons does Mars have?")
    assert any("Phobos" in x for x in texts(speaker, n))


def test_backchannel_resumes_without_bridge(lesson, speaker):
    n = len(speaker.lines)
    lesson.barge_in("mm-hmm", words_heard=4)
    new = texts(speaker, n)
    assert not any("back to where we were" in x for x in new)
    assert len(new) == 1 and "muscular organ" in new[0]


def test_barge_in_without_cursor_means_heard_everything(lesson, speaker):
    lesson.barge_in("okay")
    assert "72 times per minute" in speaker.lines[-1].text        # moved on to the next beat


def test_slower_changes_alpha_in_the_configured_direction(lesson, speaker):
    import config
    a0 = lesson.state["speed_alpha"]
    lesson.barge_in("slower")
    a1 = lesson.state["speed_alpha"]
    assert (a1 < a0) == config.SPEED_LOWER_IS_SLOWER
    assert speaker.lines[-1].speed_alpha == a1


def test_navigate_by_topic_jumps_to_section(lesson, speaker):
    lesson.barge_in("go back to the part about chambers")
    st = lesson.state
    assert st["lesson_plan"][st["beat_index"]]["section_title"] == "Chambers"
    # A section change is introduced like a teacher would, not read as a heading.
    assert speaker.lines[-1].text.startswith("Now, let's look at Chambers.")


def test_navigate_prev_next(lesson):
    lesson.barge_in("skip this")
    assert lesson.state["lesson_plan"][lesson.state["beat_index"]]["section_title"] == "Chambers"
    lesson.barge_in("go back")
    assert lesson.state["lesson_plan"][lesson.state["beat_index"]]["section_title"] == "The Heart"


def test_explain_uses_heard_sentence(lesson, speaker):
    lesson.barge_in("what does that mean", words_heard=12)
    ans = [l for l in speaker.lines if l.text.startswith("Let me say that again")]
    assert ans, "fallback explain should repeat the heard sentence"
    assert "muscular organ" in ans[-1].text


def test_compound_request_is_drained(lesson, speaker):
    lesson.barge_in("go back to the part about chambers and explain it simpler")
    st = lesson.state
    assert st["queued_request"] is None
    assert st["lesson_plan"][st["beat_index"]]["section_title"] == "Chambers"
    expl = [l.text for l in speaker.lines if l.text.startswith("Let me say that again")]
    assert expl, "second clause (explain) was not executed"
    assert "four chambers" in expl[-1], "explain must refer to the section just read, not the old sentence"


def test_unknown_asks_once_then_carries_on(lesson, speaker):
    lesson.barge_in("uh")
    assert "didn't catch that" in speaker.lines[-1].text
    n = len(speaker.lines)
    lesson.barge_in("uh")
    assert not any("didn't catch" in x for x in texts(speaker, n))
    assert lesson.state["clarify_count"] == 0


def test_quit_ends_session(lesson, speaker):
    lesson.barge_in("that's enough for today")
    assert "Bye" in speaker.lines[-1].text
    assert lesson.finished


def test_lesson_completes(lesson, speaker):
    for _ in range(40):
        if lesson.finished:
            break
        lesson.confirm_playback()
    assert lesson.state["lesson_done"] is True
    assert "end of the lesson" in speaker.lines[-1].text
    assert lesson.finished


def test_switch_lesson_language_changes_voice_and_rewinds_sentence(lesson, speaker):
    from config import LANG_SPEAKER
    lesson.barge_in("teach the whole lesson in hindi", words_heard=12)
    st = lesson.state
    assert st["active_lang"] == "hi" and st["speaker"] == LANG_SPEAKER["hi"]
    assert speaker.lines[-1].lang == "hi" and speaker.lines[-1].speaker == LANG_SPEAKER["hi"]


def test_quit_works_during_onboarding(runner, speaker):
    runner.start()
    runner.barge_in("English")
    runner.barge_in("that's enough for today")
    assert "Bye" in speaker.lines[-1].text
    assert runner.finished


# --- onboarding must not treat every noise as an answer --------------------
# Regression: a live session turned "hi" into a topic (the tutor then asked
# which class it was for), "I didn't understand the question" into a Wikipedia
# lookup, and "just stop" into a lesson about Just Stop Oil.

def test_greeting_is_not_a_topic(runner, speaker):
    runner.start()
    runner.barge_in("English")
    runner.barge_in("hi")
    assert runner.state["topic"] is None
    assert "What shall we study" in speaker.lines[-1].text     # asked again, not "which class?"
    runner.barge_in("the heart for class six")                 # and the real answer still lands
    assert runner.state["topic"] == "the heart"


def test_confusion_re_asks_the_question(runner, speaker):
    runner.start()
    runner.barge_in("English")
    runner.barge_in("what did you ask?")
    assert runner.state["topic"] is None
    assert "What shall we study" in speaker.lines[-1].text
    runner.barge_in("I didn't understand the question")        # twice in a row is still fine
    assert runner.state["topic"] is None
    assert runner.state["onboarding_step"] == "source"
    runner.barge_in("the heart for class six")
    assert runner.state["onboarding_step"] == "done" and runner.state["topic"] == "the heart"


def test_confusion_while_being_asked_the_class_re_asks_the_class(runner, speaker):
    runner.start()
    runner.barge_in("English")
    runner.barge_in("the heart")
    assert "which class" in speaker.lines[-1].text
    runner.barge_in("sorry, what?")
    assert "which class" in speaker.lines[-1].text              # the class, not the topic
    assert runner.state["topic"] == "the heart"
    runner.barge_in("class six")
    assert runner.state["grade"] == "class 6"


def test_bare_stop_quits_during_onboarding(runner, speaker):
    runner.start()
    runner.barge_in("English")
    runner.barge_in("just stop")
    assert runner.state["topic"] is None                        # not a lesson about Just Stop Oil
    assert "Bye" in speaker.lines[-1].text
    assert runner.finished


def test_a_real_topic_starting_with_a_question_word_still_works(runner, speaker):
    """The guard matches whole utterances only: these are topics, not confusion."""
    runner.start()
    runner.barge_in("English")
    runner.barge_in("explain photosynthesis for class six")
    assert runner.state["onboarding_step"] == "done"
    assert runner.state["topic"] == "photosynthesis" and runner.state["grade"] == "class 6"


def test_explain_in_other_language_is_one_off_then_lesson_continues(lesson, speaker):
    from config import LANG_SPEAKER
    n = len(speaker.lines)
    lesson.barge_in("say that in hindi", words_heard=12)
    new = speaker.lines[n:]
    # With the stub LLM the fallback apologises in the study language, but the
    # routing is what we test: reply_lang was requested, lesson stays English.
    assert lesson.state["active_lang"] == "en"
    assert lesson.state["speaker"] == LANG_SPEAKER["en"]
    assert new[-1].lang == "en" and "muscular organ" in new[-1].text


def test_explain_in_other_language_uses_that_voice_when_llm_answers(runner, speaker):
    from config import LANG_SPEAKER
    from conftest import make_deps, onboard
    from agents.graph import TutorRunner

    class SpanishLLM:
        provider = model = "test"
        def complete(self, system, user):
            return "El corazon es un organo muscular." if "student just heard" in user else ""

    r = TutorRunner(make_deps(speaker=speaker, llm_strong=SpanishLLM()), "es")
    onboard(r)
    n = len(speaker.lines)
    r.barge_in("explain that in spanish", words_heard=12)
    new = speaker.lines[n:]
    assert new[0].lang == "es" and new[0].speaker == LANG_SPEAKER["es"]      # one-off reply
    assert "organo muscular" in new[0].text
    assert new[-1].lang == "en" and new[-1].speaker == LANG_SPEAKER["en"]    # lesson continues in English
    assert r.state["active_lang"] == "en" and r.state["reply_lang"] is None


def test_whole_lesson_in_unsupported_language_is_refused_politely(lesson, speaker):
    lesson.barge_in("teach the whole lesson in spanish")
    assert any("only teach the whole lesson in English or Hindi" in l.text for l in speaker.lines)
    assert lesson.state["active_lang"] == "en"


def test_preset_language_skips_the_spoken_question(speaker):
    from conftest import make_deps
    from agents.graph import TutorRunner
    from config import LANG_SPEAKER
    r = TutorRunner(make_deps(speaker=speaker), "preset", preset_lang="hi")
    r.start()
    assert r.state["active_lang"] == "hi" and r.state["speaker"] == LANG_SPEAKER["hi"]
    assert "Which language" not in speaker.lines[0].text
    assert speaker.lines[0].lang == "hi"                                      # straight to "what shall we study?"


def test_language_button_press_during_question(runner, speaker):
    from config import LANG_SPEAKER
    runner.start()
    runner.press_language_button("hi")
    assert runner.state["active_lang"] == "hi"
    assert speaker.lines[-1].lang == "hi" and speaker.lines[-1].speaker == LANG_SPEAKER["hi"]


def test_unsupported_study_language_is_explained_and_reasked(runner, speaker):
    runner.start()
    runner.barge_in("Spanish")
    assert runner.state["onboarding_step"] == "language"
    assert "only teach the whole lesson in English or Hindi" in speaker.lines[-1].text
    assert "Which language" in speaker.lines[-1].text
    runner.barge_in("English")
    assert runner.state["active_lang"] == "en"


# --- "I wanted respiration, not reproduction" ------------------------------
# Regression: topic navigation only ever searched the CURRENT lesson, so a
# request for a different subject answered "I couldn't find a part about that"
# and carried on teaching the wrong one.

def _two_topic_runner(speaker, clock):
    from agents.graph import TutorRunner
    from agents.material import sections_from_plaintext
    from conftest import fixture_sections, make_deps

    resp = sections_from_plaintext(
        "== Respiration ==\nRespiration releases energy from food inside cells. "
        "Breathing moves oxygen into the lungs.\n", title="Respiration", source_url="fixture://resp")
    asked: list[str] = []

    def wiki(topic: str, lang: str):
        asked.append(topic)
        if "respirat" in topic.lower():
            return (resp, "en", "fixture://resp")
        return (fixture_sections(), "en", "fixture://heart")

    return TutorRunner(make_deps(speaker=speaker, clock=clock, wiki_fetch=wiki), "switch"), asked


def test_wrong_topic_is_switched_not_navigated(speaker, clock):
    from conftest import onboard
    runner, asked = _two_topic_runner(speaker, clock)
    onboard(runner)
    assert runner.state["topic"] == "the heart"

    runner.barge_in("oh sorry, I wanted to learn respiration not the heart")

    assert asked[-1] == "respiration"
    assert runner.state["topic"] == "respiration"
    assert runner.state["topic_switch"] is False              # cleared once ingested
    said = " ".join(l.text for l in speaker.lines[-4:])
    assert "switch to respiration" in said
    assert "Respiration releases energy" in said
    assert "couldn't find a part" not in said


def test_grade_survives_a_topic_switch(speaker, clock):
    from conftest import onboard
    runner, _ = _two_topic_runner(speaker, clock)
    onboard(runner)
    assert runner.state["grade"] == "class 6"
    runner.barge_in("change the topic to respiration")
    assert runner.state["topic"] == "respiration" and runner.state["grade"] == "class 6"


def test_in_lesson_navigation_still_navigates(lesson, speaker):
    """No switch wording: stay in this lesson, do not fetch anything new."""
    lesson.barge_in("go to the part about chambers", words_heard=5)
    assert lesson.state["topic"] == "the heart"
    assert lesson.state["topic_switch"] is False
    assert "four chambers" in speaker.lines[-1].text


# --- session memory --------------------------------------------------------
# The tutor used to answer each question as if it were the first: only two
# exchanges were kept and neither the explain path nor the intent classifier
# saw them, so it re-explained terms it had just explained.

class RecordingLLM:
    """Answers plausibly and keeps every prompt it was sent."""

    provider = "recording"
    model = "recording"

    def __init__(self, reply: str = "Here is a short answer."):
        self.reply = reply
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return self.reply


def test_answers_carry_the_conversation_and_the_lesson_so_far(speaker, clock):
    from agents.graph import TutorRunner
    from conftest import make_deps, onboard
    llm = RecordingLLM()
    r = TutorRunner(make_deps(speaker=speaker, clock=clock, llm_strong=llm), "mem")
    onboard(r)
    r.barge_in("how many chambers does the heart have")
    r.barge_in("and why is the left one strongest")

    last = llm.prompts[-1][1]
    assert "Lesson: The Heart (for class 6)" in last
    assert "Covered so far:" in last
    assert "how many chambers" in last, "the earlier question is missing from the prompt"


def test_history_keeps_more_than_two_exchanges(speaker, clock):
    import config
    from agents.graph import TutorRunner
    from conftest import make_deps, onboard
    assert config.RECENT_EXCHANGES_KEEP >= 6
    llm = RecordingLLM()
    r = TutorRunner(make_deps(speaker=speaker, clock=clock, llm_strong=llm), "mem2")
    onboard(r)
    for q in ("what is an atrium", "what is a ventricle", "what is the aorta", "what is a valve"):
        r.barge_in(q)
    assert "what is an atrium" in llm.prompts[-1][1], "oldest question fell out of a 4-turn session"


def test_explain_sees_the_conversation_not_just_the_last_sentence(speaker, clock):
    from agents.graph import TutorRunner
    from conftest import make_deps, onboard
    llm = RecordingLLM()
    r = TutorRunner(make_deps(speaker=speaker, clock=clock, llm_strong=llm), "mem3")
    onboard(r)
    r.barge_in("how many chambers does the heart have")
    r.barge_in("what does that mean", words_heard=12)
    system, user = llm.prompts[-1]
    assert "do not repeat an explanation you have already given" in system
    assert "Conversation so far:" in user and "how many chambers" in user


def test_intent_classifier_is_given_the_context(speaker, clock):
    from agents.graph import TutorRunner
    from conftest import make_deps, onboard
    fast = RecordingLLM('{"intent": "question"}')
    r = TutorRunner(make_deps(speaker=speaker, clock=clock, llm_fast=fast), "mem4")
    onboard(r)
    r.barge_in("the other one", words_heard=6)          # no rule matches: goes to the LLM
    assert fast.prompts, "the fast model was never called"
    system, user = fast.prompts[-1]
    assert "Utterance: the other one" in user
    assert "Tutor was saying:" in user
    assert 'label ONLY the line marked "Utterance:"' in system


def test_notes_miss_gets_a_second_chance_before_saying_not_sure(speaker, clock):
    """Retrieval was confident but the chunks do not answer it: the tutor should
    fall back to general knowledge rather than 'I'm not sure'."""
    from agents.graph import TutorRunner
    from conftest import make_deps, onboard

    class NotesMissLLM:
        provider, model = "scripted", "scripted"

        def __init__(self):
            self.calls = 0

        def complete(self, system: str, user: str) -> str:
            self.calls += 1
            if "NOTES_MISS" in system:            # the grounded prompt
                return "NOTES_MISS"
            return "The left ventricle is the strongest chamber."

    llm = NotesMissLLM()
    r = TutorRunner(make_deps(speaker=speaker, clock=clock, llm_strong=llm), "miss")
    onboard(r)
    n = len(speaker.lines)
    r.barge_in("which one is the strongest")
    said = " ".join(l.text for l in speaker.lines[n:])
    assert "left ventricle is the strongest" in said
    assert "not sure" not in said
    assert llm.calls >= 2, "the general-knowledge second chance was never taken"


# --- "which class?" must only ever be answered by a class ------------------
# Regression: the tutor asked for the class, Whisper heard "Plastics", and that
# silently replaced the topic -- the learner asked for the heart and got a
# lesson on plastic.

def test_a_non_class_reply_does_not_replace_the_topic(runner, speaker):
    runner.start()
    runner.barge_in("English")
    runner.barge_in("the heart")
    assert "which class" in speaker.lines[-1].text
    runner.barge_in("Plastics")                      # misheard "class six"
    assert runner.state["topic"] == "the heart"
    assert runner.state["onboarding_step"] == "source"
    assert "which class" in speaker.lines[-1].text   # asked again, topic intact
    runner.barge_in("class six")
    assert runner.state["topic"] == "the heart" and runner.state["grade"] == "class 6"


def test_the_class_question_gives_up_rather_than_looping(runner, speaker):
    runner.start()
    runner.barge_in("English")
    runner.barge_in("the heart")
    runner.barge_in("Plastics")                        # asked again
    runner.barge_in("Plastics")                        # give up on the class, keep the topic
    assert runner.state["topic"] == "the heart" and runner.state["grade"] is None
    assert runner.state["onboarding_step"] == "done"
    assert "Let's begin" in speaker.lines[-1].text


def test_a_deliberate_switch_is_still_allowed_while_being_asked_the_class(runner, speaker):
    runner.start()
    runner.barge_in("English")
    runner.barge_in("the heart")
    runner.barge_in("actually I want to learn the topic photosynthesis")
    assert runner.state["topic"] == "photosynthesis"


def test_i_want_to_learn_the_topic_x_switches_mid_lesson(speaker, clock):
    from conftest import onboard
    runner, asked = _two_topic_runner(speaker, clock)
    onboard(runner)
    runner.barge_in("I want to learn the topic respiration")
    assert asked[-1] == "respiration"
    assert runner.state["topic"] == "respiration"
    said = " ".join(l.text for l in speaker.lines[-3:])
    assert "couldn't find a part" not in said


# --- questions about the session, not the subject --------------------------
# Regression: "how long will this teaching go on?" was retrieved (score 0.22),
# missed, web-searched, and answered "about three to four months" -- the length
# of a teaching practicum.

def test_how_long_is_answered_from_the_lesson_plan(lesson, speaker):
    n = len(speaker.lines)
    lesson.barge_in("can you tell me how long will this session go on")
    said = " ".join(l.text for l in speaker.lines[n:])
    assert "part 1 of" in said and "minutes" in said
    assert "months" not in said
    assert lesson.state["answer_mode"] == "status"


def test_what_are_we_studying(lesson, speaker):
    n = len(speaker.lines)
    lesson.barge_in("what are we studying")
    assert "The Heart" in " ".join(l.text for l in speaker.lines[n:])


def test_who_are_you(lesson, speaker):
    n = len(speaker.lines)
    lesson.barge_in("who are you")
    assert "voice tutor" in " ".join(l.text for l in speaker.lines[n:])


def test_a_subject_question_is_still_a_subject_question(lesson, speaker):
    """"How many chambers" must not be swallowed by the session-status path."""
    n = len(speaker.lines)
    lesson.barge_in("how many chambers does the heart have")
    assert "four chambers" in " ".join(l.text for l in speaker.lines[n:]).lower()


def test_naming_a_topic_the_lesson_lacks_offers_a_switch(lesson, speaker):
    """Was a dead end: "I couldn't find a part about that. Let's carry on." """
    lesson.barge_in("go to the part about the moon")
    said = " ".join(l.text for l in speaker.lines[-3:])
    assert "switch to" in said and "moon" in said
