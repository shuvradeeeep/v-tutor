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
    assert speaker.lines[-1].text.startswith("Chambers.")


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
