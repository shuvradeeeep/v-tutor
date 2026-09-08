from __future__ import annotations

import pytest

from agents.intent import classify_rules, heuristic, parse_language, parse_topic_grade, split_compound


@pytest.mark.parametrize("utter,intent,extra", [
    # session
    ("pause", "session", {"session_cmd": "pause"}),
    ("hold on", "session", {"session_cmd": "pause"}),
    ("continue", "session", {"session_cmd": "continue"}),
    ("start over", "session", {"session_cmd": "restart"}),
    ("that's enough for today", "session", {"session_cmd": "quit"}),
    # commands
    ("slower", "command", {"command": "slower"}),
    ("say that again slowly", "command", {"command": "slower"}),
    ("a bit faster please", "command", {"command": "faster"}),
    ("again", "command", {"command": "repeat"}),
    ("say that again", "command", {"command": "repeat"}),
    ("teach the whole lesson in hindi", "command", {"command": "switch_lesson_lang", "command_arg": "hi"}),
    ("switch to english", "command", {"command": "switch_lesson_lang", "command_arg": "en"}),
    # navigate
    ("go back", "navigate", {"nav_target": {"kind": "prev"}}),
    ("skip this", "navigate", {"nav_target": {"kind": "next"}}),
    ("go back to the part about chambers", "navigate", {"nav_target": {"kind": "topic", "value": "part about chambers"}}),
    ("section 3", "navigate", {"nav_target": {"kind": "index", "value": 3}}),
    # explain
    ("what does ventricle mean", "explain", {}),
    ("say that in hindi", "explain", {"reply_lang": "hi"}),
    ("explain that in simpler words", "explain", {}),
    ("I didn't understand", "explain", {}),
    # backchannel
    ("mm-hmm", "backchannel", {}),
    ("okay", "backchannel", {}),
    ("yeah right", "backchannel", {}),
    # question
    ("How many chambers does the heart have?", "question", {}),
    ("who was william harvey", "question", {}),
    # hindi
    ("धीरे", "command", {"command": "slower"}),
    ("दोबारा", "command", {"command": "repeat"}),
    ("रुको", "session", {"session_cmd": "pause"}),
    ("आगे बढ़ो", "session", {"session_cmd": "continue"}),
    ("इसका मतलब क्या है", "explain", {}),
    ("हृदय में कितने कक्ष होते हैं", "question", {}),
    ("ठीक है", "backchannel", {}),
])
def test_rules(utter, intent, extra):
    c = classify_rules(utter)
    assert c is not None, utter
    assert c.intent == intent, (utter, c)
    for k, v in extra.items():
        got = getattr(c, k)
        if isinstance(v, dict):
            for kk, vv in v.items():
                assert got.get(kk) == vv, (utter, kk, got)
        else:
            assert got == v, (utter, k, got)


def test_go_on_is_backchannel_unless_paused():
    assert classify_rules("go on").intent == "backchannel"
    assert classify_rules("go on", paused=True).session_cmd == "continue"


def test_unmatched_falls_to_heuristic():
    assert classify_rules("the mitochondria thing") is None
    assert heuristic("the mitochondria thing").intent == "question"
    assert heuristic("uh").intent == "unknown"


def test_compound_split_only_when_both_halves_classify():
    assert split_compound("go back to the part about chambers and explain it simpler") == [
        "go back to the part about chambers", "explain it simpler"]
    # 'and' inside a real question is NOT a split
    assert split_compound("what are atria and ventricles") == ["what are atria and ventricles"]


def test_topic_grade_parsing():
    assert parse_topic_grade("the heart for class six") == ("the heart", "class 6")
    assert parse_topic_grade("photosynthesis, class 6") == ("photosynthesis", "class 6")
    assert parse_topic_grade("I want to study the water cycle") == ("the water cycle", None)
    assert parse_topic_grade("6th grade fractions") == ("fractions", "class 6")
    assert parse_topic_grade("कक्षा 6 के लिए प्रकाश संश्लेषण") == ("प्रकाश संश्लेषण", "class 6")


def test_language_parsing():
    assert parse_language("English") == "en"
    assert parse_language("hindi please") == "hi"
    assert parse_language("हिंदी") == "hi"
    assert parse_language("मुझे हिंदी में पढ़ना है") == "hi"
    assert parse_language("let's go with english", detected_lang="en") == "en"


def test_reply_language_accepts_all_rime_languages():
    for name, code in [("spanish", "es"), ("french", "fr"), ("german", "de"), ("japanese", "ja")]:
        c = classify_rules(f"explain that in {name}")
        assert c.intent == "explain" and c.reply_lang == code, (name, c)
    c = classify_rules("teach the whole lesson in spanish")
    assert c.intent == "command" and c.command == "switch_lesson_lang" and c.command_arg == "es"


# --- follow-up questions ---------------------------------------------------
# Regression: "and which one is the strongest" is six words, and the cap was
# five, so it was retrieved on its own, matched nothing useful, and the tutor
# answered "I'm not sure."

def test_follow_ups_are_recognised():
    from agents.intent import is_follow_up
    for q in ("and why", "but how", "why is that", "and which one is the strongest",
              "which one is the biggest", "what about the valves", "is that the same",
              "the other one"):
        assert is_follow_up(q), q


def test_self_contained_questions_are_not_follow_ups():
    from agents.intent import is_follow_up
    for q in ("what is chlorophyll", "how many chambers does the heart have",
              "who discovered the circulation of blood in sixteen twenty eight",
              "explain photosynthesis for class six"):
        assert not is_follow_up(q), q


# --- misheard class numbers ------------------------------------------------
# Real transcripts from one session: "class six" came back as "classics",
# "glass 6" and (after the tutor asked again) "Plastics".

def test_misheard_class_is_still_a_class():
    from agents.intent import parse_topic_grade
    for utter in ("the heart for classics.", "the heart for glass 6.",
                  "the heart for clas 6", "the heart for class six"):
        topic, grade = parse_topic_grade(utter)
        assert grade == "class 6", utter
        assert "heart" in topic and "class" not in topic, utter


def test_naming_the_topic_word_is_a_switch_but_asking_for_more_is_not():
    from agents.intent import is_topic_switch, parse_topic_switch
    assert parse_topic_switch("I want to learn the topic heart.") == "heart"
    assert is_topic_switch("teach me the chapter on valves")
    assert not is_topic_switch("I want to learn more about valves")
    assert not is_topic_switch("tell me more about the aorta")
