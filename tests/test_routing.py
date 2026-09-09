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


# --- a live session where the tutor ignored the learner --------------------
# "Can you pause for a while while I come?" was answered ("Sure, I will wait
# for you") and then the lesson carried straight on; and "Let's continue the
# session now. Let's start with respiration." was read as a bare "continue",
# losing the new topic completely.

def test_polite_pause_requests_are_pauses():
    from agents.intent import classify_rules
    for utter in ("Can you pause for a while while I come?", "Just pause please.",
                  "could you wait a moment", "wait for me", "give me a minute",
                  "hold on a sec", "I'll be right back", "let's take a break",
                  "please pause"):
        c = classify_rules(utter)
        assert c and c.session_cmd == "pause", utter


def test_a_question_containing_the_word_pause_is_not_a_pause():
    from agents.intent import classify_rules
    c = classify_rules("why does the heart pause between beats")
    assert c and c.intent == "question"


def test_continue_plus_new_topic_does_both():
    from agents.intent import classify_rules, split_compound
    utter = "Let's continue the session now. Let's start with respiration. Tell me what it is."
    parts = split_compound(utter, paused=True)
    assert len(parts) == 2
    first = classify_rules(parts[0], paused=True)
    assert first and first.session_cmd == "continue"
    second = classify_rules(parts[1], paused=True)
    assert second and second.nav_target == {"kind": "topic", "value": "respiration"}


def test_start_with_names_a_new_lesson():
    from agents.intent import parse_topic_switch
    assert parse_topic_switch("let's start with respiration") == "respiration"
    assert parse_topic_switch("begin with the water cycle") == "the water cycle"


def test_switch_target_is_what_follows_the_marker():
    """Regression: "can we switch the topic to volleyball now" was parsed as
    "can we switch the volleyball now" and taught as Dead or Alive Xtreme."""
    from agents.intent import parse_topic_switch
    assert parse_topic_switch("Can we switch the topic to volleyball now?") == "volleyball"
    assert parse_topic_switch("I want to switch the topic to sexy artists.") == "sexy artists"
    assert parse_topic_switch("change the topic to the water cycle please") == "the water cycle"
    assert parse_topic_switch("I want to learn football now.") == "football"


def test_naming_a_subject_switches_but_asking_for_detail_does_not():
    from agents.intent import is_topic_switch
    assert is_topic_switch("I want to learn football now")
    assert is_topic_switch("can we do algebra")
    assert not is_topic_switch("I want to learn more about valves")
    assert not is_topic_switch("I want to learn how valves work")


def test_open_ended_topic_switch_intent():
    from agents.intent import classify_rules
    for utter in ("I have to change the topic.", "I want to change the topic", "change the topic", "topic badal do"):
        c = classify_rules(utter)
        assert c and c.intent == "navigate" and c.nav_target == {"kind": "topic", "value": ""}, utter



def test_misheard_hindi_pause_still_pauses():
    """Whisper wrote "zara ruko" as "Zara Rukul." """
    from agents.intent import classify_rules
    for utter in ("Zara Rukul.", "zara ruko", "thoda rukiye", "ruk jao"):
        c = classify_rules(utter)
        assert c and c.session_cmd == "pause", utter


def test_lets_stop_quits_without_needing_the_model():
    from agents.intent import classify_rules
    for utter in ("Let's stop now.", "we're done", "that's all"):
        c = classify_rules(utter)
        assert c and c.session_cmd == "quit", utter


def test_session_questions_are_meta():
    from agents.intent import is_meta_question
    assert is_meta_question("can you just tell me how long will this teaching go on") == "length"
    assert is_meta_question("how much is left") == "length"
    assert is_meta_question("are we almost done") == "length"
    assert is_meta_question("what are we studying") == "topic"
    assert is_meta_question("who are you") == "identity"
    # subject questions must not be captured
    assert is_meta_question("how long does a heartbeat last") is None
    assert is_meta_question("how many chambers does the heart have") is None


def test_initials_do_not_end_a_sentence():
    from agents.material import split_sentences
    out = split_sentences("William G. Morgan created volleyball in 1895. He was a director.")
    assert out[0].startswith("William G. Morgan created")
    assert len(out) == 2


# --- Hindi / Hinglish topic changes ---------------------------------------
# Spoken Hindi came back from Whisper in Urdu script, was retrieved (score
# 0.006), web-searched, and answered with a paragraph of Urdu prose read out
# by an English voice. The transcription is fixed in stt/transcriber.py; these
# cover the phrasings themselves.

def test_hinglish_topic_change_is_a_switch():
    from agents.intent import classify_rules
    cases = {
        "topic change krte hai mujhe virat kohli ke bare me janna hai": "virat kohli",
        "topic change karte hain mujhe photosynthesis ke baare mein janna hai": "photosynthesis",
        "chalo topic badal do mujhe cricket ke bare me janna hai": "cricket",
        "mujhe virat kohli ke bare me janna hai": "virat kohli",
        "topic change karo volleyball": "volleyball",
    }
    for utter, wanted in cases.items():
        c = classify_rules(utter)
        assert c and c.intent == "navigate", utter
        assert c.nav_target == {"kind": "topic", "value": wanted}, (utter, c.nav_target)


def test_devanagari_topic_change_is_a_switch():
    from agents.intent import classify_rules
    c = classify_rules("मुझे विराट कोहली के बारे में जानना है")
    assert c and c.nav_target == {"kind": "topic", "value": "विराट कोहली"}


def test_hindi_confusion_and_questions_are_not_topic_changes():
    from agents.intent import classify_rules
    assert classify_rules("mujhe samajh nahi aaya").intent == "explain"
    assert classify_rules("iska matlab kya hai").intent == "explain"
    assert classify_rules("हृदय में कितने कक्ष होते हैं").intent == "question"


def test_a_language_the_tutor_cannot_teach_is_decoded_as_one_it_can():
    """Spoken Hindi is routinely detected as Urdu; the transcript then arrives
    in Arabic script and is useless to every stage after it."""
    from stt.transcriber import WhisperTranscriber
    t = WhisperTranscriber.__new__(WhisperTranscriber)
    t.allowed_languages = {"en", "hi"}
    t.language_aliases = {"ur": "hi", "pa": "hi"}
    t.language_fallback = "en"
    assert t._allowed("ur") == "hi"
    assert t._allowed("pa") == "hi"
    assert t._allowed("hi") == "hi"
    assert t._allowed("en") == "en"
    assert t._allowed("ja") == "en"          # no alias: the fallback
    t.allowed_languages = set()              # unrestricted
    assert t._allowed("ur") == "ur"


def test_unspeakable_scripts_are_not_read_aloud():
    from agents.material import readable_in
    urdu = "کولیسٹرول ہمارے جسم اور، ہائی کولیسٹرول والے افراد کو کم"
    assert not readable_in(urdu, "en")
    assert not readable_in(urdu, "hi")
    assert readable_in("The heart has four chambers.", "en")
    assert readable_in("हृदय में चार कक्ष होते हैं।", "hi")
    assert readable_in("The heart has four chambers.", "hi")   # Hinglish is fine
