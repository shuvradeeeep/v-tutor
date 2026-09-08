"""Regressions from the 2026-09-09 live session (evidence/demo-1788896902.csv):
a request about the question became the topic, and a topic-plus-class said
mid-lesson was answered instead of switching the lesson."""
from __future__ import annotations

import pytest

from agents.intent import classify_rules, is_confusion, parse_lesson_request


@pytest.mark.parametrize("utter", [
    "Yo yo, ask me the language again.",
    "ask me the question again",
    "can you ask that again?",
    "bro ask again",
    "what did you ask",
])
def test_requests_about_the_question_are_not_answers(utter):
    assert is_confusion(utter)


@pytest.mark.parametrize("utter", ["photosynthesis", "the heart for class six", "English"])
def test_real_answers_are_not_confusion(utter):
    assert not is_confusion(utter)


@pytest.mark.parametrize("utter,topic", [
    ("photosynthesis for glass 6th", "photosynthesis"),
    ("photosynthesis for class 6", "photosynthesis"),
    ("I want to study the heart for class six", "the heart"),
    ("water cycle class 7", "water cycle"),
])
def test_topic_plus_class_mid_lesson_is_a_new_lesson(utter, topic):
    assert parse_lesson_request(utter) == topic
    c = classify_rules(utter)
    assert c is not None and c.intent == "navigate"
    assert c.nav_target == {"kind": "topic", "value": topic}


@pytest.mark.parametrize("utter", [
    "how many chambers does the heart have",
    "what is photosynthesis?",
    "is this for class 6?",
    "tell me more about valves",
    "slower",
])
def test_questions_and_commands_are_not_lesson_requests(utter):
    assert parse_lesson_request(utter) is None
