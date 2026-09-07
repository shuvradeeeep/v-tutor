"""
Rule-first intent classification, in English and Hindi.

Why rules first: the common utterances ("slower", "again", "pause", "mm-hmm",
"go back") are short, closed-class, and must resolve in microseconds. Only what
the rules miss goes to the fast LLM; if that is unavailable, a heuristic keeps
the tutor moving. Nothing here is allowed to block the barge-in stop -- the stop
already happened before classification starts.

Hindi patterns use substring matching, not \\b: Python's \\w excludes Devanagari
vowel signs, so word boundaries land inside words.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

VALID_INTENTS = {"question", "explain", "navigate", "command", "session", "backchannel", "unknown"}


@dataclass
class Classification:
    intent: str
    command: str | None = None          # repeat | slower | faster | switch_lesson_lang
    command_arg: str | None = None      # e.g. "hi" for switch_lesson_lang
    session_cmd: str | None = None      # pause | continue | restart | quit
    nav_target: dict | None = None      # {"kind": prev|next|index|topic, "value": ...}
    reply_lang: str | None = None       # for explain: answer in this language only

    def as_updates(self) -> dict:
        return {
            "intent": self.intent, "command": self.command, "command_arg": self.command_arg,
            "session_cmd": self.session_cmd, "nav_target": self.nav_target,
            "reply_lang": self.reply_lang,
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_NUM_EN = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_NUM_HI = {"एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6,
           "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
           # romanised (Whisper often transcribes spoken Hindi in Latin script)
           "ek": 1, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6,
           "che": 6, "chah": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10, "gyarah": 11, "barah": 12}
_NUM_ANY = "|".join(list(_NUM_EN) + list(_NUM_HI)) + r"|\d{1,2}"


def _num(tok: str) -> int | None:
    tok = tok.strip().lower()
    if tok.isdigit():
        return int(tok)
    return _NUM_EN.get(tok) or _NUM_HI.get(tok)


def _en(text: str, *alts: str) -> re.Match | None:
    return re.search(r"\b(?:" + "|".join(alts) + r")\b", text, re.IGNORECASE)


def _hi(text: str, *alts: str) -> bool:
    return any(a in text for a in alts)


def _norm(utter: str) -> str:
    return re.sub(r"\s+", " ", utter.strip().lower())


def _words(utter: str) -> int:
    return len(re.findall(r"\S+", utter))


# Every language Rime's coda model can speak. Study languages are a subset
# (config.SUPPORTED_LANGS); the rest are valid for a one-off "explain that in X".
LANG_NAME_TO_CODE = {
    "hindi": "hi", "hindee": "hi", "हिंदी": "hi", "हिन्दी": "hi",
    "english": "en", "angrezi": "en", "अंग्रेज़ी": "en", "अंग्रेजी": "en", "इंग्लिश": "en",
    "spanish": "es", "español": "es", "espanol": "es", "स्पेनिश": "es",
    "french": "fr", "français": "fr", "francais": "fr", "फ्रेंच": "fr",
    "german": "de", "deutsch": "de", "जर्मन": "de",
    "italian": "it", "italiano": "it", "इतालवी": "it",
    "portuguese": "pt", "português": "pt", "portugues": "pt", "पुर्तगाली": "pt",
    "japanese": "ja", "日本語": "ja", "जापानी": "ja",
    "arabic": "ar", "العربية": "ar", "अरबी": "ar",
}
_LANG_NAMES_RX = "|".join(sorted(map(re.escape, LANG_NAME_TO_CODE), key=len, reverse=True))


def _lang_mentioned(text: str) -> str | None:
    for name, code in LANG_NAME_TO_CODE.items():
        if name in text:
            return code
    return None


# --------------------------------------------------------------------------
# onboarding parsers
# --------------------------------------------------------------------------

def parse_language(utter: str, detected_lang: str | None = None,
                   supported: tuple[str, ...] = ("en", "hi")) -> str | None:
    """Name wins; otherwise the language the learner replied in."""
    text = _norm(utter)
    named = _lang_mentioned(text)
    if named:
        return named if named in supported else None   # named but unsupported: don't guess
    if detected_lang in supported:
        return detected_lang
    from agents.material import detect_lang
    guess = detect_lang(utter)
    return guess if guess in supported and _words(utter) >= 1 else None


_GRADE_EN = re.compile(
    rf"\b(?:class|grade|std|standard|kaksha)\s*(?:of\s*)?({_NUM_ANY})(?:th|st|nd|rd)?\b"
    rf"|\b({_NUM_ANY})(?:th|st|nd|rd)?\s*(?:class|grade|standard|std|kaksha)\b", re.IGNORECASE)
_GRADE_ONLY = re.compile(
    rf"^\s*(?:class|grade|std|standard|kaksha|कक्षा|क्लास)?\s*({_NUM_ANY})(?:th|st|nd|rd|वीं|वी)?"
    rf"\s*(?:class|grade|standard|std|kaksha|कक्षा|क्लास)?\s*[.!]*$", re.IGNORECASE)


def parse_grade_only(utter: str) -> int | None:
    """'six', '6', 'class 6', 'chhe' -- the whole utterance is just a grade.
    Used when the tutor has asked 'which class?' and the topic is already known."""
    m = _GRADE_ONLY.match(utter.strip().lower())
    return _num(m.group(1)) if m else None
_GRADE_HI = re.compile(
    rf"(?:कक्षा|क्लास)\s*({_NUM_ANY})|({_NUM_ANY})\s*(?:वीं|वी)?\s*(?:कक्षा|क्लास)")

_FILLER_EN = re.compile(
    r"\b(?:i want to study|i want to learn|i'd like to study|i would like to study|"
    r"let's study|lets study|let us study|can we study|can we do|we can do|"
    r"teach me|teach|study|learn|about|the topic is|topic is|topic|please|"
    r"today|for|i|want|to|do|the chapter on|chapter on|chapter|lesson on|lesson)\b",
    re.IGNORECASE)
_FILLER_HI = ["मुझे", "पढ़ना चाहता हूँ", "पढ़ना चाहती हूँ", "पढ़ना है", "पढ़ाओ", "पढ़ाइए",
              "सिखाओ", "सिखाइए", "के बारे में", "के लिए", "विषय", "चाहिए", "आज", "हम", "पढ़ेंगे",
              "पढ़ें", "कृपया", "का पाठ", "पाठ"]
_FILLER_HINGLISH = re.compile(
    r"\b(?:mujhe|padhna hai|padhna chahta hoon|padhna chahti hoon|padhao|padhaiye|sikhao|"
    r"ke baare mein|ke bare me|ke liye|vishay|chahiye|aaj|hum|padhenge|kripya|ka paath|paath)\b",
    re.IGNORECASE)


def parse_topic_grade(utter: str) -> tuple[str | None, str | None]:
    text = utter.strip()
    grade: str | None = None
    for rx in (_GRADE_EN, _GRADE_HI):
        m = rx.search(text)
        if m:
            n = _num(next(g for g in m.groups() if g))
            if n:
                grade = f"class {n}"
                text = text[:m.start()] + " " + text[m.end():]
            break
    text = _FILLER_EN.sub(" ", text)
    text = _FILLER_HINGLISH.sub(" ", text)
    for f in _FILLER_HI:
        text = text.replace(f, " ")
    # Strip punctuation by Unicode category, NOT with \w: Python's \w excludes
    # Devanagari vowel signs and virama, which would shred Hindi words.
    text = "".join(
        ch if (unicodedata.category(ch)[0] in "LMN" or ch.isspace() or ch in "-'") else " "
        for ch in text)
    text = re.sub(r"\s+", " ", text).strip(" -'")
    topic = text if text and len(text) > 1 else None
    return topic, grade


# --------------------------------------------------------------------------
# lesson-time classification
# --------------------------------------------------------------------------

_BACKCHANNEL_EN = re.compile(
    r"^(?:(?:mm+-?hm+|mhm+|hmm+|hm+|uh-?huh|yeah|yes|yep|yup|ok|okay|right|i see|sure|"
    r"got it|alright|all right|cool|go on|fine|nice|great|good|"
    r"haan|han|ha|theek hai|thik hai|theek|achha|accha|acha|samajh gaya|samajh gayi|ji|sahi)[.,!\s]*)+$",
    re.IGNORECASE)
_BACKCHANNEL_HI = {"हाँ", "हां", "हम्म", "हम", "ठीक", "ठीक है", "अच्छा", "समझ गया", "समझ गयी",
                   "समझ गई", "जी", "सही", "हाँ जी", "जी हाँ"}

_QUESTION_START_EN = re.compile(
    r"^(?:what|why|how|when|where|who|whom|whose|which|is|are|was|were|does|do|did|can|"
    r"could|would|will|should|tell me|name|give me|"
    r"kya|kyun|kyon|kaise|kab|kahan|kaun|kitne|kitna|kitni|batao|bataiye)\b", re.IGNORECASE)
_QUESTION_HINGLISH = re.compile(r"\b(?:kya|kyun|kyon|kaise|kab|kahan|kaun|kitne|kitna|kitni|batao|bataiye)\b",
                                re.IGNORECASE)
_QUESTION_HI = ["क्या", "क्यों", "कैसे", "कब", "कहाँ", "कहां", "कौन", "कितन", "बताओ", "बताइए",
                "किस", "किसे", "किसने"]


def classify_rules(utter: str, *, paused: bool = False) -> Classification | None:
    text = _norm(utter)
    if not text:
        return None
    n = _words(text)

    # ---- session ---------------------------------------------------------
    # Romanised Hindi ("Hinglish") is what Whisper often produces for spoken
    # Hindi, so every rule carries Latin-script forms beside the Devanagari.
    if _en(text, "start over", "start again", "from the beginning", "restart", "begin again",
           "phir se shuru", "shuru se") \
            or _hi(text, "फिर से शुरू", "शुरू से"):
        return Classification("session", session_cmd="restart")
    if _en(text, "stop for today", "that's enough", "that is enough", "thats enough", "i'm done",
           "im done", "i am done", "quit", "exit", "goodbye", "bye", "see you", "stop the lesson",
           "end the lesson", "stop teaching", "band karo", "bas ho gaya", "aaj ke liye bas", "alvida") \
            or _hi(text, "बंद करो", "आज के लिए बस", "बस हो गया", "अलविदा", "बाय"):
        return Classification("session", session_cmd="quit")
    if (n <= 4 and _en(text, "continue", "carry on", "resume", "keep going", "aage badho", "aage badhe",
                       "jaari rakho", "jari rakho", "chalo aage")) \
            or (paused and _en(text, "go on", "ok", "okay", "yes", "start", "chalo", "shuru karo", "haan")) \
            or _hi(text, "जारी रखो", "जारी रखें", "आगे बढ़ो", "आगे बढ़ें", "चलो आगे") \
            or (paused and _hi(text, "चलो", "शुरू करो", "हाँ")):
        return Classification("session", session_cmd="continue")
    if re.fullmatch(r"(?:pause|stop|stop please|stop it|stop now|hold on|hang on|wait|wait up|"
                    r"one (?:sec|second|moment|minute)|wait a (?:sec|second|moment|minute)|"
                    r"just a (?:sec|second|moment|minute)|bas|ruko|rukiye|ruk jao|ek minute|ek second|"
                    r"thoda ruko)[.!]*", text) \
            or (n <= 4 and _hi(text, "रुको", "रुकिए", "रुक जाओ", "एक मिनट", "थोड़ा रुको")):
        return Classification("session", session_cmd="pause")

    # ---- command: whole-lesson language switch --------------------------
    m = re.search(r"\b(?:switch|change|teach|lesson|from now on|everything|whole|entire|rest)\b.*?"
                  rf"(?:in|to|into)\s+({_LANG_NAMES_RX})\b", text)
    if m:
        return Classification("command", command="switch_lesson_lang", command_arg=LANG_NAME_TO_CODE[m.group(1)])
    if (_hi(text, "पढ़ाओ", "पढ़ाइए", "सिखाओ", "पूरा", "सब कुछ", "अब से")
            or _en(text, "padhao", "padhaiye", "sikhao", "pura", "poora", "sab kuch", "ab se")) \
            and _lang_mentioned(text):
        return Classification("command", command="switch_lesson_lang", command_arg=_lang_mentioned(text))

    # ---- command: speed --------------------------------------------------
    if _en(text, "faster", "speed up", "quicker", "too slow", "quick", "tez", "tezi se", "jaldi") \
            or _hi(text, "तेज़", "तेज", "जल्दी"):
        return Classification("command", command="faster")
    if _en(text, "slower", "slow down", "slowly", "too fast", "slow", "dheere", "dhire", "dheeme", "aaram se") \
            or _hi(text, "धीरे", "धीमे", "आराम से"):
        return Classification("command", command="slower")

    # ---- navigate --------------------------------------------------------
    m = re.search(r"\b(?:back to|go to|jump to|take me to|the (?:part|section|bit|chapter) "
                  r"(?:about|on|of|where|with))\s+(.+?)[.?!]*$", text)
    if m and not re.fullmatch(r"(?:where we were|the lesson|it)", m.group(1)):
        return Classification("navigate", nav_target={"kind": "topic", "value": re.sub(r"^the\s+", "", m.group(1))})
    m = re.search(r"(.+?)\s*(?:वाले|वाला|वाली)\s*(?:हिस्से|हिस्सा|भाग)", text) \
        or re.search(r"(.+?)\s*के बारे में\s*(?:वाला|वाले)?\s*(?:हिस्सा|हिस्से|भाग)", text)
    if m:
        return Classification("navigate", nav_target={"kind": "topic", "value": m.group(1).strip()})
    m = re.search(rf"\b(?:section|part|chapter)\s*(?:number\s*)?({_NUM_ANY})\b", text) \
        or re.search(rf"(?:भाग|अध्याय|हिस्सा)\s*({_NUM_ANY})", text)
    if m and _num(m.group(1)):
        return Classification("navigate", nav_target={"kind": "index", "value": _num(m.group(1))})
    if _en(text, "go back", "previous", "last section", "last part", "last paragraph", "last bit",
           "before that", "back up", "rewind", "peeche", "pichla", "pichhla", "wapas", "vapas") \
            or _hi(text, "पीछे", "पिछला", "पिछले", "वापस"):
        return Classification("navigate", nav_target={"kind": "prev"})
    if _en(text, "skip", "next section", "next part", "next one", "next chapter", "move on",
           "jump ahead", "go ahead", "agla", "agle", "chhodo", "chodo", "aage wala") \
            or _hi(text, "अगला", "अगले", "छोड़ो", "छोड़ दो", "आगे वाला"):
        return Classification("navigate", nav_target={"kind": "next"})

    # ---- explain ---------------------------------------------------------
    reply_lang = None
    m = re.search(rf"\b(?:in|into)\s+({_LANG_NAMES_RX})\b", text)
    if m:
        reply_lang = LANG_NAME_TO_CODE[m.group(1)]
    else:
        # Hindi phrasing: "<language> में" / romanised "<language> me|mein" = "in <language>"
        m = re.search(rf"({_LANG_NAMES_RX})\s*(?:में|\bme\b|\bmein\b)", text)
        if m:
            reply_lang = LANG_NAME_TO_CODE[m.group(1)]
    if _en(text, "what does .+ mean", "what do you mean", "meaning", "means", "define", "definition",
           "explain", "simpler", "simple words", "simply", "easier", "easy words", "break it down",
           "break that down", "elaborate", "in other words", "didn't understand", "did not understand",
           "don't understand", "dont understand", "confused", "what is that", "what's that",
           "matlab", "samjhao", "samjhaiye", "aasan", "saral", "samajh nahi aaya", "samajh nahin aaya",
           "samajh nahi aya", "kya hota hai") \
            or _hi(text, "मतलब", "अर्थ", "समझाओ", "समझाइए", "आसान", "सरल", "समझ नहीं आया",
                   "समझ में नहीं आया", "क्या होता है", "क्या है वो", "क्या है ये") \
            or (reply_lang and _en(text, "say", "that", "this", "it", "again", "repeat", "bolo", "kaho", "batao")) \
            or (reply_lang and _hi(text, "बोलो", "कहो", "बताओ", "बताइए")):
        return Classification("explain", reply_lang=reply_lang)

    # ---- command: repeat -------------------------------------------------
    if _en(text, "again", "repeat", "once more", "say that again", "come again", "pardon", "one more time",
           "dobara", "phir se bolo", "phir se kaho", "phir se batao", "ek baar aur", "phir se") \
            or _hi(text, "दोबारा", "फिर से बोलो", "फिर से कहो", "फिर से बताओ", "एक बार और", "फिर से"):
        return Classification("command", command="repeat")

    # ---- backchannel -----------------------------------------------------
    if _BACKCHANNEL_EN.fullmatch(text) or text.strip(" .!,") in _BACKCHANNEL_HI:
        return Classification("backchannel")

    # ---- question (explicit) ---------------------------------------------
    if text.endswith("?") or _QUESTION_START_EN.match(text) or any(q in text for q in _QUESTION_HI) \
            or _QUESTION_HINGLISH.search(text):
        return Classification("question")

    return None


def heuristic(utter: str) -> Classification:
    """When neither rules nor LLM decided: long enough to be a question, else unknown."""
    return Classification("question" if _words(utter) >= 3 else "unknown")


def parse_llm_json(text: str) -> Classification | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    intent = str(obj.get("intent", "")).lower()
    if intent not in VALID_INTENTS:
        return None
    return Classification(
        intent, command=obj.get("command"), command_arg=obj.get("command_arg"),
        session_cmd=obj.get("session_cmd"), nav_target=obj.get("nav_target"),
        reply_lang=obj.get("reply_lang"),
    )


_COMPOUND_EN = re.compile(r"\s+(?:and then|and also|and|then)\s+", re.IGNORECASE)
_COMPOUND_HI = re.compile(r"\s+(?:और फिर|और|फिर)\s+")


def split_compound(utter: str) -> list[str]:
    """'go back to cells and explain it simpler' -> two clauses, if BOTH classify.
    Otherwise the utterance stays whole (an 'and' inside a question is not a split)."""
    for rx in (_COMPOUND_EN, _COMPOUND_HI):
        parts = rx.split(utter, maxsplit=1)
        if len(parts) == 2 and all(_words(p) >= 2 for p in parts):
            a, b = parts[0].strip(), parts[1].strip()
            ca, cb = classify_rules(a), classify_rules(b)
            if ca and cb and (ca.intent, ca.command, ca.session_cmd) != (cb.intent, cb.command, cb.session_cmd):
                return [a, b]
            if ca and cb and ca.intent in ("navigate", "command", "explain") and cb.intent in ("navigate", "command", "explain"):
                return [a, b]
    return [utter]


LLM_SYSTEM = (
    "Classify a student's spoken utterance to a voice tutor. Reply with JSON only: "
    '{"intent": one of question|explain|navigate|command|session|backchannel|unknown, '
    '"command": repeat|slower|faster|switch_lesson_lang|null, "command_arg": "hi"|"en"|null, '
    '"session_cmd": pause|continue|restart|quit|null, '
    '"nav_target": {"kind": prev|next|index|topic, "value": ...}|null, "reply_lang": "hi"|"en"|null}. '
    "explain = about the sentence just heard (define, simplify, translate the reply). "
    "question = anything needing the notes or the web."
)


def classify(utter: str, *, paused: bool = False, llm=None) -> Classification:
    c = classify_rules(utter, paused=paused)
    if c:
        return c
    if llm is not None:
        c = parse_llm_json(llm.complete(LLM_SYSTEM, utter))
        if c:
            return c
    return heuristic(utter)


_FOLLOW_UP_EN = re.compile(
    r"^(?:and|but|so|then|why|how come|what about|how about|is that|does that|was that)\b",
    re.IGNORECASE)
_FOLLOW_UP_HI = ("और", "लेकिन", "तो", "क्यों", "फिर")


def is_follow_up(utter: str) -> bool:
    """'and why?', 'but how?', 'what about the roots?' depend on the previous
    question and should be retrieved together with it. A short but
    self-contained question ('What is chlorophyll?') must not be."""
    text = _norm(utter)
    if _words(text) > 5:
        return False
    return bool(_FOLLOW_UP_EN.match(text)) or any(text.startswith(w) for w in _FOLLOW_UP_HI)
