"""
Every fixed phrase the tutor speaks, in both study languages.

Written for the ear, the way Rime's prompting guide asks: short sentences,
contractions, punctuation as prosody (a comma is a breath, an ellipsis is a
hesitation, a question mark lifts the pitch), exclamation marks only where a
person would actually use one. Coda has no emotion tags or SSML, so the
warmth has to be in the words.

A value may be a LIST of variants. The tutor rotates through them across turns
(agents/nodes.py::_T), so "let me check that" is not the same sentence twenty
times in a lesson. Variant 0 is the canonical wording; tests and the echo
guard fixtures use it. Fillers are also disk-cached by Rime text, so each
variant costs one synthesis, ever.
"""
from __future__ import annotations

Phrase = "str | list[str]"

STRINGS: dict[str, dict[str, "str | list[str]"]] = {
    "en": {
        # ---- onboarding ----
        # Spoken in English only. The screen shows language buttons alongside;
        # a button press arrives as if the student had said the language name.
        "ask_language": [
            "Hi! Which language shall we study in? You can say it, or tap it on the screen.",
            "Hi there. Good to have you. Which language should we learn in today? Just say it, or tap it on the screen.",
            "Hello! Let's get started. English or Hindi? You can say it, or tap it on the screen.",
        ],
        "ask_source": [
            "Great. What shall we study today, and for which class?",
            "Lovely. So, what are we learning today, and which class is it for?",
            "Okay. Tell me the topic, and which class you're in.",
        ],
        "ask_source_again": "Sorry, I didn't get the topic. What would you like to study?",
        "ask_grade": "Got it. And which class is this for?",
        "cant_read_pdf": "I can't read that file, it has no text I can use. Could you tell me the topic instead?",
        "not_found_topic": "I couldn't find good material on that. Could you try a different topic?",
        "fallback_english": "I couldn't find this in Hindi, so I'm using English material and translating it.",
        "lets_begin": [
            "Let's begin. Today's lesson is {title}.",
            "Right, here we go. Today we're looking at {title}.",
            "Okay, all set. Today's lesson is {title}. Stop me whenever you have a question.",
        ],
        "prepare_wait": "Give me a moment while I get the lesson ready.",
        "prepare_wait_topic": [
            "Okay, {topic}{grade}. Give me a moment while I get the lesson ready.",
            "{topic}{grade}, got it. Give me a moment while I get the lesson ready.",
        ],
        "section_intro": [
            "Now, let's look at {title}.",
            "Next up, {title}.",
            "Okay. On to {title}.",
            "Now, {title}.",
        ],
        "disambiguation": "I found a few different things called {title}. Did you mean {options}?",
        # ---- fillers: spoken after 0.7 s of dead air while a tool or model runs ----
        "filler_check": [
            "Let me check that.",
            "Hmm, good question. Let me check.",
            "One sec, let me look that up.",
            "Let me see...",
        ],
        "filler_moment": [
            "One moment.",
            "Just a sec.",
            "Hang on...",
            "Right, one moment.",
        ],
        # ---- lesson flow ----
        "bridge": [
            "So, back to where we were.",
            "Okay. Where were we... right.",
            "Anyway, back to the lesson.",
            "Good. So, carrying on.",
        ],
        "bridge_short": ["Right, carrying on.", "Okay, on we go.", "So."],
        "lesson_done": [
            "That's the end of the lesson. Well done! Say start over if you want to hear it again.",
            "And that's the whole lesson. Nicely done. If you'd like to hear it again, just say start over.",
        ],
        "paused": ["Okay, pausing. Say continue when you're ready.", "Sure, take your time. Say continue when you're back."],
        "restart": "Okay, starting from the beginning.",
        "goodbye": [
            "Alright, that's it for today. Good work. Bye!",
            "Okay, we'll stop there. You did well today. See you next time!",
            "Sure. That's a good place to stop. Bye for now!",
        ],
        "clarify": [
            "Sorry, I didn't catch that. Could you say it again?",
            "Hmm, I missed that. One more time?",
            "Sorry, say that again for me?",
        ],
        "go_ahead": ["Sure, go ahead.", "Of course. What is it?", "Yeah, go on."],
        "lang_switched": "Okay, I'll teach in English from here.",
        "lesson_lang_unsupported": "I can explain things in that language, but I can only teach the whole lesson in {langs}.",
        "nav_not_found": "I couldn't find a part about that. Let's carry on.",
        "going_to": "Going to the part about {title}.",
        "switch_topic": "Okay, let's switch to {title}. Give me a moment while I get it ready.",
        "nav_switch_offer": "That isn't part of this lesson. Say switch to {title} if you'd like a new lesson on it. For now, let's carry on.",
        "ask_switch_topic": "Sure, what topic would you like to switch to, and which class is it for?",
        # ---- questions about the session itself ----
        "status_length": "About {minutes} more minutes if we don't stop. We're on part {done} of {total}, with {sections} sections still to come.",
        "status_topic": "We're studying {title}{grade}.",
        "status_progress": "We're on part {done} of {total}.",
        "who_am_i": "I'm your voice tutor. I teach from Wikipedia or your own notes, and you can interrupt me any time to ask something.",
        # ---- answers ----
        "from_notes": "From the notes:",
        "from_web": "I looked that up online.",
        "not_found_answer": "I couldn't find that in the notes or online. Let's keep going.",
        "go_back_offer": "That's in the part about {title}. Say go there if you want to hear it.",
        "explain_fallback": "Let me say that again, slowly.",
        "explain_fallback_lang": "I can't translate right now, but here it is again.",
        "slower_ack": ["Okay, slower.", "Sure. A little slower."],
        "faster_ack": ["Okay, a bit faster.", "Sure, picking up the pace."],
    },
    "hi": {
        "ask_language": "Hi! Which language shall we study in? You can say it, or tap it on the screen.",
        "ask_source": "बहुत अच्छा। आज हम क्या पढ़ें, और किस कक्षा के लिए?",
        "ask_source_again": "माफ़ कीजिए, विषय समझ नहीं आया। आप क्या पढ़ना चाहेंगे?",
        "ask_grade": "ठीक है। और यह किस कक्षा के लिए है?",
        "cant_read_pdf": "मैं यह फ़ाइल नहीं पढ़ पा रही हूँ। क्या आप विषय बता सकते हैं?",
        "not_found_topic": "इस विषय पर मुझे अच्छी सामग्री नहीं मिली। कोई और विषय बताएँगे?",
        "fallback_english": "यह हिंदी में नहीं मिला, इसलिए मैं अंग्रेज़ी सामग्री का अनुवाद करके पढ़ा रही हूँ।",
        "lets_begin": "चलिए शुरू करते हैं। आज का पाठ है {title}।",
        "prepare_wait": "एक पल दीजिए, मैं पाठ तैयार कर रही हूँ।",
        "prepare_wait_topic": "ठीक है, {topic}{grade}। एक पल दीजिए, मैं पाठ तैयार कर रही हूँ।",
        "section_intro": "अब {title} के बारे में।",
        "disambiguation": "{title} नाम की कई चीज़ें मिलीं। क्या आपका मतलब {options} था?",
        "filler_check": ["मैं देखती हूँ।", "अच्छा सवाल है। एक पल, देखती हूँ।", "हम्म, ज़रा देखने दीजिए..."],
        "filler_moment": ["एक पल।", "बस एक सेकंड।", "ज़रा रुकिए..."],
        "bridge": ["तो, चलिए वहीं से आगे बढ़ते हैं।", "अच्छा, हम कहाँ थे... हाँ।", "चलिए, पाठ पर वापस।"],
        "bridge_short": ["ठीक है, आगे चलते हैं।", "तो, आगे।"],
        "lesson_done": "पाठ यहाँ समाप्त होता है। बहुत अच्छा! दोबारा सुनने के लिए कहिए, फिर से शुरू करो।",
        "paused": "ठीक है, रुकते हैं। तैयार होने पर कहिए, आगे बढ़ो।",
        "restart": "ठीक है, शुरू से शुरू करते हैं।",
        "goodbye": "ठीक है, आज के लिए बस इतना ही। बहुत अच्छा काम किया। नमस्ते!",
        "clarify": "माफ़ कीजिए, समझ नहीं आया। क्या आप फिर से कह सकते हैं?",
        "lang_switched": "ठीक है, अब से मैं हिंदी में पढ़ाऊँगी।",
        "lesson_lang_unsupported": "मैं उस भाषा में समझा सकती हूँ, पर पूरा पाठ केवल {langs} में पढ़ा सकती हूँ।",
        "nav_not_found": "उस विषय का हिस्सा मुझे नहीं मिला। चलिए आगे बढ़ते हैं।",
        "switch_topic": "ठीक है, चलिए {title} पर चलते हैं। एक पल दीजिए, मैं तैयार करती हूँ।",
        "nav_switch_offer": "यह इस पाठ का हिस्सा नहीं है। अगर {title} पर नया पाठ चाहिए तो कहिए, स्विच टू {title}। अभी हम आगे बढ़ते हैं।",
        "ask_switch_topic": "ज़रूर, आप किस विषय पर बदलना चाहते हैं, और यह किस कक्षा के लिए है?",
        "status_length": "हम {total} में से {done} भाग पर हैं, {sections} भाग बाकी हैं। बिना रुके लगभग {minutes} मिनट और।",
        "status_topic": "हम {title}{grade} पढ़ रहे हैं।",
        "status_progress": "हम {total} में से {done} भाग पर हैं।",
        "who_am_i": "मैं आपकी वॉइस ट्यूटर हूँ। मैं विकिपीडिया या आपके नोट्स से पढ़ाती हूँ, और आप मुझे कभी भी रोककर सवाल पूछ सकते हैं।",
        "going_to": "{title} वाले हिस्से पर चलते हैं।",
        "from_notes": "नोट्स के अनुसार:",
        "from_web": "मैंने यह ऑनलाइन देखा।",
        "not_found_answer": "यह मुझे नोट्स में या ऑनलाइन नहीं मिला। चलिए आगे बढ़ते हैं।",
        "go_back_offer": "यह {title} वाले हिस्से में है। सुनना चाहें तो कहिए, वहाँ चलो।",
        "explain_fallback": "मैं इसे फिर से, धीरे से कहती हूँ।",
        "explain_fallback_lang": "अभी मैं अनुवाद नहीं कर पा रही, पर यह फिर से सुनिए।",
        "slower_ack": "ठीक है, धीरे।",
        "faster_ack": "ठीक है, थोड़ा तेज़।",
    },
}


def variants(lang: str, key: str) -> list[str]:
    table = STRINGS.get(lang) or STRINGS["en"]
    s = table.get(key) or STRINGS["en"][key]
    return list(s) if isinstance(s, list) else [s]


def t(lang: str, key: str, variant: int = 0, **fmt: object) -> str:
    """The phrase for `key`. `variant` picks among alternatives (wrapped, so
    any integer is valid); 0 is always the canonical wording."""
    vs = variants(lang, key)
    s = vs[variant % len(vs)]
    return s.format(**fmt) if fmt else s
