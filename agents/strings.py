"""
Every fixed phrase the tutor speaks, in both study languages.

Written for the ear: short sentences, no lists, no symbols. These are also the
phrases pre-rendered as audio for the gap filler and bridges, so wording here
must stay in sync with assets/fillers/ once those exist.
"""
from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "en": {
        # ---- onboarding ----
        # Spoken in English only. The screen shows language buttons alongside;
        # a button press arrives as if the student had said the language name.
        "ask_language": "Hi! Which language shall we study in? You can say it, or tap it on the screen.",
        "ask_source": "Great. What shall we study today, and for which class?",
        "ask_source_again": "Sorry, I didn't get the topic. What would you like to study?",
        "ask_grade": "Got it. And which class is this for?",
        "cant_read_pdf": "I can't read that file, it has no text I can use. Could you tell me the topic instead?",
        "not_found_topic": "I couldn't find good material on that. Could you try a different topic?",
        "fallback_english": "I couldn't find this in Hindi, so I'm using English material and translating it.",
        "lets_begin": "Let's begin. Today's lesson is {title}.",
        "prepare_wait": "Give me a moment while I get the lesson ready.",
        "disambiguation": "I found a few different things called {title}. Did you mean {options}?",
        # ---- fillers (pre-rendered later; must stay in sync with assets/fillers) ----
        "filler_check": "Let me check that.",
        "filler_moment": "One moment.",
        # ---- lesson flow ----
        "bridge": "So, back to where we were.",
        "bridge_short": "Right, carrying on.",
        "lesson_done": "That's the end of the lesson. Well done! Say start over if you want to hear it again.",
        "paused": "Okay, pausing. Say continue when you're ready.",
        "restart": "Okay, starting from the beginning.",
        "goodbye": "Alright, that's it for today. Good work. Bye!",
        "clarify": "Sorry, I didn't catch that. Could you say it again?",
        "go_ahead": "Sure, go ahead.",
        "lang_switched": "Okay, I'll teach in English from here.",
        "lesson_lang_unsupported": "I can explain things in that language, but I can only teach the whole lesson in {langs}.",
        "nav_not_found": "I couldn't find a part about that. Let's carry on.",
        "going_to": "Going to the part about {title}.",
        "switch_topic": "Okay, let's switch to {title}. Give me a moment while I get it ready.",
        # ---- answers ----
        "from_notes": "From the notes:",
        "from_web": "I looked that up online.",
        "not_found_answer": "I couldn't find that in the notes or online. Let's keep going.",
        "go_back_offer": "That's in the part about {title}. Say go there if you want to hear it.",
        "explain_fallback": "Let me say that again, slowly.",
        "explain_fallback_lang": "I can't translate right now, but here it is again.",
        "slower_ack": "Okay, slower.",
        "faster_ack": "Okay, a bit faster.",
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
        "disambiguation": "{title} नाम की कई चीज़ें मिलीं। क्या आपका मतलब {options} था?",
        "filler_check": "मैं देखती हूँ।",
        "filler_moment": "एक पल।",
        "bridge": "तो, चलिए वहीं से आगे बढ़ते हैं।",
        "bridge_short": "ठीक है, आगे चलते हैं।",
        "lesson_done": "पाठ यहाँ समाप्त होता है। बहुत अच्छा! दोबारा सुनने के लिए कहिए, फिर से शुरू करो।",
        "paused": "ठीक है, रुकते हैं। तैयार होने पर कहिए, आगे बढ़ो।",
        "restart": "ठीक है, शुरू से शुरू करते हैं।",
        "goodbye": "ठीक है, आज के लिए बस इतना ही। बहुत अच्छा काम किया। नमस्ते!",
        "clarify": "माफ़ कीजिए, समझ नहीं आया। क्या आप फिर से कह सकते हैं?",
        "lang_switched": "ठीक है, अब से मैं हिंदी में पढ़ाऊँगी।",
        "lesson_lang_unsupported": "मैं उस भाषा में समझा सकती हूँ, पर पूरा पाठ केवल {langs} में पढ़ा सकती हूँ।",
        "nav_not_found": "उस विषय का हिस्सा मुझे नहीं मिला। चलिए आगे बढ़ते हैं।",
        "switch_topic": "ठीक है, चलिए {title} पर चलते हैं। एक पल दीजिए, मैं तैयार करती हूँ।",
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


def t(lang: str, key: str, **fmt: object) -> str:
    table = STRINGS.get(lang) or STRINGS["en"]
    s = table.get(key) or STRINGS["en"][key]
    return s.format(**fmt) if fmt else s
