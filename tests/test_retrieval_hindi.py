"""Hindi retrieval over Hindi notes. BM25 carries most of this: Devanagari
tokens must survive tokenisation intact (vowel signs are combining marks)."""
from __future__ import annotations

import pytest

import config
from agents.material import make_chunks, sections_from_plaintext
from agents.retrieval import Doc, HashEmbedder, HybridRetriever, tokenize

HINDI_TEXT = """\
== हृदय ==
हृदय एक मांसपेशीय अंग है जो मुट्ठी के आकार का होता है। यह पूरे शरीर में रक्त पंप करता है। एक वयस्क का हृदय विश्राम में प्रति मिनट लगभग 72 बार धड़कता है।

== कक्ष ==
हृदय में चार कक्ष होते हैं। ऊपर के दो कक्ष आलिंद कहलाते हैं। नीचे के दो कक्ष निलय कहलाते हैं। बायाँ निलय सबसे मज़बूत कक्ष है।

== रक्त प्रवाह ==
रक्त महाधमनी से होकर हृदय से बाहर जाता है, जो शरीर की सबसे बड़ी धमनी है। शिराएँ रक्त को हृदय तक वापस लाती हैं। वाल्व रक्त को पीछे बहने से रोकते हैं।

== इतिहास ==
विलियम हार्वे ने 1628 में रक्त परिसंचरण का वर्णन किया। उनसे पहले कई लोग मानते थे कि यकृत रक्त बनाता है।
"""

IN_NOTES = [
    ("हृदय में कितने कक्ष होते हैं?", "कक्ष"),
    ("ऊपर के कक्ष क्या कहलाते हैं?", "कक्ष"),
    ("हृदय प्रति मिनट कितनी बार धड़कता है?", "हृदय"),
    ("1628 में क्या हुआ?", "इतिहास"),
    ("शिराएँ क्या करती हैं?", "रक्त प्रवाह"),
    ("सबसे बड़ी धमनी कौन सी है?", "रक्त प्रवाह"),
]
OUT_OF_NOTES = ["मंगल ग्रह के कितने चंद्रमा हैं?", "हैमलेट किसने लिखा?"]


def test_devanagari_words_are_single_tokens():
    assert tokenize("हृदय में") == ["हृदय", "में"]


@pytest.fixture(scope="module")
def retriever():
    secs = sections_from_plaintext(HINDI_TEXT, title="हृदय")
    chunks = make_chunks(secs, config.CHUNK_SENTENCES, config.CHUNK_OVERLAP)
    return HybridRetriever([Doc(c["id"], c["text"], c["section_id"], c["section_title"]) for c in chunks],
                           HashEmbedder())


@pytest.mark.parametrize("q,section", IN_NOTES)
def test_hindi_in_notes(retriever, q, section):
    hit = retriever.best(q)
    assert hit is not None and hit.doc.section_title == section, (q, hit and hit.doc.section_title, hit and round(hit.score, 3))
    assert hit.score >= config.RETRIEVAL_TAU, (q, round(hit.score, 3))


@pytest.mark.parametrize("q", OUT_OF_NOTES)
def test_hindi_out_of_notes(retriever, q):
    hit = retriever.best(q)
    assert hit is None or hit.score < config.RETRIEVAL_TAU, (q, round(hit.score, 3))
