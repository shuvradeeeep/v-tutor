"""
Hybrid retrieval over the whole document: BM25 keyword side + dense side.

Why hybrid: embeddings blur names, numbers and dates ("what year was the
treaty signed" lands near any sentence *about* the treaty). BM25 nails exact
tokens but misses paraphrase. Fusing both gives specific-fact questions and
conceptual questions a fair shot -- and the PS explicitly probes names/numbers.

Why the BM25 normalisation is the way it is: dividing by the max score would
make the best chunk score 1.0 for ANY query, including out-of-syllabus ones.
We divide by the maximum score the query *could* achieve given the corpus IDF,
so a question whose terms don't appear in the notes scores low, and
route_retrieval correctly sends it to web search.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# \w alone splits Devanagari words at vowel signs (they are combining marks, not
# letters), so the whole Devanagari block is included explicitly.
_TOKEN_RE = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)

# Function words carry no evidence that a question is answered by the notes.
# Without this, "how MANY moons does Mars have" scores high on any chunk that
# happens to contain "many". Small lists on purpose; content words stay.
_STOP_EN = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
    "have", "has", "had", "of", "in", "on", "at", "to", "for", "from", "by", "with", "about",
    "and", "or", "but", "if", "so", "that", "this", "these", "those", "it", "its", "as",
    "what", "which", "who", "whom", "whose", "how", "why", "when", "where", "many", "much",
    "can", "could", "would", "will", "should", "me", "my", "you", "your", "we", "our", "they",
    "them", "their", "he", "she", "his", "her", "i", "us", "there", "here", "than", "then",
    "tell", "please", "called", "mean", "means",
}
_STOP_HI = {
    "है", "हैं", "था", "थे", "थी", "हो", "होता", "होती", "होते", "का", "की", "के", "को", "से",
    "में", "पर", "और", "या", "कि", "यह", "वह", "ये", "वो", "एक", "क्या", "क्यों", "कैसे", "कब",
    "कहाँ", "कहां", "कौन", "किस", "कितने", "कितनी", "कितना", "मुझे", "हम", "आप", "तुम", "बताओ",
    "बताइए", "लिए", "भी", "ही", "तो", "ने", "जो",
}
_STOP = _STOP_EN | _STOP_HI

try:  # light English stemming so beat/beats, carry/carries match. Optional.
    from py_rust_stemmers import SnowballStemmer as _Snowball  # installed with fastembed
    _stem = _Snowball("english").stem_word
except Exception:  # noqa: BLE001
    _stem = None


def tokenize(text: str) -> list[str]:
    """All tokens (for BM25 document statistics)."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def content_tokens(text: str) -> list[str]:
    """Tokens minus stopwords, stemmed. Used for the query side and coverage."""
    out = []
    for t in tokenize(text):
        if t in _STOP:
            continue
        out.append(_stem(t) if _stem and t.isascii() else t)
    return out


# --------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------

class HashEmbedder:
    """Offline, deterministic, language-agnostic character n-gram hashing.

    Not semantic -- it is a lexical similarity in disguise -- but it needs no
    download, works for Devanagari as well as Latin script, and is stable
    across runs, which is what tests and the harness need. Production uses
    FastEmbedEmbedder; the dense weight is lowered when this one is active.
    """
    name = "hash"

    def __init__(self, dim: int = 512, ngram: tuple[int, int] = (3, 5)) -> None:
        self.dim = dim
        self.ngram = ngram

    def _grams(self, text: str):
        t = f" {re.sub(r'\s+', ' ', text.lower().strip())} "
        lo, hi = self.ngram
        for n in range(lo, hi + 1):
            for i in range(max(0, len(t) - n + 1)):
                yield t[i:i + n]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for g in self._grams(text):
                h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class FastEmbedEmbedder:
    """Multilingual dense embeddings via fastembed (ONNX, no torch)."""
    name = "fastembed"

    def __init__(self, model_name: str | None = None) -> None:
        from config import FASTEMBED_MODEL
        from fastembed import TextEmbedding  # lazy: heavy import + model download
        self._model = TextEmbedding(model_name or FASTEMBED_MODEL)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def make_embedder(kind: str | None = None):
    from config import EMBEDDER
    kind = (kind or EMBEDDER or "hash").lower()
    if kind == "fastembed":
        return FastEmbedEmbedder()
    return HashEmbedder()


# --------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------

@dataclass
class Doc:
    id: str
    text: str
    section_id: str
    section_title: str
    meta: dict = field(default_factory=dict)


@dataclass
class Hit:
    doc: Doc
    score: float          # fused, in [0, 1]
    bm25: float           # normalised
    dense: float          # cosine clipped to [0, 1]

    def as_dict(self) -> dict:
        return {
            "text": self.doc.text, "section_id": self.doc.section_id,
            "section_title": self.doc.section_title, "score": round(self.score, 3),
            "source": self.doc.meta.get("source", "notes"),
        }


class HybridRetriever:
    def __init__(self, docs: list[Doc], embedder, dense_weight: float | None = None) -> None:
        from rank_bm25 import BM25Okapi

        self.docs = docs
        self.embedder = embedder
        if dense_weight is None:
            # Hash embeddings are only weakly discriminative; lean on BM25.
            dense_weight = 0.3 if getattr(embedder, "name", "") == "hash" else 0.5
        self.dense_weight = dense_weight

        self._tokens = [content_tokens(d.text) for d in docs]
        self._token_sets = [set(t) for t in self._tokens]
        self._bm25 = BM25Okapi(self._tokens) if docs else None
        self._vecs = embedder.embed([d.text for d in docs]) if docs else np.zeros((0, 1), np.float32)

    # -- BM25 normalised so that "every query word present once" scores ~1.0 --
    def _bm25_norm(self, q_tokens: list[str]) -> np.ndarray:
        if not self._bm25 or not q_tokens:
            return np.zeros(len(self.docs), dtype=np.float32)
        raw = np.asarray(self._bm25.get_scores(q_tokens), dtype=np.float32)
        # With tf=1 and average length, a term contributes ~idf. Summing idf over
        # the in-vocabulary query terms is therefore the realistic ceiling.
        ceiling = sum(self._bm25.idf.get(t, 0.0) for t in set(q_tokens) if t in self._bm25.idf)
        if ceiling <= 0:
            return np.zeros_like(raw)
        return np.clip(raw / ceiling, 0.0, 1.0)

    def _coverage(self, q_tokens: list[str]) -> np.ndarray:
        """Fraction of the query's content words present in each doc. A query
        whose words are simply not in the notes cannot be 'grounded' no matter
        how the other signals fall."""
        q = set(q_tokens)
        if not q:
            return np.zeros(len(self.docs), dtype=np.float32)
        return np.asarray([len(q & ts) / len(q) for ts in self._token_sets], dtype=np.float32)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if not self.docs:
            return []
        q_tokens = content_tokens(query)
        bm = self._bm25_norm(q_tokens)
        qv = self.embedder.embed([query])[0]
        dense = np.clip(self._vecs @ qv, 0.0, 1.0)
        cov = self._coverage(q_tokens)
        fused = ((1.0 - self.dense_weight) * bm + self.dense_weight * dense) * (0.4 + 0.6 * cov)
        order = np.argsort(-fused)[:k]
        return [Hit(self.docs[i], float(fused[i]), float(bm[i]), float(dense[i])) for i in order]

    def best(self, query: str) -> Hit | None:
        hits = self.search(query, k=1)
        return hits[0] if hits else None


# --------------------------------------------------------------------------
# Extractive fallback when no LLM is available
# --------------------------------------------------------------------------

def best_sentences(text: str, query: str, n: int = 2, sentence_splitter=None) -> str:
    """Pick the n sentences of `text` with the most query-token overlap, in
    original order. Deterministic, language-agnostic, good enough to keep the
    tutor answering when the LLM is down."""
    from agents.material import split_sentences
    splitter = sentence_splitter or split_sentences
    sents = splitter(text)
    if not sents:
        return text.strip()
    q = set(content_tokens(query))
    scored = [(len(q & set(content_tokens(s))), i, s) for i, s in enumerate(sents)]
    top = sorted(scored, key=lambda t: (-t[0], t[1]))[:n]
    top.sort(key=lambda t: t[1])
    return " ".join(s for _, _, s in top).strip()
