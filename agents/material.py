"""
Where lesson text comes from and how it is cut up.

Sources
  * parse_pdf       -- the learner's own file (pymupdf; headings by font size,
                       running headers/footers and page numbers removed)
  * fetch_wikipedia -- topic mode: real text from the target-language edition;
                       disambiguation pages raise DisambiguationError

Cutting
  * split_sentences  -- Latin punctuation and the Devanagari danda
  * make_chunks      -- retrieval side: 2-4 sentences with overlap, ORIGINAL text
  * make_beats       -- spoken side: 1-2 sentences, cleaned for the ear
  * clean_for_speech -- what "cleaned for the ear" means (citations, symbols,
                       ranges, parentheticals, chemical formulas)
"""
from __future__ import annotations

import logging
import re
from collections import Counter

logger = logging.getLogger("v-tutor.material")

# --------------------------------------------------------------------------
# Language + sentences
# --------------------------------------------------------------------------

_DEVANAGARI = re.compile("[ऀ-ॿ]")
_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)
_DANDA = "।"


def detect_lang(text: str) -> str:
    """'hi' if a meaningful share of letters are Devanagari, else 'en'."""
    letters = _LETTERS.findall(text)
    if not letters:
        return "en"
    dev = len(_DEVANAGARI.findall(text))
    return "hi" if dev / len(letters) > 0.3 else "en"


_SENT_SPLIT = re.compile(rf"(?<=[.!?{_DANDA}])\s+|\n{{2,}}")
_ABBREV = re.compile(r"\b(e\.g|i\.e|etc|vs|Dr|Mr|Mrs|Ms|St|No|approx|c|Jr|Sr|Prof|Fig|Ltd|Mt|Rev)\.$",
                     re.IGNORECASE)
# A middle initial is not the end of a sentence. "William G. Morgan created
# volleyball" was split after "G.", and the tutor read out "William G. was a
# person associated with the sport. Morgan created the sport..." -- case
# sensitive on purpose, so a sentence ending in a lone lowercase letter is
# still a sentence.
_INITIAL = re.compile(r"\b[A-Z]\.$")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text.strip())     # wrapped PDF lines -> one line
    text = re.sub(r"[ \t]+", " ", text)
    if not text:
        return []
    parts: list[str] = []
    buf = ""
    for piece in _SENT_SPLIT.split(text):
        piece = piece.strip()
        if not piece:
            continue
        buf = f"{buf} {piece}".strip() if buf else piece
        if _ABBREV.search(buf) or _INITIAL.search(buf):
            continue                     # "e.g." / "William G." is not a sentence end
        parts.append(buf)
        buf = ""
    if buf:
        parts.append(buf)
    return parts


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


# --------------------------------------------------------------------------
# Writing for the ear
# --------------------------------------------------------------------------

_CITATION = re.compile(r"\[\s*(?:\d+|[a-z]|note\s*\d+|citation needed|clarification needed)\s*\]", re.I)
_PAREN = re.compile(r"\s*\(([^()]*)\)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*[–—-]\s*(\d+(?:\.\d+)?)")
_DEG_C = re.compile(r"°\s*C\b")
_DEG_F = re.compile(r"°\s*F\b")
_ARROW = re.compile(r"\s*(?:→|⟶|->|⇌|⇒)\s*")
_ABBR_MAP = [
    (re.compile(r"\be\.g\.,?", re.I), "for example,"),
    (re.compile(r"\bi\.e\.,?", re.I), "that is,"),
    (re.compile(r"\betc\.", re.I), "and so on."),
    (re.compile(r"\bvs\.", re.I), "versus"),
    (re.compile(r"\bapprox\.", re.I), "about"),
    (re.compile(r"\bc\.\s*(?=\d{3,4}\b)"), "around "),
]
# Tokens like CO2, H2O, C6H12O6: element symbols with at least one digit.
_FORMULA = re.compile(r"\b\d*(?=[A-Za-z]*\d)(?:[A-Z][a-z]?\d*){1,}\b")   # 6CO2, H2O, C6H12O6
_FORMULA_NAMES = {
    "CO2": "carbon dioxide", "H2O": "water", "O2": "oxygen", "C6H12O6": "glucose",
    "N2": "nitrogen", "H2": "hydrogen", "CO": "carbon monoxide", "CH4": "methane",
    "NH3": "ammonia", "NaCl": "salt", "O3": "ozone", "CaCO3": "calcium carbonate",
}
_SYMBOL_WORDS = {
    "en": {"percent": "percent", "to": "to", "degc": "degrees Celsius", "degf": "degrees Fahrenheit",
           "deg": "degrees", "and": "and", "gives": "gives"},
    "hi": {"percent": "प्रतिशत", "to": "से", "degc": "डिग्री सेल्सियस", "degf": "डिग्री फ़ारेनहाइट",
           "deg": "डिग्री", "and": "और", "gives": "देता है"},
}


def _spell_formula(tok: str) -> str:
    if tok in _FORMULA_NAMES:
        return _FORMULA_NAMES[tok]
    return " ".join(re.findall(r"[A-Z][a-z]?|\d+", tok))          # "C O 2"


def is_equation(sentence: str) -> bool:
    """A chemical equation: two or more formula tokens joined by + or an arrow."""
    toks = _FORMULA.findall(sentence)
    return len(toks) >= 2 and bool(_ARROW.search(sentence) or re.search(r"\s\+\s", sentence))


def clean_for_speech(text: str, lang: str = "en") -> str:
    """Rewrite one sentence so a TTS voice reads it naturally. Returns "" if
    the sentence is not worth speaking (a bare chemical equation)."""
    w = _SYMBOL_WORDS.get(lang, _SYMBOL_WORDS["en"])
    s = _CITATION.sub("", text)
    if is_equation(s):
        return ""
    # parentheticals: keep short ones as an aside, drop long ones and IPA guides
    def _paren(m: re.Match) -> str:
        inner = m.group(1).strip()
        if not inner or "/" in inner or inner.lower().startswith("pronounced") or word_count(inner) > 3:
            return ""
        return f", {inner},"
    s = _PAREN.sub(_paren, s)
    for rx, rep in _ABBR_MAP:
        s = rx.sub(rep, s)
    s = _RANGE.sub(rf"\1 {w['to']} \2", s)
    s = _PERCENT.sub(rf"\1 {w['percent']}", s)
    s = _DEG_C.sub(f" {w['degc']}", s)
    s = _DEG_F.sub(f" {w['degf']}", s)
    s = s.replace("°", f" {w['deg']}")
    s = s.replace("&", f" {w['and']} ")
    s = _ARROW.sub(f" {w['gives']} ", s)
    s = _FORMULA.sub(lambda m: _spell_formula(m.group(0)), s)
    s = re.sub(r"==+", " ", s)
    s = re.sub(r"\s+([,.;:!?।])", r"\1", s)
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,;")
    if not s or word_count(s) < 2:
        return ""
    if s[-1] not in ".!?।":
        s += "।" if lang == "hi" else "."
    return s


# --------------------------------------------------------------------------
# Sections from plain text (wiki "== H ==" or markdown "# H")
# --------------------------------------------------------------------------

_HEADING = re.compile(r"^\s*(?:(=+)\s*(.+?)\s*=+|(#{1,6})\s+(.+?))\s*$")

# Sections that are noise for a spoken lesson, in the two supported languages.
_SKIP_TITLES = {
    "references", "external links", "see also", "notes", "further reading",
    "bibliography", "sources", "gallery", "citations", "footnotes",
    "सन्दर्भ", "संदर्भ", "बाहरी कड़ियाँ", "इन्हें भी देखें", "टिप्पणी", "स्रोत",
}


def sections_from_plaintext(raw: str, title: str = "Introduction",
                            source_url: str = "") -> list[dict]:
    sections: list[dict] = []
    cur_title = title
    cur_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(cur_lines).strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        if body and cur_title.strip().lower() not in _SKIP_TITLES:
            sections.append({
                "id": f"s{len(sections)}", "title": cur_title.strip(),
                "text": body, "source_url": source_url,
            })

    for line in raw.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            cur_title = (m.group(2) or m.group(4) or "").strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    flush()
    return sections


# --------------------------------------------------------------------------
# Topic mode: Wikipedia
# --------------------------------------------------------------------------

# Wikimedia returns 403 unless the User-Agent carries a contact URL. It ALSO
# blocks httpx by client fingerprint (same UA gets 200 via requests/curl and 403
# via httpx, verified 2026-09-07), so this module uses `requests` for Wikipedia.
_UA = "v-tutor/0.1 (https://github.com/shuvradeeeep/v-tutor; hackathon voice tutor)"


class DisambiguationError(Exception):
    """The topic names several things. `options` are the first few, for the
    tutor to read back: 'Did you mean the planet, the element, or the god?'"""

    def __init__(self, title: str, options: list[str]) -> None:
        super().__init__(f"{title} is ambiguous")
        self.title = title
        self.options = options


def disambiguation_options(extract: str, limit: int = 3) -> list[str]:
    opts: list[str] = []
    for line in extract.splitlines():
        line = line.strip()
        if not line or _HEADING.match(line) or len(line) > 160:
            continue
        if line.lower().endswith("may refer to:") or line.lower().endswith("may also refer to:"):
            continue
        opts.append(re.split(r"[,;:]| – | - ", line, maxsplit=1)[0].strip())   # keep "(planet)" qualifiers
        if len(opts) >= limit:
            break
    return opts


def fetch_wikipedia(topic: str, lang: str, timeout: float = 10.0) -> tuple[list[dict], str, str] | None:
    """Return (sections, detected_lang, url) from the `lang` edition, or None.
    Raises DisambiguationError for pages that list several meanings.

    MediaWiki API, one round trip: `generator=search` resolves the title and
    `prop=extracts&explaintext` on the same call gives clean section text with
    "== Heading ==" markers. No API key.
    """
    import requests

    api = f"https://{lang}.wikipedia.org/w/api.php"
    headers = {"User-Agent": _UA, "Accept": "application/json"}
    try:
        r = requests.get(api, params={
            "action": "query", "generator": "search", "gsrsearch": topic, "gsrlimit": 1,
            "prop": "extracts|pageprops", "explaintext": 1, "exsectionformat": "wiki",
            "redirects": 1, "format": "json",
        }, headers=headers, timeout=timeout)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {}) if pages else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("wikipedia fetch failed for %r/%s: %s", topic, lang, exc)
        return None

    title = page.get("title") or ""
    extract = (page.get("extract") or "").strip()
    is_disambig = "disambiguation" in (page.get("pageprops") or {}) \
        or title.lower().endswith("(disambiguation)")
    if not title or not extract:
        return None
    if is_disambig:
        raise DisambiguationError(title, disambiguation_options(extract))
    url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
    sections = sections_from_plaintext(extract, title=title, source_url=url)
    return (sections, detect_lang(extract), url) if sections else None


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

_PAGE_NUMBER = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*(?:/\s*\d{1,4})?\s*$", re.I)


def parse_pdf(paths: list[str]) -> tuple[list[dict], str, str] | None:
    """Return (sections, lang, title) or None if no text layer was found.

    Headings are detected by font size relative to the page's body text, so a
    textbook chapter keeps its structure. Lines that repeat on most pages
    (running headers/footers) and bare page numbers are dropped. Scanned PDFs
    return None and the tutor asks for a topic instead -- OCR is out of scope.
    """
    import fitz  # pymupdf

    pages: list[list[tuple[str, bool]]] = []          # per page: [(text, is_heading)]
    total_chars = 0
    for path in paths:
        try:
            doc = fitz.open(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cannot open %s: %s", path, exc)
            continue
        for page in doc:
            d = page.get_text("dict")
            sizes = [s["size"] for b in d.get("blocks", []) for l in b.get("lines", [])
                     for s in l.get("spans", []) if s.get("text", "").strip()]
            if not sizes:
                pages.append([])
                continue
            body = sorted(sizes)[len(sizes) // 2]
            out: list[tuple[str, bool]] = []
            for b in d.get("blocks", []):
                for l in b.get("lines", []):
                    spans = [s for s in l.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    text = " ".join(s["text"].strip() for s in spans)
                    total_chars += len(text)
                    size = max(s["size"] for s in spans)
                    out.append((text, size >= body * 1.25 and word_count(text) <= 12))
            pages.append(out)
    if total_chars < 200:
        return None

    # running headers / footers: same normalised line on >= half the pages
    norm = lambda t: re.sub(r"\d+", "#", t.lower()).strip()
    freq = Counter(norm(t) for pg in pages for t, _ in pg)
    repeated = {k for k, n in freq.items() if len(pages) >= 2 and n >= max(2, len(pages) // 2)}

    lines_out: list[str] = []
    for pg in pages:
        for text, is_heading in pg:
            if _PAGE_NUMBER.match(text) or norm(text) in repeated:
                continue
            lines_out.append(f"== {text} ==" if is_heading else text)
        lines_out.append("")
    raw = "\n".join(lines_out)
    title = re.split(r"[\\/]", paths[0])[-1].rsplit(".", 1)[0]
    sections = sections_from_plaintext(raw, title=title)
    return (sections, detect_lang(raw), title) if sections else None


# --------------------------------------------------------------------------
# Chunks (retrieval) and beats (speech)
# --------------------------------------------------------------------------

def make_chunks(sections: list[dict], size: int = 3, overlap: int = 1) -> list[dict]:
    chunks: list[dict] = []
    step = max(1, size - overlap)
    for sec in sections:
        sents = split_sentences(sec["text"])
        if not sents:
            continue
        for start in range(0, len(sents), step):
            window = sents[start:start + size]
            if not window:
                break
            chunks.append({
                "id": f"{sec['id']}c{len(chunks)}", "section_id": sec["id"],
                "section_title": sec["title"], "text": " ".join(window),
                "source": sec.get("source_url") or "notes",
            })
            if start + size >= len(sents):
                break
    return chunks


def make_beats(sections: list[dict], sentences_per_beat: int = 2, lang: str = "en") -> list[dict]:
    """Spoken units. Every sentence passes clean_for_speech; empties are dropped."""
    beats: list[dict] = []
    for sec in sections:
        sents = [c for c in (clean_for_speech(s, lang) for s in split_sentences(sec["text"])) if c]
        for i in range(0, len(sents), sentences_per_beat):
            group = sents[i:i + sentences_per_beat]
            beats.append({
                "id": f"b{len(beats)}", "section_id": sec["id"],
                "section_title": sec["title"], "text": " ".join(group),
                "sentences": group,
            })
    return beats
