"""
Web search for out-of-syllabus questions. Provider chosen by config; the stub
returns nothing so the tutor says "not in the notes or online". DuckDuckGo
needs no API key. Every provider returns [{"title", "snippet", "url"}].
"""
from __future__ import annotations

import logging

logger = logging.getLogger("v-tutor.web")


def stub_search(query: str) -> list[dict]:
    return []


# ddgs fans out to several engines. Measured 2026-09-07 from here: brave ~1.1 s,
# auto ~2.6 s, bing ~6 s. Try the fast one first, then let the library choose.
DDGS_BACKENDS = ("brave", "auto")


def duckduckgo_search(query: str, max_results: int = 3) -> list[dict]:
    try:
        from ddgs import DDGS  # renamed from duckduckgo_search in 2025
    except ImportError:  # pragma: no cover
        from duckduckgo_search import DDGS  # type: ignore
    rows: list[dict] = []
    for backend in DDGS_BACKENDS:
        try:
            with DDGS(timeout=8) as d:
                rows = d.text(query, max_results=max_results, backend=backend) or []
        except Exception as exc:  # noqa: BLE001 -- never let a search failure crash the voice loop
            logger.warning("web search (%s) failed for %r: %s", backend, query, exc)
            rows = []
        if rows:
            break
    out = []
    for r in rows:
        snippet = (r.get("body") or r.get("snippet") or "").strip()
        if not snippet or is_boilerplate(snippet):
            continue
        out.append({"title": r.get("title", ""), "snippet": snippet,
                    "url": r.get("href") or r.get("url") or "web"})
    return out


_BOILERPLATE = ("this page was last edited", "cookies", "sign in", "log in", "subscribe",
                "javascript", "all rights reserved", "privacy policy")


def is_boilerplate(snippet: str) -> bool:
    """Search engines sometimes return a page footer instead of content."""
    low = snippet.lower()
    return any(low.startswith(b) or (b in low and len(low) < 80) for b in _BOILERPLATE)


def make_web_search(provider: str | None = None):
    from config import WEB_SEARCH_PROVIDER
    provider = (provider or WEB_SEARCH_PROVIDER or "stub").lower()
    if provider == "duckduckgo":
        return duckduckgo_search
    return stub_search
