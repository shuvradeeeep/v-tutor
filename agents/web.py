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


def duckduckgo_search(query: str, max_results: int = 3) -> list[dict]:
    try:
        from ddgs import DDGS  # renamed from duckduckgo_search in 2025
    except ImportError:  # pragma: no cover
        from duckduckgo_search import DDGS  # type: ignore
    try:
        with DDGS() as d:
            rows = d.text(query, max_results=max_results) or []
    except Exception as exc:  # noqa: BLE001 -- never let a search failure crash the voice loop
        logger.warning("duckduckgo failed for %r: %s", query, exc)
        return []
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
