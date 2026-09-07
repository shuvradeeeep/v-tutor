"""
Two LLM roles behind one adapter.

  fast()   -- intent classification. Latency matters; quality bar is low.
  strong() -- answers, explanations, the ingest translate+simplify pass.

Provider and model per role come from the environment (see .env.example), so
the final "which model where" decision is a config change, not a code change.

The default provider is "stub": complete() returns "". Every node treats "" as
"LLM unavailable" and takes a deterministic fallback path. This is what lets
the whole graph, the fencing test, and the text harness run offline -- and it
doubles as the tutor's failure behaviour if a provider is down mid-demo.
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger("v-tutor.llm")


class StubLLM:
    provider = "stub"
    model = "-"

    def complete(self, system: str, user: str) -> str:
        return ""


# Models return typographic punctuation (non-breaking hyphens, curly quotes, em
# dashes) and stray markdown. TTS reads those literally or chokes on them, and a
# Windows console cannot even print them. Every provider's output passes here.
_TYPOGRAPHIC = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": ", ", "―": ", ",
    "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',
    "…": "...", " ": " ", "•": " ", "→": " gives ",
})
_MD_INLINE = re.compile(r"(\*\*|__|`+)")
_MD_LINE = re.compile(r"^\s*(?:[#>]+\s*|[-*]\s+)", re.MULTILINE)


def clean_output(text: str) -> str:
    """Plain speakable text. Line structure and "1." numbering are kept (the
    ingest pass relies on one numbered line per sentence); markdown decoration
    and typographic glyphs are not."""
    s = (text or "").translate(_TYPOGRAPHIC)
    s = _MD_INLINE.sub("", s)
    s = _MD_LINE.sub("", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*,\s*,", ",", s)
    s = re.sub(r" +([,.;:!?])", r"\1", s)
    return "\n".join(l.strip() for l in s.splitlines()).strip()


class _LazyProvider:
    """Wraps a provider SDK that is imported only on first use, so the agent
    layer has no hard dependency on any vendor package."""

    def __init__(self, provider: str, model: str | None, *, max_tokens: int = 1200,
                 reasoning_effort: str | None = None, timeout: float = 20.0) -> None:
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        # Reasoning models (gpt-oss on Groq) think before answering; at the default
        # effort a one-word intent takes 2-3 s. "low" keeps it around half a second
        # with no visible quality loss on our short prompts. Only sent when set,
        # because non-reasoning models reject the parameter.
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self._client = None

    def complete(self, system: str, user: str) -> str:
        try:
            return clean_output(self._complete(system, user))
        except Exception as exc:  # noqa: BLE001 -- degrade, never crash the voice loop
            logger.warning("LLM %s/%s failed (%s); falling back", self.provider, self.model, exc)
            return ""

    def _complete(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            if self._client is None:
                import anthropic  # type: ignore
                self._client = anthropic.Anthropic(timeout=self.timeout)
            msg = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

        if self.provider in ("openai", "groq"):
            if self._client is None:
                from openai import OpenAI  # type: ignore
                kw: dict = {"timeout": self.timeout, "max_retries": 1}
                if self.provider == "groq":
                    kw.update(base_url="https://api.groq.com/openai/v1",
                              api_key=os.getenv("GROQ_API_KEY"))
                self._client = OpenAI(**kw)
            extra: dict = {}
            if self.reasoning_effort:
                extra["reasoning_effort"] = self.reasoning_effort
            r = self._client.chat.completions.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                **extra,
            )
            return (r.choices[0].message.content or "").strip()

        raise ValueError(f"unknown LLM provider: {self.provider}")


def _default_effort(provider: str, model: str) -> str | None:
    """Reasoning models we know about get "low"; everything else gets nothing."""
    from config import LLM_REASONING_EFFORT
    if LLM_REASONING_EFFORT:
        return None if LLM_REASONING_EFFORT.lower() == "none" else LLM_REASONING_EFFORT
    m = model.lower()
    if provider in ("groq", "openai") and ("gpt-oss" in m or m.startswith("o1") or m.startswith("o3")
                                            or m.startswith("o4") or "qwen3" in m or "deepseek-r1" in m):
        return "low"
    return None


def make_llm(provider: str | None, model: str | None):
    provider = (provider or "stub").lower()
    if provider == "stub":
        return StubLLM()
    if not model:
        logger.warning("LLM provider %s set but no model given; using stub", provider)
        return StubLLM()
    from config import LLM_MAX_TOKENS, LLM_TIMEOUT_S
    return _LazyProvider(provider, model, max_tokens=LLM_MAX_TOKENS,
                         reasoning_effort=_default_effort(provider, model), timeout=LLM_TIMEOUT_S)


def fast():
    from config import LLM_FAST_MODEL, LLM_FAST_PROVIDER
    return make_llm(LLM_FAST_PROVIDER, LLM_FAST_MODEL)


def strong():
    from config import LLM_STRONG_MODEL, LLM_STRONG_PROVIDER
    return make_llm(LLM_STRONG_PROVIDER, LLM_STRONG_MODEL)
