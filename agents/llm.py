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

logger = logging.getLogger("v-tutor.llm")


class StubLLM:
    provider = "stub"
    model = "-"

    def complete(self, system: str, user: str) -> str:
        return ""


class _LazyProvider:
    """Wraps a provider SDK that is imported only on first use, so the agent
    layer has no hard dependency on any vendor package."""

    def __init__(self, provider: str, model: str | None) -> None:
        self.provider = provider
        self.model = model
        self._client = None

    def complete(self, system: str, user: str) -> str:
        try:
            return self._complete(system, user)
        except Exception as exc:  # noqa: BLE001 -- degrade, never crash the voice loop
            logger.warning("LLM %s/%s failed (%s); falling back", self.provider, self.model, exc)
            return ""

    def _complete(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            if self._client is None:
                import anthropic  # type: ignore
                self._client = anthropic.Anthropic()
            msg = self._client.messages.create(
                model=self.model, max_tokens=600, system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

        if self.provider in ("openai", "groq"):
            if self._client is None:
                from openai import OpenAI  # type: ignore
                kw = {}
                if self.provider == "groq":
                    kw = {"base_url": "https://api.groq.com/openai/v1",
                          "api_key": os.getenv("GROQ_API_KEY")}
                self._client = OpenAI(**kw)
            r = self._client.chat.completions.create(
                model=self.model, max_tokens=600,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return (r.choices[0].message.content or "").strip()

        raise ValueError(f"unknown LLM provider: {self.provider}")


def make_llm(provider: str | None, model: str | None):
    provider = (provider or "stub").lower()
    if provider == "stub":
        return StubLLM()
    if not model:
        logger.warning("LLM provider %s set but no model given; using stub", provider)
        return StubLLM()
    return _LazyProvider(provider, model)


def fast():
    from config import LLM_FAST_MODEL, LLM_FAST_PROVIDER
    return make_llm(LLM_FAST_PROVIDER, LLM_FAST_MODEL)


def strong():
    from config import LLM_STRONG_MODEL, LLM_STRONG_PROVIDER
    return make_llm(LLM_STRONG_PROVIDER, LLM_STRONG_MODEL)
