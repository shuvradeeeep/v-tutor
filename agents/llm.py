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
import threading
import time
from collections import deque

logger = logging.getLogger("v-tutor.llm")


class StubLLM:
    provider = "stub"
    model = "-"

    def complete(self, system: str, user: str) -> str:
        return ""


class RateMeter:
    """
    Rolling tokens-per-minute meter, one per model.

    Providers count `max_tokens` as reserved quota, so a burst of section
    rewrites at lesson start can exhaust a minute's budget in seconds. Rather
    than let those calls 429 -- which returns "" and drops the tutor to its
    deterministic fallback mid-lesson -- callers wait for room. Background work
    can wait a long time; the answer path waits briefly and then goes anyway,
    because a late answer is worse than a slightly rate-limited one.
    """

    def __init__(self, limit_per_min: int) -> None:
        self.limit = limit_per_min
        self._spent: deque[tuple[float, int]] = deque()   # (when, tokens)
        self._lock = threading.Lock()
        self.waited_s = 0.0          # all waiting, including background prep
        self.waited_fg_s = 0.0       # waiting the learner actually felt
        self.throttles = 0

    def _prune(self, now: float) -> int:
        while self._spent and now - self._spent[0][0] >= 60.0:
            self._spent.popleft()
        return sum(n for _, n in self._spent)

    def spent_last_minute(self) -> int:
        with self._lock:
            return self._prune(time.monotonic())

    def reserve(self, tokens: int, max_wait: float, ceiling: float = 1.0) -> None:
        """
        ceiling: the share of the limit this caller may consume. Background
        section prep is capped below 1.0 so it cannot spend the whole minute
        that the learner's next question needs -- prep can wait, an answer
        cannot.
        """
        if self.limit <= 0:
            return
        budget = max(tokens, int(self.limit * ceiling))
        deadline = time.monotonic() + max_wait
        while True:
            with self._lock:
                now = time.monotonic()
                used = self._prune(now)
                if used + tokens <= budget or now >= deadline:
                    self._spent.append((now, tokens))
                    return
                # Wait only until the oldest spend ages out of the window.
                sleep_for = min(60.0 - (now - self._spent[0][0]) + 0.05, deadline - now)
                self.throttles += 1
            if sleep_for > 0:
                self.waited_s += sleep_for
                if ceiling >= 1.0:
                    self.waited_fg_s += sleep_for
                logger.info("LLM rate: %d/%d tokens used this minute (budget %d), waiting %.1fs",
                            used, self.limit, budget, sleep_for)
                time.sleep(sleep_for)

    def correct(self, estimated: int, actual: int) -> None:
        """Replace the estimate with what the provider actually charged."""
        if self.limit <= 0 or actual == estimated:
            return
        with self._lock:
            for i in range(len(self._spent) - 1, -1, -1):
                when, n = self._spent[i]
                if n == estimated:
                    self._spent[i] = (when, actual)
                    return


_meters: dict[str, RateMeter] = {}
_meters_lock = threading.Lock()


def meter_for(model: str, limit: int) -> RateMeter:
    """One meter per model: providers apply the limit per model, not per key."""
    with _meters_lock:
        m = _meters.get(model)
        if m is None or m.limit != limit:
            m = _meters[model] = RateMeter(limit)
        return m


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

    def __init__(self, provider: str, model: str | None, *, max_tokens: int = 400,
                 max_tokens_long: int | None = None, reasoning_effort: str | None = None,
                 timeout: float = 20.0, tpm_limit: int = 0, tpm_max_wait: float = 20.0) -> None:
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.max_tokens_long = max_tokens_long or max_tokens
        self.meter = meter_for(model or provider, tpm_limit)
        self.tpm_max_wait = tpm_max_wait
        # Reasoning models (gpt-oss on Groq) think before answering; at the default
        # effort a one-word intent takes 2-3 s. "low" keeps it around half a second
        # with no visible quality loss on our short prompts. Only sent when set,
        # because non-reasoning models reject the parameter.
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self._client = None

    # The answer path is what the learner is waiting on. It pauses only very
    # briefly for quota and then goes regardless: if that request is refused,
    # the 429 retry below recovers it with the provider's own suggested delay,
    # which is cheaper than making every answer wait for a limit that may not
    # even be close. Pacing exists to keep BACKGROUND prep out of the way.
    FOREGROUND_WAIT_FRACTION = 0.075
    # Share of the minute background prep may take. The rest is held for
    # whatever the learner asks next.
    BACKGROUND_CEILING = 0.6

    def complete(self, system: str, user: str) -> str:
        return self._guarded(system, user, self.max_tokens,
                             self.tpm_max_wait * self.FOREGROUND_WAIT_FRACTION, ceiling=1.0)

    def complete_long(self, system: str, user: str) -> str:
        """For section rewrites: a bigger output budget, and it can afford to
        wait for quota because nothing is being spoken while it runs."""
        return self._guarded(system, user, self.max_tokens_long, self.tpm_max_wait,
                             ceiling=self.BACKGROUND_CEILING)

    def _guarded(self, system: str, user: str, max_tokens: int, max_wait: float,
                 ceiling: float) -> str:
        # In flight, assume the worst the provider might charge (prompt at ~4
        # characters per token, plus the whole max_tokens reservation). Once it
        # reports usage, correct down to what was really charged: measured
        # against the live API, reserving max_tokens made the meter roughly 3x
        # pessimistic and inserted waits for a limit that was never reached.
        estimate = (len(system) + len(user)) // 4 + max_tokens
        self.meter.reserve(estimate, max_wait, ceiling)
        for attempt in (0, 1):
            try:
                text, charged = self._complete(system, user, max_tokens)
                if charged:
                    self.meter.correct(estimate, charged)
                return clean_output(text)
            except Exception as exc:  # noqa: BLE001 -- degrade, never crash the voice loop
                delay = _retry_after(exc)
                if attempt == 0 and delay is not None and delay <= max(max_wait, 5.0):
                    # The provider told us exactly how long to wait. Waiting
                    # beats answering from the deterministic fallback.
                    logger.info("LLM %s/%s rate limited; retrying in %.1fs", self.provider, self.model, delay)
                    time.sleep(delay)
                    continue
                logger.warning("LLM %s/%s failed (%s); falling back", self.provider, self.model, exc)
                return ""
        return ""

    def _complete(self, system: str, user: str, max_tokens: int) -> tuple[str, int]:
        """Returns (text, tokens_charged). 0 = the provider did not say."""
        if self.provider == "anthropic":
            if self._client is None:
                import anthropic  # type: ignore
                self._client = anthropic.Anthropic(timeout=self.timeout)
            msg = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
            usage = getattr(msg, "usage", None)
            used = (getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)) if usage else 0
            return text, used

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
                model=self.model, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                **extra,
            )
            used = getattr(getattr(r, "usage", None), "total_tokens", 0) or 0
            return (r.choices[0].message.content or "").strip(), used

        raise ValueError(f"unknown LLM provider: {self.provider}")


_RETRY_AFTER = re.compile(r"try again in\s*([\d.]+)\s*(ms|s)?", re.IGNORECASE)


def _retry_after(exc: Exception) -> float | None:
    """Seconds the provider asked us to wait, if this was a rate limit."""
    msg = str(exc)
    if "rate_limit" not in msg and "429" not in msg:
        return None
    m = _RETRY_AFTER.search(msg)
    if not m:
        return 2.0                                  # rate limited without advice
    value = float(m.group(1))
    return value / 1000.0 if (m.group(2) or "s").lower() == "ms" else value


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
    from config import (LLM_MAX_TOKENS, LLM_MAX_TOKENS_LONG, LLM_TIMEOUT_S,
                        LLM_TPM_LIMIT, LLM_TPM_MAX_WAIT)
    return _LazyProvider(provider, model, max_tokens=LLM_MAX_TOKENS,
                         max_tokens_long=LLM_MAX_TOKENS_LONG,
                         reasoning_effort=_default_effort(provider, model), timeout=LLM_TIMEOUT_S,
                         tpm_limit=LLM_TPM_LIMIT, tpm_max_wait=LLM_TPM_MAX_WAIT)


def fast():
    from config import LLM_FAST_MODEL, LLM_FAST_PROVIDER
    return make_llm(LLM_FAST_PROVIDER, LLM_FAST_MODEL)


def strong():
    from config import LLM_STRONG_MODEL, LLM_STRONG_PROVIDER
    return make_llm(LLM_STRONG_PROVIDER, LLM_STRONG_MODEL)
