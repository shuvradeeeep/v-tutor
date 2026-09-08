"""
Rate pacing and token budgets.

A 429 from the provider is not a loud failure: complete() returns "" and every
node falls back to its deterministic path, so the tutor quietly gets worse
mid-lesson. These tests cover the pacing that keeps a long session inside the
limit, and the retry that recovers when it slips anyway.
"""
from __future__ import annotations

import time

from agents.llm import RateMeter, _LazyProvider, _retry_after


def test_meter_lets_traffic_through_under_the_limit():
    m = RateMeter(8000)
    t0 = time.monotonic()
    for _ in range(10):
        m.reserve(500, max_wait=5)
    assert time.monotonic() - t0 < 0.5          # 5000 of 8000: no waiting
    assert m.spent_last_minute() == 5000
    assert m.throttles == 0


def test_meter_waits_when_the_minute_is_spent():
    m = RateMeter(1000)
    m.reserve(900, max_wait=5)
    t0 = time.monotonic()
    m.reserve(900, max_wait=0.3)                # would exceed: waits, then goes anyway
    waited = time.monotonic() - t0
    assert 0.2 <= waited < 2.0, waited
    assert m.throttles >= 1


def test_meter_off_when_no_limit_configured():
    m = RateMeter(0)
    m.reserve(10 ** 9, max_wait=5)
    assert m.spent_last_minute() == 0


def test_actual_usage_replaces_the_estimate():
    m = RateMeter(8000)
    m.reserve(1200, max_wait=1)
    m.correct(1200, 250)                        # provider charged far less
    assert m.spent_last_minute() == 250


def test_retry_after_reads_the_providers_own_advice():
    err = Exception("Error code: 429 - rate_limit_exceeded ... Please try again in 3.3675s.")
    assert abs(_retry_after(err) - 3.3675) < 1e-6
    assert _retry_after(Exception("Error code: 429 - rate_limit_exceeded")) == 2.0
    assert _retry_after(Exception("Error code: 500 - server error")) is None


def test_a_rate_limited_call_is_retried_once_then_succeeds():
    calls: list[int] = []

    class Flaky(_LazyProvider):
        def _complete(self, system, user, max_tokens):
            calls.append(max_tokens)
            if len(calls) == 1:
                raise RuntimeError("Error code: 429 - rate_limit_exceeded. "
                                   "Please try again in 0.05s.")
            return "the answer", 120

    p = Flaky("groq", "m", max_tokens=400, tpm_limit=8000, tpm_max_wait=4)
    assert p.complete("sys", "user") == "the answer"
    assert len(calls) == 2


def test_long_calls_get_the_bigger_budget_and_wait_longer():
    seen: list[int] = []

    class Sizes(_LazyProvider):
        def _complete(self, system, user, max_tokens):
            seen.append(max_tokens)
            return "ok", 0

    p = Sizes("groq", "m", max_tokens=400, max_tokens_long=800, tpm_limit=8000)
    p.complete("s", "u")
    p.complete_long("s", "u")
    assert seen == [400, 800]
