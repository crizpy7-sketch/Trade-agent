"""Failure handling: circuit breakers, degradation levels, and honest gaps.

1.x contained failures per agent, which was right, but it had two holes:

  * a dead provider was retried on every symbol of every run, turning one
    outage into hundreds of timeouts;
  * a run that lost a critical agent published anyway, so option ideas could be
    built with no options data behind them.

Both are fixed here. The circuit breaker stops hammering a broken dependency,
and `DegradationTracker` distinguishes "we lost something optional" from "we
lost something the conclusion depended on" — and makes the second one visible
in the output rather than silently absent.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger("marketswarm.resilience")


class BreakerState(str, Enum):
    CLOSED = "closed"        # healthy
    OPEN = "open"            # failing; calls short-circuit
    HALF_OPEN = "half_open"  # probing for recovery


@dataclass
class CircuitBreaker:
    """One breaker per dependency (provider host, agent, model).

    After `failure_threshold` consecutive failures the circuit opens and calls
    fail immediately for `recovery_seconds` — the caller degrades instantly
    instead of waiting on a timeout it has already learned to expect.
    """

    name: str
    failure_threshold: int = 4
    recovery_seconds: float = 120.0
    half_open_successes: int = 2

    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    opened_at: float | None = None
    total_calls: int = 0
    total_failures: int = 0
    total_short_circuits: int = 0

    def _maybe_half_open(self) -> None:
        if (self.state is BreakerState.OPEN and self.opened_at is not None
                and time.monotonic() - self.opened_at >= self.recovery_seconds):
            self.state = BreakerState.HALF_OPEN
            self.consecutive_successes = 0
            log.info("circuit %s half-open, probing", self.name)

    @property
    def is_open(self) -> bool:
        self._maybe_half_open()
        return self.state is BreakerState.OPEN

    def allow(self) -> bool:
        self._maybe_half_open()
        if self.state is BreakerState.OPEN:
            self.total_short_circuits += 1
            return False
        return True

    def record_success(self) -> None:
        self.total_calls += 1
        self.consecutive_failures = 0
        if self.state is BreakerState.HALF_OPEN:
            self.consecutive_successes += 1
            if self.consecutive_successes >= self.half_open_successes:
                self.state = BreakerState.CLOSED
                self.opened_at = None
                log.info("circuit %s closed — dependency recovered", self.name)

    def record_failure(self) -> None:
        self.total_calls += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        if self.state is BreakerState.HALF_OPEN:
            self.state = BreakerState.OPEN
            self.opened_at = time.monotonic()
            log.warning("circuit %s reopened on probe failure", self.name)
        elif self.consecutive_failures >= self.failure_threshold:
            self.state = BreakerState.OPEN
            self.opened_at = time.monotonic()
            log.warning("circuit %s opened after %d consecutive failures",
                        self.name, self.consecutive_failures)

    def stats(self) -> dict:
        return {
            "name": self.name, "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "total_calls": self.total_calls, "total_failures": self.total_failures,
            "short_circuits": self.total_short_circuits,
            "failure_rate": (self.total_failures / self.total_calls
                             if self.total_calls else 0.0),
        }


class BreakerRegistry:
    def __init__(self, **defaults):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._defaults = defaults

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name=name, **self._defaults)
        return self._breakers[name]

    def all_stats(self) -> list[dict]:
        return [b.stats() for b in self._breakers.values()]

    def open_circuits(self) -> list[str]:
        return [n for n, b in self._breakers.items() if b.is_open]

    def reset(self, name: str | None = None) -> None:
        targets = [self._breakers[name]] if name else list(self._breakers.values())
        for b in targets:
            b.state = BreakerState.CLOSED
            b.consecutive_failures = 0
            b.opened_at = None


BREAKERS = BreakerRegistry()


class CircuitOpenError(RuntimeError):
    pass


async def call_with_resilience(
    coro_factory,
    breaker: CircuitBreaker,
    retries: int = 2,
    base_delay: float = 0.5,
    timeout: float | None = 20.0,
    fallback=None,
):
    """Run an async call behind a breaker, with capped exponential backoff.

    `coro_factory` is a zero-arg callable returning a fresh coroutine — a
    coroutine object cannot be awaited twice, so retrying requires a factory.
    """
    if not breaker.allow():
        log.debug("circuit %s open — short-circuiting", breaker.name)
        if fallback is not None:
            return fallback
        raise CircuitOpenError(f"circuit {breaker.name} is open")

    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            coro = coro_factory()
            result = (await asyncio.wait_for(coro, timeout=timeout)
                      if timeout else await coro)
            breaker.record_success()
            return result
        except asyncio.TimeoutError as exc:
            last = exc
            log.debug("%s timed out (attempt %d)", breaker.name, attempt + 1)
        except Exception as exc:  # noqa: BLE001
            last = exc
            log.debug("%s failed (attempt %d): %s", breaker.name, attempt + 1, exc)

        if attempt < retries:
            # Jittered backoff: synchronised retries across symbols would
            # hammer a recovering provider all at once.
            await asyncio.sleep(base_delay * (2 ** attempt) + random.random() * 0.3)

    breaker.record_failure()
    if fallback is not None:
        return fallback
    raise last if last else RuntimeError(f"{breaker.name} failed")


# --------------------------------------------------------------------------
# degradation
# --------------------------------------------------------------------------

class DegradationLevel(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"          # optional inputs missing
    CRITICALLY_DEGRADED = "critically_degraded"   # a conclusion-bearing input missing
    UNUSABLE = "unusable"          # cannot produce a defensible answer

    @property
    def can_publish_ideas(self) -> bool:
        return self in (DegradationLevel.HEALTHY, DegradationLevel.DEGRADED)


# Losing one of these does not merely narrow the report — it removes something
# the conclusions are built on.
CRITICAL_AGENTS: frozenset[str] = frozenset({
    "technicals",       # no levels ⇒ no brackets
    "cross_verify",     # no fusion ⇒ no probability
    "red_team",         # no adversary ⇒ no reviewed recommendation
})

# Losing one of these invalidates a specific *class* of output only.
CAPABILITY_CRITICAL: dict[str, str] = {
    "options_flow": "option ideas",
    "earnings": "earnings-reaction ideas",
    "sec_filings": "filing-driven ideas",
}


@dataclass
class DegradationTracker:
    """Tracks what is missing and what that invalidates.

    The rule this enforces: missing evidence is represented explicitly. A
    conclusion is never allowed to rest on data that silently was not there.
    """

    failed_agents: set[str] = field(default_factory=set)
    degraded_agents: set[str] = field(default_factory=set)
    failed_providers: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def record_agent(self, name: str, status: str) -> None:
        if status == "failed":
            self.failed_agents.add(name)
        elif status == "degraded":
            self.degraded_agents.add(name)

    def record_provider(self, name: str) -> None:
        self.failed_providers.add(name)

    @property
    def level(self) -> DegradationLevel:
        if self.failed_agents & CRITICAL_AGENTS:
            return DegradationLevel.UNUSABLE
        if self.failed_agents & set(CAPABILITY_CRITICAL):
            return DegradationLevel.CRITICALLY_DEGRADED
        if self.failed_agents or len(self.degraded_agents) > 2:
            return DegradationLevel.DEGRADED
        return DegradationLevel.HEALTHY

    def suppressed_outputs(self) -> list[str]:
        """Output classes that must NOT be produced given what is missing."""
        out = []
        for agent, capability in CAPABILITY_CRITICAL.items():
            if agent in self.failed_agents:
                out.append(capability)
        if self.failed_agents & CRITICAL_AGENTS:
            out.append("all directional ideas")
        return out

    def missing_evidence_statement(self) -> str:
        """Text for the report. An absent input must be stated, not omitted."""
        if self.level is DegradationLevel.HEALTHY:
            return ""
        parts = []
        if self.failed_agents:
            parts.append(f"no data from: {', '.join(sorted(self.failed_agents))}")
        if self.failed_providers:
            parts.append(f"providers unreachable: {', '.join(sorted(self.failed_providers))}")
        suppressed = self.suppressed_outputs()
        if suppressed:
            parts.append(f"suppressed as unsupportable: {', '.join(suppressed)}")
        return ("Coverage gap — " + "; ".join(parts)
                + ". Conclusions below are drawn from a narrower evidence base "
                  "than normal and should be weighted accordingly.")

    def data_quality(self) -> str:
        return {
            DegradationLevel.HEALTHY: "ok",
            DegradationLevel.DEGRADED: "degraded",
            DegradationLevel.CRITICALLY_DEGRADED: "poor",
            DegradationLevel.UNUSABLE: "poor",
        }[self.level]

    def summary(self) -> dict:
        return {
            "level": self.level.value,
            "failed_agents": sorted(self.failed_agents),
            "degraded_agents": sorted(self.degraded_agents),
            "failed_providers": sorted(self.failed_providers),
            "suppressed_outputs": self.suppressed_outputs(),
            "can_publish_ideas": self.level.can_publish_ideas,
            "statement": self.missing_evidence_statement(),
        }
