"""Agent contract.

Every swarm member returns the same shape: findings a human can read, evidence
with provenance, and zero or more directional Signals the fusion layer can
combine. Agents never talk to each other directly — they publish to the bus and
the orchestrator resolves conflicts. That keeps failure isolated and makes each
agent independently testable.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..providers.base import Evidence
from ..stats.bayes import Signal

log = logging.getLogger("marketswarm.agents")


@dataclass
class AgentReport:
    agent: str
    headline: str                                  # one-line summary for the report
    findings: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5                        # the agent's own self-assessment
    status: str = "ok"                             # ok | degraded | failed | skipped
    error: str | None = None
    duration_ms: int = 0

    @property
    def usable(self) -> bool:
        return self.status in ("ok", "degraded")

    def add(self, finding: str) -> None:
        self.findings.append(finding)

    def cite(self, claim: str, source: str, url: str | None = None,
             reliability: float = 0.6, **kw) -> Evidence:
        ev = Evidence(claim=claim, source=source, url=url, reliability=reliability, **kw)
        self.evidence.append(ev)
        return ev

    def signal(self, name: str, probability: float, weight: float = 1.0,
               source: str = "", note: str = "") -> Signal:
        s = Signal(name=name, probability=probability, weight=weight,
                   source=source or self.agent, note=note)
        self.signals.append(s)
        return s

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "headline": self.headline,
            "status": self.status,
            "confidence": self.confidence,
            "findings": self.findings,
            "signals": [
                {"name": s.name, "p": round(s.probability, 4), "weight": round(s.weight, 3), "note": s.note}
                for s in self.signals
            ],
            "evidence": [e.to_dict() for e in self.evidence],
            "data": {k: v for k, v in self.data.items() if k != "chain"},
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


# The standard per-agent budget. An agent that declares something else is
# stating a *relative* need — "I want twice the standard" — which the
# orchestrator scales against the operator's configured budget rather than
# treating as an absolute. Config and this constant share a value so that an
# untouched config changes nothing.
DEFAULT_AGENT_TIMEOUT = 45.0


class BaseAgent:
    """Subclasses implement `run(ctx) -> AgentReport`."""

    name: str = "base"
    description: str = ""
    timeout: float = DEFAULT_AGENT_TIMEOUT
    # Agents whose output this one consumes. The orchestrator uses this to
    # stage execution; agents with no dependencies all run concurrently.
    depends_on: tuple[str, ...] = ()

    async def run(self, ctx: "SwarmContext") -> AgentReport:  # noqa: F821
        raise NotImplementedError

    async def execute(self, ctx: "SwarmContext") -> AgentReport:  # noqa: F821
        """Wrap run() with timing, timeout and failure containment."""
        started = time.monotonic()
        try:
            report = await asyncio.wait_for(self.run(ctx), timeout=self.timeout)
        except asyncio.TimeoutError:
            log.warning("%s timed out after %.0fs", self.name, self.timeout)
            report = AgentReport(agent=self.name, headline=f"{self.name}: timed out",
                                 status="failed", error=f"timeout after {self.timeout}s")
        except Exception as exc:  # noqa: BLE001 — one agent must not kill the run
            log.exception("%s failed", self.name)
            report = AgentReport(agent=self.name, headline=f"{self.name}: failed",
                                 status="failed", error=str(exc))
        report.duration_ms = int((time.monotonic() - started) * 1000)
        return report


@dataclass
class SwarmContext:
    """Shared state passed to every agent."""

    run_date: dt.date
    prev_session: dt.date
    universe: list[str]
    index_symbols: list[str]
    market: Any                      # providers.market.MarketData
    options: Any                     # providers.options.OptionsData
    news: Any                        # providers.news.NewsData
    econ: Any                        # providers.econ.EconData
    edgar: Any                       # providers.edgar.EdgarData
    earnings: Any                    # providers.earnings.EarningsData
    config: Any
    reports: dict[str, AgentReport] = field(default_factory=dict)
    agent_weights: dict[str, float] = field(default_factory=dict)
    lessons: list[dict] = field(default_factory=list)

    # Session context, filled in by the orchestrator once the Event Brain has
    # classified the morning. Agents may read it; the value is "any" when
    # nothing was detected, which is an observation rather than a guess.
    detected_events: list = field(default_factory=list)
    event_type: str = "any"
    regime: str = ""
    started_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))

    def report_of(self, agent: str) -> AgentReport | None:
        r = self.reports.get(agent)
        return r if r and r.usable else None

    def data_of(self, agent: str, key: str, default: Any = None) -> Any:
        r = self.report_of(agent)
        return r.data.get(key, default) if r else default

    def weight_for(self, agent: str, default: float = 1.0) -> float:
        """Learned reliability weight, defaulting to neutral for a new agent."""
        return float(self.agent_weights.get(agent, default))
