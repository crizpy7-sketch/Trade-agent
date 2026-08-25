"""Structured observability.

1.x logged with print statements and had no run identity, so "why was yesterday
slow?" and "what did the options agent cost last month?" were unanswerable.

Everything here writes to SQLite (migration 004) as well as to the log, because
an operator needs to query history, not grep it. Secrets are redacted on the way
in — the log is the most common place credentials escape.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

from .security import redact

log = logging.getLogger("marketswarm.observability")


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    INVESTIGATING = "INVESTIGATING"
    WAITING = "WAITING"
    REVIEWING = "REVIEWING"
    RED_TEAM = "RED_TEAM"
    REVISING = "REVISING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class AgentTrace:
    agent: str
    run_id: int | None = None
    investigation_id: str | None = None
    status: AgentStatus = AgentStatus.IDLE
    started_at: str | None = None
    finished_at: str | None = None
    latency_ms: int = 0
    retries: int = 0
    error: str | None = None
    tool_calls: int = 0
    providers: list[str] = field(default_factory=list)
    llm_model: str | None = None
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    cost_usd: float = 0.0
    evidence_count: int = 0
    signal_count: int = 0
    headline: str | None = None
    findings: list[str] = field(default_factory=list)

    def to_row(self) -> tuple:
        return (
            self.run_id, self.investigation_id, self.agent, self.status.value,
            self.started_at, self.finished_at, self.latency_ms, self.retries,
            redact(self.error) if self.error else None, self.tool_calls,
            json.dumps(self.providers), self.llm_model, self.llm_input_tokens,
            self.llm_output_tokens, self.cost_usd, self.evidence_count,
            self.signal_count,
            redact(self.headline) if self.headline else None,
            json.dumps([redact(f) for f in self.findings]) if self.findings else None,
        )


class Observatory:
    """Central recorder. Safe to use with `conn=None` — tracing must never be
    the reason a research run fails."""

    def __init__(self, conn: sqlite3.Connection | None = None,
                 run_id: int | None = None):
        self.conn = conn
        self.run_id = run_id
        self.session_id = f"obs_{uuid.uuid4().hex[:8]}"
        self._traces: dict[str, AgentTrace] = {}
        self._status: dict[str, AgentStatus] = {}
        self._t0 = time.monotonic()

    # ---------- agent tracing ----------

    @contextmanager
    def trace_agent(self, agent: str, investigation_id: str | None = None):
        t = AgentTrace(
            agent=agent, run_id=self.run_id, investigation_id=investigation_id,
            status=AgentStatus.INVESTIGATING,
            started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        self._traces[agent] = t
        self._status[agent] = AgentStatus.INVESTIGATING
        start = time.monotonic()
        try:
            yield t
            t.status = AgentStatus.COMPLETE
        except Exception as exc:  # noqa: BLE001 — record then re-raise
            t.status = AgentStatus.FAILED
            t.error = str(exc)[:500]
            raise
        finally:
            t.latency_ms = int((time.monotonic() - start) * 1000)
            t.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
            self._status[agent] = t.status
            self._persist_trace(t)

    def record_report(self, report, investigation_id: str | None = None) -> None:
        """Capture an AgentReport produced by the existing 1.x contract.

        Defensive by design: tracing a report must never be the reason a
        research run dies, so a malformed report is recorded as unknown rather
        than raising.
        """
        name = getattr(report, "agent", None)
        if not name:
            log.debug("skipping trace for a report with no agent name")
            return
        t = self._traces.get(name) or AgentTrace(
            agent=name, run_id=self.run_id, investigation_id=investigation_id)
        t.status = {
            "ok": AgentStatus.COMPLETE, "degraded": AgentStatus.COMPLETE,
            "failed": AgentStatus.FAILED, "skipped": AgentStatus.SKIPPED,
        }.get(getattr(report, "status", "ok"), AgentStatus.COMPLETE)
        t.latency_ms = t.latency_ms or int(getattr(report, "duration_ms", 0))
        t.error = getattr(report, "error", None)
        t.evidence_count = len(getattr(report, "evidence", []) or [])
        t.signal_count = len(getattr(report, "signals", []) or [])
        t.headline = getattr(report, "headline", None)
        # Capped: this is a Discord answer, not an archive, and an unbounded
        # findings list from one agent would crowd out the other fifteen.
        t.findings = [str(f) for f in (getattr(report, "findings", None) or [])][:12]
        t.finished_at = t.finished_at or dt.datetime.now(dt.timezone.utc).isoformat()
        self._traces[name] = t
        self._status[name] = t.status
        self._persist_trace(t)

    def _persist_trace(self, t: AgentTrace) -> None:
        if self.conn is None:
            return
        try:
            self.conn.execute(
                """INSERT INTO agent_runs
                   (run_id, investigation_id, agent, status, started_at, finished_at,
                    latency_ms, retries, error, tool_calls, providers, llm_model,
                    llm_input_tokens, llm_output_tokens, cost_usd, evidence_count,
                    signal_count, headline, findings)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", t.to_row())
            self.conn.commit()
        except sqlite3.Error as exc:
            log.debug("trace persist failed: %s", exc)

    # ---------- events ----------

    def event(self, component: str, event: str, level: str = "info",
              investigation_id: str | None = None, **payload) -> None:
        msg = redact(f"{component}: {event}")
        getattr(log, level if level in ("debug", "info", "warning", "error") else "info")(msg)
        if self.conn is None:
            return
        try:
            self.conn.execute(
                """INSERT INTO system_events
                   (created_at, level, component, event, run_id, investigation_id, payload)
                   VALUES (?,?,?,?,?,?,?)""",
                (dt.datetime.now(dt.timezone.utc).isoformat(), level, component,
                 redact(event), self.run_id, investigation_id,
                 redact(json.dumps(payload, default=str))))
            self.conn.commit()
        except sqlite3.Error as exc:
            log.debug("event persist failed: %s", exc)

    # ---------- live status ----------

    def status_snapshot(self) -> dict:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "elapsed_s": round(time.monotonic() - self._t0, 1),
            "agents": {a: s.value for a, s in self._status.items()},
            "active": [a for a, s in self._status.items()
                       if s in (AgentStatus.INVESTIGATING, AgentStatus.PLANNING,
                                AgentStatus.REVIEWING, AgentStatus.RED_TEAM,
                                AgentStatus.REVISING)],
            "failed": [a for a, s in self._status.items() if s is AgentStatus.FAILED],
            "total_cost_usd": round(sum(t.cost_usd for t in self._traces.values()), 4),
            "total_latency_ms": sum(t.latency_ms for t in self._traces.values()),
            "total_evidence": sum(t.evidence_count for t in self._traces.values()),
        }

    def slowest(self, n: int = 5) -> list[dict]:
        traces = sorted(self._traces.values(), key=lambda t: -t.latency_ms)[:n]
        return [{"agent": t.agent, "latency_ms": t.latency_ms,
                 "status": t.status.value} for t in traces]

    def cost_report(self) -> dict:
        by_agent = {t.agent: round(t.cost_usd, 4)
                    for t in self._traces.values() if t.cost_usd > 0}
        return {
            "total_usd": round(sum(by_agent.values()), 4),
            "by_agent": by_agent,
            "llm_calls": sum(1 for t in self._traces.values() if t.llm_model),
            "total_tokens": sum(t.llm_input_tokens + t.llm_output_tokens
                                for t in self._traces.values()),
        }


def configure_logging(verbose: bool = False, log_file=None) -> None:
    """Structured logging with secret redaction on every handler."""
    from .security import RedactingFilter

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        from pathlib import Path
        p = Path(log_file).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p))

    fmt = ("%(asctime)s %(levelname)-7s %(name)-32s %(message)s")
    for h in handlers:
        h.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        h.addFilter(RedactingFilter())

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        handlers=handlers, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
