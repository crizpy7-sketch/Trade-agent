"""Investigation plans and their hard limits.

The budget is not advisory. An autonomous investigator without enforced caps is
an unbounded bill and an unbounded latency, and "it usually stops" is not a
safety property. `PlanBudget.exceeded()` is checked by the runner on every
iteration and the plan terminates with a recorded reason.
"""

from __future__ import annotations

import datetime as dt
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from .event_brain import DetectedEvent, EventPriority, EventType


class StopReason(str, Enum):
    SUFFICIENT_EVIDENCE = "sufficient_evidence"
    NO_NEW_INFORMATION = "no_new_information"
    ITERATION_LIMIT = "iteration_limit"
    COST_LIMIT = "cost_limit"
    TIME_LIMIT = "time_limit"
    AGENT_LIMIT = "agent_limit"
    CONTRADICTION_UNRESOLVED = "contradiction_unresolved"
    INSUFFICIENT_DATA = "insufficient_data"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PlanBudget:
    max_iterations: int = 3
    max_agents: int = 16
    max_cost: float = 40.0             # registry cost units
    max_seconds: float = 180.0
    started_at: float = field(default_factory=time.monotonic)

    iterations_used: int = 0
    agents_used: int = 0
    cost_used: float = 0.0

    def __post_init__(self):
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.max_agents < 1:
            raise ValueError("max_agents must be >= 1")

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def exceeded(self) -> StopReason | None:
        if self.iterations_used >= self.max_iterations:
            return StopReason.ITERATION_LIMIT
        if self.agents_used >= self.max_agents:
            return StopReason.AGENT_LIMIT
        if self.cost_used >= self.max_cost:
            return StopReason.COST_LIMIT
        if self.elapsed >= self.max_seconds:
            return StopReason.TIME_LIMIT
        return None

    def charge(self, agents: int = 0, cost: float = 0.0) -> None:
        self.agents_used += agents
        self.cost_used += cost

    def snapshot(self) -> dict:
        return {
            "iterations": f"{self.iterations_used}/{self.max_iterations}",
            "agents": f"{self.agents_used}/{self.max_agents}",
            "cost": f"{self.cost_used:.1f}/{self.max_cost:.1f}",
            "elapsed_s": round(self.elapsed, 1),
            "limit_s": self.max_seconds,
        }


@dataclass
class InvestigationPlan:
    subject: str
    trigger: str
    priority: EventPriority = EventPriority.NORMAL
    hypotheses: list[str] = field(default_factory=list)
    required_agents: list[str] = field(default_factory=list)
    optional_agents: list[str] = field(default_factory=list)
    evidence_needed: list[str] = field(default_factory=list)
    stop_conditions: list[str] = field(default_factory=list)
    events: list[DetectedEvent] = field(default_factory=list)
    budget: PlanBudget = field(default_factory=PlanBudget)
    id: str = field(default_factory=lambda: f"inv_{uuid.uuid4().hex[:12]}")
    created_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    status: str = "PLANNING"
    stop_reason: StopReason | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def event_types(self) -> set[EventType]:
        return {e.event_type for e in self.events}

    @property
    def all_agents(self) -> list[str]:
        seen, out = set(), []
        for a in self.required_agents + self.optional_agents:
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "trigger": self.trigger,
            "priority": self.priority.value,
            "hypotheses": self.hypotheses,
            "required_agents": self.required_agents,
            "optional_agents": self.optional_agents,
            "evidence_needed": self.evidence_needed,
            "stop_conditions": self.stop_conditions,
            "events": [e.to_dict() for e in self.events],
            "max_iterations": self.budget.max_iterations,
            "max_cost": self.budget.max_cost,
            "status": self.status,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "created_at": self.created_at,
            "notes": self.notes,
        }

    def describe(self) -> str:
        lines = [
            f"[{self.priority.value}] {self.subject} — {self.trigger}",
            f"  agents: {', '.join(self.all_agents) or 'none'}",
        ]
        if self.hypotheses:
            lines.append("  hypotheses:")
            lines.extend(f"    - {h}" for h in self.hypotheses[:5])
        if self.evidence_needed:
            lines.append(f"  evidence needed: {'; '.join(self.evidence_needed[:4])}")
        lines.append(f"  budget: {self.budget.snapshot()}")
        return "\n".join(lines)
