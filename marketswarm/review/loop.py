"""The generate → attack → revise → attack again loop.

Bounded by construction: `max_iterations` caps the cycles, every pass must
either change something or terminate, and a repeated identical decision breaks
the loop rather than spinning. An agent that can argue with itself forever is a
cost centre, not a thinker.

Every iteration is recorded. The audit trail can reconstruct the original idea,
each objection, what was demanded, what changed, and why the final answer is
what it is.
"""

from __future__ import annotations

import copy
import datetime as dt
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable

from .gate import (Finding, ReviewDecision, ReviewExecutionStatus, ReviewGate,
                   ReviewStatus, Severity)

log = logging.getLogger("marketswarm.review.loop")

# A critic and an optional investigator. Both are plain callables so the loop
# can be tested without agents, an LLM, or a network.
Critic = Callable[[dict, int], list[Finding]]
Investigator = Callable[[dict, list[str]], dict]


@dataclass
class CriticResult:
    """What one adversarial pass produced, and the state it was produced from.

    A bare list of findings cannot answer the two questions that matter after
    follow-up research: did the reviewer actually run, and did it look at the
    new evidence? Carrying the execution status and the evidence-graph version
    alongside the findings makes both auditable, and makes a stale second round
    detectable instead of invisible.

    A critic may still return a plain list; `CriticResult.of` normalises it.
    """

    findings: list[Finding] = field(default_factory=list)
    execution_status: ReviewExecutionStatus = ReviewExecutionStatus.COMPLETED
    graph_version: int = 0
    evidence_nodes: int = 0
    note: str = ""

    @classmethod
    def of(cls, raw) -> "CriticResult":
        if isinstance(raw, CriticResult):
            return raw
        return cls(findings=list(raw))


@dataclass
class RevisionRecord:
    iteration: int
    stage: str                       # original | red_team | revision | final
    decision: ReviewDecision | None
    findings: list[Finding] = field(default_factory=list)
    snapshot: dict = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    # Which evidence state this round saw. Round 2 carrying the same version as
    # round 1 means the follow-up bought nothing, and that is worth recording
    # rather than inferring.
    graph_version: int = 0
    evidence_nodes: int = 0
    execution_status: str = ReviewExecutionStatus.COMPLETED.value
    agents_executed: list[str] = field(default_factory=list)

    @property
    def round_number(self) -> int:
        return self.iteration + 1

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "round_number": self.round_number,
            "stage": self.stage,
            "created_at": self.created_at,
            "graph_version": self.graph_version,
            "evidence_nodes": self.evidence_nodes,
            "execution_status": self.execution_status,
            "agents_executed": list(self.agents_executed),
            "decision": self.decision.to_dict() if self.decision else None,
            "findings": [f.to_dict() for f in self.findings],
            "snapshot": self.snapshot,
        }


@dataclass
class ReviewOutcome:
    recommendation_id: str
    final_idea: dict | None                     # None when rejected
    final_decision: ReviewDecision
    history: list[RevisionRecord] = field(default_factory=list)
    iterations_used: int = 0
    terminated_because: str = ""

    @property
    def rejected(self) -> bool:
        return self.final_idea is None

    @property
    def confidence_change(self) -> int:
        if not self.history:
            return 0
        first = self.history[0].decision
        return (self.final_decision.revised_confidence
                - (first.original_confidence if first
                   else self.final_decision.original_confidence))

    def audit_trail(self) -> list[dict]:
        return [r.to_dict() for r in self.history]

    def explain(self) -> str:
        lines = [f"Recommendation {self.recommendation_id}: "
                 f"{self.final_decision.summary()}"]
        for r in self.history:
            if r.decision:
                lines.append(f"  iter {r.iteration} [{r.stage}]: {r.decision.summary()}")
                for reason in r.decision.reasons[:3]:
                    lines.append(f"      · {reason}")
        lines.append(f"  terminated: {self.terminated_because}")
        return "\n".join(lines)


class ReviewLoop:
    def __init__(
        self,
        gate: ReviewGate | None = None,
        max_iterations: int = 2,
        critic: Critic | None = None,
        investigator: Investigator | None = None,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.gate = gate or ReviewGate()
        self.max_iterations = max_iterations
        self.critic = critic
        self.investigator = investigator

    def run(self, idea: dict, critic: Critic | None = None,
            investigator: Investigator | None = None) -> ReviewOutcome:
        critic = critic or self.critic
        if critic is None:
            raise ValueError("a critic is required")
        investigator = investigator or self.investigator

        rec_id = idea.get("recommendation_id") or f"rec_{uuid.uuid4().hex[:12]}"
        current = copy.deepcopy(idea)
        current.setdefault("recommendation_id", rec_id)

        history: list[RevisionRecord] = [
            RevisionRecord(0, "original", None, [], copy.deepcopy(current))
        ]

        decision: ReviewDecision | None = None
        terminated = "completed"
        evidence_requested = False
        seen_signatures: set[tuple] = set()
        i = 0

        for i in range(self.max_iterations):
            # A fresh adversarial pass against whatever the evidence state is
            # *now*. The critic re-derives findings each round; it does not
            # replay a cached objection set.
            result = CriticResult.of(critic(current, i))
            findings = list(result.findings)

            decision = self.gate.review(
                findings,
                original_confidence=int(current.get("confidence", 50)),
                iteration=i,
                max_iterations=self.max_iterations,
                evidence_already_requested=evidence_requested,
                execution_status=result.execution_status,
            )
            history.append(RevisionRecord(
                i, "red_team", decision, findings, copy.deepcopy(current),
                graph_version=result.graph_version,
                evidence_nodes=result.evidence_nodes,
                execution_status=result.execution_status.value,
                agents_executed=list(current.get("followup_agents", [])),
            ))
            log.info("review round %d: %s [graph v%d, %d nodes, review %s]",
                     i + 1, decision.summary(), result.graph_version,
                     result.evidence_nodes, result.execution_status.value)

            if decision.status is ReviewStatus.REVIEW_INCOMPLETE:
                terminated = f"adversarial review did not complete ({result.execution_status.value})"
                return ReviewOutcome(rec_id, None, decision, history, i + 1, terminated)

            if decision.status is ReviewStatus.REJECT:
                terminated = "rejected by review gate"
                return ReviewOutcome(rec_id, None, decision, history, i + 1, terminated)

            if not decision.status.needs_another_pass:
                terminated = ("approved" if decision.status is ReviewStatus.APPROVE
                              else "approved with reduced confidence")
                revised = self.gate.apply(current, decision)
                history.append(RevisionRecord(i, "final", decision, [],
                                              copy.deepcopy(revised or {})))
                return ReviewOutcome(rec_id, revised, decision, history, i + 1, terminated)

            # Loop guard: the same objections producing the same verdict twice
            # means another pass will not help.
            signature = (decision.status.value,
                         tuple(sorted(f.objection for f in findings)))
            if signature in seen_signatures:
                terminated = "no progress between iterations — stopping"
                revised = self.gate.apply(current, decision)
                history.append(RevisionRecord(i, "final", decision, [],
                                              copy.deepcopy(revised or {})))
                return ReviewOutcome(rec_id, revised, decision, history, i + 1, terminated)
            seen_signatures.add(signature)

            # --- act on the decision ---
            if decision.status is ReviewStatus.REQUEST_MORE_RESEARCH:
                evidence_requested = True
                if investigator:
                    try:
                        current = investigator(copy.deepcopy(current),
                                               list(decision.required_followups))
                    except Exception as exc:  # noqa: BLE001 — investigation is best-effort
                        log.warning("follow-up investigation failed: %s", exc)
                        current["open_questions"] = list(decision.required_followups)
                        # A failed investigation is not a satisfied one. The
                        # next round must see that the demanded evidence never
                        # arrived, or the loop launders a request into an
                        # answer by doing nothing.
                        current["followup_failed"] = True
                        current["followup_error"] = str(exc)[:200]
                else:
                    # No investigator wired: the questions stay open and are
                    # published as unresolved rather than quietly dropped.
                    current["open_questions"] = list(decision.required_followups)
                    current["followup_failed"] = True
                current["confidence"] = decision.revised_confidence
                history.append(RevisionRecord(i, "revision", decision, [],
                                              copy.deepcopy(current)))

            elif decision.status is ReviewStatus.MODIFY:
                modified = self.gate.apply(current, decision)
                if modified is None:
                    terminated = "modification rejected the idea"
                    return ReviewOutcome(rec_id, None, decision, history, i + 1, terminated)
                current = modified
                history.append(RevisionRecord(i, "revision", decision, [],
                                              copy.deepcopy(current)))

        # Iteration budget exhausted. Publish what survived, at the reduced
        # confidence, with the open questions attached — never at face value.
        terminated = f"iteration limit ({self.max_iterations}) reached"
        assert decision is not None
        final = self.gate.apply(current, decision)
        if final is not None:
            final["confidence"] = min(final.get("confidence", 100),
                                      decision.revised_confidence)
            final["review_status"] = ReviewStatus.APPROVE_WITH_REDUCED_CONFIDENCE.value
            if decision.required_followups:
                final["open_questions"] = list(decision.required_followups)
        history.append(RevisionRecord(i, "final", decision, [],
                                      copy.deepcopy(final or {})))
        return ReviewOutcome(rec_id, final, decision, history, self.max_iterations,
                             terminated)


def redteam_execution_status(report) -> ReviewExecutionStatus:
    """Did the adversarial review actually run, and is its output usable?

    Read from the agent report rather than inferred from the findings list,
    because the whole point is that an empty findings list is ambiguous.
    """
    if report is None:
        return ReviewExecutionStatus.UNAVAILABLE

    status = str(getattr(report, "status", "") or "").lower()
    if status in ("timeout", "timed_out"):
        return ReviewExecutionStatus.TIMED_OUT
    if status in ("error", "failed"):
        return ReviewExecutionStatus.FAILED
    if status == "skipped":
        return ReviewExecutionStatus.UNAVAILABLE
    if not getattr(report, "usable", False):
        return ReviewExecutionStatus.FAILED

    data = getattr(report, "data", None)
    if not isinstance(data, dict) or "objections" not in data:
        # The agent claims success but produced nothing the gate can read.
        # Silently treating that as "no objections" is the bug this guards.
        return ReviewExecutionStatus.INVALID

    return ReviewExecutionStatus.COMPLETED


def findings_from_redteam_report(report, symbol: str | None = None) -> list[Finding]:
    """Adapt the existing RedTeamAgent output into structured findings.

    The 1.x agent already produces severity-tagged objections; this maps them
    onto the gate's vocabulary without rewriting a working agent.
    """
    out: list[Finding] = []
    if not report or not getattr(report, "data", None):
        return out

    for o in report.data.get("objections", []):
        sev_raw = str(o.get("severity", "medium")).lower()
        sev = {"info": Severity.INFO, "low": Severity.LOW, "medium": Severity.MEDIUM,
               "high": Severity.HIGH, "critical": Severity.CRITICAL}.get(
                   sev_raw, Severity.MEDIUM)
        objection = o.get("objection", "")
        if symbol and not _applies_to(objection, symbol):
            continue
        out.append(Finding(
            objection=objection,
            severity=sev,
            test=o.get("test", ""),
            category=o.get("category", "structural"),
            targets=[symbol] if symbol else [],
            requires_evidence=o.get("requires_evidence"),
            suggested_modification=o.get("suggested_modification"),
        ))
    return out


_TICKER_RE = __import__("re").compile(r"\b[A-Z]{2,5}\b")

# Uppercase words that are not tickers. Without this, "VIX up 10%" would look
# like an objection about a symbol called VIX.
_NOT_TICKERS = frozenset({
    "VIX", "CPI", "FOMC", "SEC", "EPS", "ETF", "IV", "OI", "ATR", "EMA", "RSI",
    "USD", "GDP", "PPI", "AI", "CEO", "CFO", "R", "EV", "PC", "US", "UK", "EU",
})


def _applies_to(objection: str, symbol: str) -> bool:
    """Does this objection bear on this idea?

    Fail-safe by design. An objection that names *another* symbol is filtered;
    an objection that names none applies to everything. The 1.x-era mistake was
    the opposite default — requiring an explicit symbol match silently discarded
    every general objection, including critical ones, which is precisely the
    class of finding that must never be dropped.
    """
    sym = symbol.upper()
    if sym in objection.upper():
        return True
    mentioned = {t for t in _TICKER_RE.findall(objection) if t not in _NOT_TICKERS}
    if mentioned:
        return sym in mentioned      # names other symbols, not this one
    return True                       # names none — general, so it applies
