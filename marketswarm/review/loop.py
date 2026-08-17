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

from .gate import Finding, ReviewDecision, ReviewGate, ReviewStatus, Severity

log = logging.getLogger("marketswarm.review.loop")

# A critic and an optional investigator. Both are plain callables so the loop
# can be tested without agents, an LLM, or a network.
Critic = Callable[[dict, int], list[Finding]]
Investigator = Callable[[dict, list[str]], dict]


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

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "stage": self.stage,
            "created_at": self.created_at,
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
            findings = list(critic(current, i))
            decision = self.gate.review(
                findings,
                original_confidence=int(current.get("confidence", 50)),
                iteration=i,
                max_iterations=self.max_iterations,
                evidence_already_requested=evidence_requested,
            )
            history.append(RevisionRecord(i, "red_team", decision, findings,
                                          copy.deepcopy(current)))
            log.info("review iter %d: %s", i, decision.summary())

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
                else:
                    # No investigator wired: the questions stay open and are
                    # published as unresolved rather than quietly dropped.
                    current["open_questions"] = list(decision.required_followups)
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
