"""The Review Gate.

In 1.x the Red Team ran last, produced objections, and nothing consumed them.
The playbook was already final. Criticism that cannot change the output is
theatre, and it was the single worst defect in the system.

The gate makes findings binding. Every recommendation passes through it, and
the gate returns a `ReviewDecision` that the pipeline is obliged to apply:
confidence is lowered, the idea is modified, more research is demanded, or the
idea is dropped.

Two rules that are enforced rather than encouraged:

  * A CRITICAL finding can never resolve to APPROVE. There is no code path
    that lets one through untouched.
  * Every decision records what it rejected and why, so the audit trail can
    reconstruct the reasoning later.

The gate itself is deterministic. Whether a finding is severe is a judgement an
LLM may help make; what happens once it is severe is arithmetic and policy, and
belongs in code where it can be tested.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}[self.value]


class ReviewStatus(str, Enum):
    APPROVE = "APPROVE"
    APPROVE_WITH_REDUCED_CONFIDENCE = "APPROVE_WITH_REDUCED_CONFIDENCE"
    REQUEST_MORE_RESEARCH = "REQUEST_MORE_RESEARCH"
    MODIFY = "MODIFY"
    REJECT = "REJECT"

    @property
    def survives(self) -> bool:
        """Does the idea live on in some form?"""
        return self is not ReviewStatus.REJECT

    @property
    def needs_another_pass(self) -> bool:
        return self in (ReviewStatus.REQUEST_MORE_RESEARCH, ReviewStatus.MODIFY)


@dataclass
class Finding:
    """One red-team objection."""

    objection: str
    severity: Severity = Severity.MEDIUM
    test: str = ""
    category: str = "general"
    targets: list[str] = field(default_factory=list)   # symbols/claims affected
    invalidates_claim: str | None = None
    requires_evidence: str | None = None               # what would settle it
    suggested_modification: dict | None = None         # e.g. {"confidence_cap": 30}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        sev = d.get("severity", "medium")
        return cls(
            objection=d.get("objection", ""),
            severity=Severity(sev) if isinstance(sev, str) else sev,
            test=d.get("test", ""),
            category=d.get("category", "general"),
            targets=list(d.get("targets", [])),
            invalidates_claim=d.get("invalidates_claim"),
            requires_evidence=d.get("requires_evidence"),
            suggested_modification=d.get("suggested_modification"),
        )


@dataclass
class ReviewDecision:
    status: ReviewStatus
    original_confidence: int
    revised_confidence: int
    severity: Severity
    reasons: list[str] = field(default_factory=list)
    required_followups: list[str] = field(default_factory=list)
    rejected_claims: list[str] = field(default_factory=list)
    accepted_claims: list[str] = field(default_factory=list)
    modifications: dict = field(default_factory=dict)
    iteration: int = 0
    decided_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    # Stable handle so a published recommendation can cite the decision that
    # let it through, and a rejected one the decision that stopped it.
    id: str = field(default_factory=lambda: f"rev_{uuid.uuid4().hex[:12]}")

    @property
    def confidence_delta(self) -> int:
        return self.revised_confidence - self.original_confidence

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["severity"] = self.severity.value
        return d

    def summary(self) -> str:
        arrow = (f"{self.original_confidence}→{self.revised_confidence}"
                 if self.confidence_delta else f"{self.revised_confidence}")
        return (f"{self.status.value} (confidence {arrow}, "
                f"worst finding {self.severity.value})")


@dataclass
class GatePolicy:
    """Configurable thresholds. Defaults are deliberately strict — the failure
    mode this gate exists to prevent is waving things through."""

    critical_rejects: bool = True
    high_findings_to_reject: int = 3          # this many HIGH forces REJECT
    high_confidence_penalty: int = 25         # per HIGH finding
    medium_confidence_penalty: int = 8
    low_confidence_penalty: int = 2
    min_confidence_to_survive: int = 15       # below this, the idea is not worth publishing
    research_severity: Severity = Severity.HIGH
    max_confidence_after_high: int = 55       # a HIGH finding caps confidence

    def __post_init__(self):
        if self.high_findings_to_reject < 1:
            raise ValueError("high_findings_to_reject must be >= 1")


class ReviewGate:
    """Deterministic policy engine turning findings into a binding decision."""

    def __init__(self, policy: GatePolicy | None = None):
        self.policy = policy or GatePolicy()

    def review(
        self,
        findings: list[Finding],
        original_confidence: int,
        iteration: int = 0,
        max_iterations: int = 2,
        evidence_already_requested: bool = False,
    ) -> ReviewDecision:
        p = self.policy
        conf = int(max(0, min(100, original_confidence)))

        if not findings:
            return ReviewDecision(
                status=ReviewStatus.APPROVE,
                original_confidence=conf,
                revised_confidence=conf,
                severity=Severity.INFO,
                reasons=["no objections raised"],
                accepted_claims=["all"],
                iteration=iteration,
            )

        worst = max(findings, key=lambda f: f.severity.rank)
        criticals = [f for f in findings if f.severity is Severity.CRITICAL]
        highs = [f for f in findings if f.severity is Severity.HIGH]
        mediums = [f for f in findings if f.severity is Severity.MEDIUM]
        lows = [f for f in findings if f.severity in (Severity.LOW, Severity.INFO)]

        reasons = [f"{f.severity.value}: {f.objection}" for f in findings]

        # --- 1. CRITICAL is non-negotiable ---------------------------------
        if criticals and p.critical_rejects:
            return ReviewDecision(
                status=ReviewStatus.REJECT,
                original_confidence=conf,
                revised_confidence=0,
                severity=Severity.CRITICAL,
                reasons=reasons,
                rejected_claims=[f.invalidates_claim or f.objection for f in criticals],
                required_followups=[f.requires_evidence for f in criticals
                                    if f.requires_evidence],
                iteration=iteration,
            )

        # --- 2. enough HIGH findings and the idea is not salvageable -------
        if len(highs) >= p.high_findings_to_reject:
            return ReviewDecision(
                status=ReviewStatus.REJECT,
                original_confidence=conf,
                revised_confidence=0,
                severity=Severity.HIGH,
                reasons=reasons,
                rejected_claims=[f.invalidates_claim or f.objection for f in highs],
                iteration=iteration,
            )

        # --- 3. apply the confidence penalty -------------------------------
        penalty = (len(highs) * p.high_confidence_penalty
                   + len(mediums) * p.medium_confidence_penalty
                   + len(lows) * p.low_confidence_penalty)
        revised = max(0, conf - penalty)
        if highs:
            revised = min(revised, p.max_confidence_after_high)

        # An explicit cap from a finding always wins over the computed value.
        modifications: dict = {}
        for f in findings:
            if not f.suggested_modification:
                continue
            modifications.update(f.suggested_modification)
            cap = f.suggested_modification.get("confidence_cap")
            if cap is not None:
                revised = min(revised, int(cap))

        # --- 4. below the floor, publishing it would mislead ---------------
        if revised < p.min_confidence_to_survive:
            return ReviewDecision(
                status=ReviewStatus.REJECT,
                original_confidence=conf,
                revised_confidence=revised,
                severity=worst.severity,
                reasons=reasons + [
                    f"confidence fell to {revised}, below the {p.min_confidence_to_survive} "
                    f"floor for publication"
                ],
                rejected_claims=[f.invalidates_claim or f.objection
                                 for f in highs + criticals],
                modifications=modifications,
                iteration=iteration,
            )

        # --- 5. a severe finding with a named test earns another pass ------
        needs_research = [f for f in findings
                          if f.severity.rank >= p.research_severity.rank
                          and f.requires_evidence]
        if needs_research and iteration < max_iterations and not evidence_already_requested:
            return ReviewDecision(
                status=ReviewStatus.REQUEST_MORE_RESEARCH,
                original_confidence=conf,
                revised_confidence=revised,
                severity=worst.severity,
                reasons=reasons,
                required_followups=[f.requires_evidence for f in needs_research],
                accepted_claims=[f.objection for f in lows],
                modifications=modifications,
                iteration=iteration,
            )

        # --- 6. structural changes requested -------------------------------
        if modifications:
            return ReviewDecision(
                status=ReviewStatus.MODIFY,
                original_confidence=conf,
                revised_confidence=revised,
                severity=worst.severity,
                reasons=reasons,
                modifications=modifications,
                required_followups=[f.requires_evidence for f in findings
                                    if f.requires_evidence],
                iteration=iteration,
            )

        # --- 7. survives, but not unscathed --------------------------------
        if penalty > 0:
            return ReviewDecision(
                status=ReviewStatus.APPROVE_WITH_REDUCED_CONFIDENCE,
                original_confidence=conf,
                revised_confidence=revised,
                severity=worst.severity,
                reasons=reasons,
                accepted_claims=[f.objection for f in findings],
                iteration=iteration,
            )

        return ReviewDecision(
            status=ReviewStatus.APPROVE,
            original_confidence=conf,
            revised_confidence=revised,
            severity=worst.severity,
            reasons=reasons,
            iteration=iteration,
        )

    # ------------------------------------------------------------------

    def apply(self, idea: dict, decision: ReviewDecision) -> dict | None:
        """Apply a decision to an idea. Returns None when the idea is rejected.

        This is where the decision actually bites — the caller cannot receive a
        modified idea without the modification having been made.
        """
        if decision.status is ReviewStatus.REJECT:
            return None

        out = dict(idea)
        out["confidence"] = decision.revised_confidence
        out["review_status"] = decision.status.value
        out["review_severity"] = decision.severity.value
        out["review_reasons"] = decision.reasons
        out["original_confidence"] = decision.original_confidence

        m = decision.modifications
        if "target" in m:
            out["target"] = float(m["target"])
        if "stop" in m:
            out["stop"] = float(m["stop"])
        if "size_multiplier" in m:
            out["size_multiplier"] = float(m["size_multiplier"])
        if "downgrade_to_watch" in m and m["downgrade_to_watch"]:
            out["downgraded"] = True
        if decision.required_followups:
            out["open_questions"] = list(decision.required_followups)

        # An idea that survived a HIGH finding must say so where a reader will
        # see it, not only in an audit table.
        if decision.severity.rank >= Severity.HIGH.rank:
            note = (f"Red team raised a {decision.severity.value} objection; "
                    f"confidence cut from {decision.original_confidence} to "
                    f"{decision.revised_confidence}.")
            out["invalidation"] = (out.get("invalidation", "") + " " + note).strip()
        return out
