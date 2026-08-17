"""The publication authority.

There is exactly one collection of recommendations that may be published, and
it lives here. Every downstream consumer — the Markdown report, the HTML
report, JSON output, the webhook, the database, the monitor, the scorer, the
API — reads a `PublicationSet` and nothing else.

This module exists because of a specific defect. In 1.x, and in 2.0.0 before
this change, the review gate ran and correctly rejected candidates, and then
the orchestrator rebuilt the published output straight from the pre-review
playbook. The gate worked; nothing downstream was listening. Correctness that
is not on the control path is decoration.

The invariant, enforced by `assert_no_leak()` and by construction:

    REJECTED  ==>  not published, not persisted as active, not monitored.

A rejected candidate keeps its full audit trail. It is stored as review
history, which is a different thing from a recommendation.

The legacy `ideas` dict is retained for the report renderer and the webhook,
but it is a *derived view*: `ideas_view()` reads the approved recommendations
and overwrites every decision-bearing field from the canonical object. A
presentation payload can carry a strike and an expiration; it may never carry
its own idea of the confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from .recommend.engine import Recommendation

log = logging.getLogger("marketswarm.publication")


class RecordStatus(str, Enum):
    """Lifecycle of a candidate. Only APPROVED and MODIFIED are ever active."""

    CANDIDATE = "CANDIDATE"           # built, not yet reviewed
    APPROVED = "APPROVED"             # survived review unchanged
    MODIFIED = "MODIFIED"             # survived review with changes applied
    REJECTED = "REJECTED"             # killed by review — audit only
    SUPPRESSED = "SUPPRESSED"         # survived review, blocked by degradation
    EXPIRED = "EXPIRED"               # horizon passed without resolution
    INVALIDATED = "INVALIDATED"       # invalidation condition hit intraday
    RESOLVED = "RESOLVED"             # outcome scored

    @property
    def is_active(self) -> bool:
        return self in (RecordStatus.APPROVED, RecordStatus.MODIFIED)


@dataclass
class RejectedCandidate:
    """A candidate the review gate killed, kept for audit and learning."""

    subject: str
    candidate_id: str
    reason: str
    findings: list[str] = field(default_factory=list)
    audit: list[dict] = field(default_factory=list)
    original_confidence: int = 0
    status: RecordStatus = RecordStatus.REJECTED

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "findings": list(self.findings),
            "original_confidence": self.original_confidence,
            "status": self.status.value,
            "audit": self.audit,
        }


class PublicationError(RuntimeError):
    """Raised when a rejected candidate reaches an active-output path."""


@dataclass
class PublicationSet:
    """The authoritative final recommendation set for one run.

    `approved` holds recommendations that survived review. `rejected` holds the
    audit trail of those that did not. Nothing else may reach a reader.
    """

    approved: list[Recommendation] = field(default_factory=list)
    rejected: list[RejectedCandidate] = field(default_factory=list)
    suppressed: bool = False
    suppression_reason: str | None = None
    mode: str = "dynamic"
    review_iterations: int = 0

    # ---------- construction ----------

    @classmethod
    def suppressed_set(cls, reason: str, mode: str = "dynamic",
                       rejected: list[RejectedCandidate] | None = None) -> "PublicationSet":
        """No recommendations may be published, and no legacy output stands in.

        This is the safe fallback. A failure in the 2.0 control layer produces
        an empty publication set with a stated reason — never a pre-review
        idea that nothing reviewed.
        """
        return cls(approved=[], rejected=list(rejected or []), suppressed=True,
                   suppression_reason=reason, mode=mode)

    # ---------- reading ----------

    def active(self) -> list[Recommendation]:
        """Recommendations that may be shown, stored active, and monitored."""
        if self.suppressed:
            return []
        return list(self.approved)

    def actionable(self) -> list[Recommendation]:
        return [r for r in self.active() if r.actionable]

    def active_subjects(self) -> set[str]:
        return {r.subject for r in self.active()}

    def rejected_subjects(self) -> set[str]:
        return {r.subject for r in self.rejected}

    def may_publish_index_call(self) -> bool:
        """Whether the session-level directional call may be filed.

        The index call is a directional prediction like any other, and it feeds
        the global calibrator. If the review gate rejected every candidate, the
        system has just said its read is unsupported — filing a market call
        anyway would contradict its own review and quietly keep scoring a
        forecast nothing approved.
        """
        if self.suppressed:
            return False
        if self.rejected and not self.approved:
            return False
        return True

    def by_id(self, rec_id: str) -> Recommendation | None:
        for r in self.active():
            if r.id == rec_id:
                return r
        return None

    # ---------- the legacy view ----------

    def ideas_view(self) -> dict:
        """The 1.x `{calls, puts, stocks}` shape, derived from final decisions.

        Only actionable recommendations produce an idea. Everything else — a
        WATCH, a NO_EDGE, an INSUFFICIENT_EVIDENCE — is a real output of the
        system and is rendered from `active()`, not smuggled into a section
        headed "best call options".
        """
        view: dict[str, list[dict]] = {"calls": [], "puts": [], "stocks": []}
        if self.suppressed:
            return view
        for rec in self.approved:
            if not rec.actionable:
                continue
            bucket = {"call": "calls", "put": "puts", "stock": "stocks"}.get(
                rec.source_kind or "stock")
            if bucket is None:
                continue
            view[bucket].append(_payload_from(rec))
        return view

    # ---------- the invariant ----------

    def assert_no_leak(self) -> None:
        """Fail loudly if a rejected candidate appears in any active output.

        Called on every production run. The cost is a set intersection; the
        thing it prevents is the entire reason 2.0 exists.
        """
        rejected = {r.candidate_id for r in self.rejected}
        for rec in self.active():
            if rec.id in rejected:
                raise PublicationError(
                    f"rejected candidate {rec.id} ({rec.subject}) reached the active set")

        ids = [r.id for r in self.approved]
        if len(ids) != len(set(ids)):
            raise PublicationError("duplicate recommendation ids in the active set")

        for kind, ideas in self.ideas_view().items():
            for idea in ideas:
                rec = self.by_id(idea.get("recommendation_id", ""))
                if rec is None:
                    raise PublicationError(
                        f"{kind} idea {idea.get('symbol')} has no backing recommendation")
                if idea["confidence"] != rec.confidence:
                    raise PublicationError(
                        f"{rec.subject} legacy view shows confidence "
                        f"{idea['confidence']} but the recommendation says {rec.confidence}")

    # ---------- reporting ----------

    def summary(self) -> dict:
        by_type: dict[str, int] = {}
        by_conviction: dict[str, int] = {}
        for r in self.active():
            by_type[r.rec_type.value] = by_type.get(r.rec_type.value, 0) + 1
            by_conviction[r.conviction.value] = by_conviction.get(r.conviction.value, 0) + 1
        return {
            "mode": self.mode,
            "approved": len(self.approved),
            "active": len(self.active()),
            "actionable": len(self.actionable()),
            "rejected": len(self.rejected),
            "modified": sum(1 for r in self.active() if r.revision_count > 0),
            "suppressed": self.suppressed,
            "suppression_reason": self.suppression_reason,
            "by_type": by_type,
            "by_conviction": by_conviction,
        }


def _payload_from(rec: Recommendation) -> dict:
    """Render one recommendation into the legacy idea shape.

    The presentation payload supplies the things the canonical object has no
    opinion about — option strike, expiration, premium zones, liquidity notes.
    Every field that encodes a *decision* is overwritten from the
    recommendation, so a stale pre-review number cannot survive here. That
    overwrite is the whole point of this function.
    """
    payload = dict(rec.source_payload or {})

    payload.update({
        "symbol": rec.subject,
        "direction": rec.direction,
        "probability": rec.forecast_probability,
        "confidence": rec.confidence,
        "expected_r": rec.expected_r,
        "entry": rec.entry,
        "target": rec.target,
        "stop": rec.stop,
        "rationale": rec.rationale,
        "invalidation": "; ".join(rec.invalidation) or payload.get("invalidation", ""),
        "recommendation_id": rec.id,
        "rec_type": rec.rec_type.value,
        "conviction": rec.conviction.value,
        "review_status": rec.review_status,
        "original_confidence": rec.original_confidence,
        "revision_count": rec.revision_count,
    })
    if rec.supporting_evidence:
        payload["evidence"] = list(rec.supporting_evidence[:6])
    return payload
