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
    # Every adversarial round this candidate went through, oldest first.
    # Round 1 is never overwritten by round 2 — after-action review needs both.
    review_rounds: list[dict] = field(default_factory=list)
    source_kind: str | None = None
    source_payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "findings": list(self.findings),
            "original_confidence": self.original_confidence,
            "status": self.status.value,
            "review_rounds": list(self.review_rounds),
            "source_kind": self.source_kind,
            "source_payload": dict(self.source_payload),
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

    def screening_view(self, playbook_data: dict | None, slots: int = 3) -> dict:
        """Exactly ``slots`` call and put rows, each carrying a safety status.

        This is an audit/watch surface, not the active recommendation surface.
        Rejected candidates may be displayed here only with an unmistakable
        ``REJECTED`` label; they never enter :meth:`ideas_view`, predictions,
        monitoring, or scoring. A failed review layer yields withheld
        placeholders instead of leaking unreviewed candidates.
        """
        slots = max(1, min(int(slots), 10))
        out: dict[str, list[dict]] = {"calls": [], "puts": []}
        if self.suppressed:
            reason = self.suppression_reason or "publication suppressed"
            for key, kind in (("calls", "call"), ("puts", "put")):
                out[key] = [_empty_screen_slot(kind, "WITHHELD", reason, n + 1)
                            for n in range(slots)]
            return out

        active = {(r.subject, r.source_kind): r for r in self.active()
                  if r.source_kind in ("call", "put")}
        rejected = {(r.subject, r.source_kind): r for r in self.rejected
                    if r.source_kind in ("call", "put")}
        playbook_data = playbook_data or {}

        for key, kind in (("calls", "call"), ("puts", "put")):
            rows: list[dict] = []
            for candidate in (playbook_data.get(key, []) or [])[:slots]:
                symbol = str(candidate.get("symbol") or "?")
                rec = active.get((symbol, kind))
                rej = rejected.get((symbol, kind))
                if rec is not None:
                    row = _payload_from(rec)
                    row["candidate_direction"] = "long" if kind == "call" else "short"
                    if rec.actionable:
                        row["screen_status"] = "QUALIFIED"
                        row["screen_reason"] = "cleared the recommendation and review gates"
                    else:
                        row["screen_status"] = "WATCH ONLY"
                        notes = rec.uncertainty_notes or [rec.rec_type.value]
                        row["screen_reason"] = "; ".join(str(n) for n in notes[:2])
                elif rej is not None:
                    # These numbers describe what was screened, not what was
                    # approved. The status is part of the row by construction.
                    row = dict(candidate)
                    row.update({
                        "candidate_direction": "long" if kind == "call" else "short",
                        "screen_status": "REJECTED",
                        "screen_reason": rej.reason,
                        "candidate_id": rej.candidate_id,
                    })
                else:
                    row = dict(candidate)
                    row.update({
                        "candidate_direction": "long" if kind == "call" else "short",
                        "screen_status": "WITHHELD",
                        "screen_reason": "candidate did not complete the review path",
                    })
                rows.append(row)

            while len(rows) < slots:
                rows.append(_empty_screen_slot(
                    kind,
                    "DATA UNAVAILABLE",
                    "no valid liquid contract and sane bracket were available for this slot",
                    len(rows) + 1,
                ))
            out[key] = rows
        return out

    # ---------- the invariant ----------

    def assert_no_leak(self) -> None:
        """Fail loudly if a rejected candidate appears in any active output.

        The identity that matters is the **candidate lineage**, not the
        recommendation id. In production a recommendation carries a fresh
        `rec_...` id and a `candidate_id` pointing back at the `cand_...` it
        was built from, so comparing `rec.id` against the rejected candidate
        ids compares two namespaces that never collide — a check that can
        never fire is not a check.

        Called on every production run. The cost is a set intersection; the
        thing it prevents is the entire reason 2.0 exists.
        """
        rejected = {r.candidate_id for r in self.rejected if r.candidate_id}
        for rec in self.active():
            # Match on lineage first, then on the raw id, so a recommendation
            # that never recorded a candidate_id is still caught.
            for identity in (rec.candidate_id, rec.id):
                if identity and identity in rejected:
                    raise PublicationError(
                        f"rejected candidate {identity} reached the active set as "
                        f"recommendation {rec.id} ({rec.subject})")

            # A revision descends from a rejected parent only if that parent
            # was rejected — which would mean the rejection was overturned
            # without a new review.
            if rec.revision_parent_id and rec.revision_parent_id in rejected:
                raise PublicationError(
                    f"recommendation {rec.id} ({rec.subject}) descends from "
                    f"rejected candidate {rec.revision_parent_id}")

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


def _empty_screen_slot(kind: str, status: str, reason: str, slot: int) -> dict:
    return {
        "symbol": None,
        "kind": kind,
        "slot": slot,
        "candidate_direction": "long" if kind == "call" else "short",
        "screen_status": status,
        "screen_reason": reason,
    }
