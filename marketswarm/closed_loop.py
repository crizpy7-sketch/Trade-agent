"""The closed learning loop.

Everything downstream of an outcome, in one place, running automatically:

    resolved outcome
        → after-action review        (what did we get wrong, and was it avoidable?)
        → contribution analysis      (which agent actually added value here?)
        → mistake + institutional memory
        → calibration
        → research proposals         (only when a failure mode recurs)

Before this module, all five components existed and every one of them had to
be invoked by hand from the CLI. `marketswarm score` ran the 1.x
`LearningEngine` and stopped. The learning architecture was real and inert.

Two design rules worth stating, because both are load-bearing:

**Thresholds, not enthusiasm.** The Research Scientist is only asked for
proposals when a failure mode has actually recurred. A system that proposes an
architecture change after every losing trade is fitting noise with extra steps.

**Propose, never promote.** This module calls `ResearchScientist.propose()` and
registers challengers. It has no code path to promotion, because promotion is
`ExperimentLab.promote()` behind human approval. Automating the closed loop
must not automate deployment.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from dataclasses import dataclass, field

from .after_action import AfterActionReviewer
from .memory.contribution import ContextKey, ContributionTracker
from .memory.institutional import InstitutionalMemory

log = logging.getLogger("marketswarm.closed_loop")

# A failure mode must recur this many times before it is worth an experiment.
# Below it, the honest reading is variance.
PROPOSAL_THRESHOLD = 5

# The scientist is not consulted until there is enough resolved history for
# "recurring" to mean anything.
MIN_RESOLVED_FOR_PROPOSALS = 30


@dataclass
class LearningCycleResult:
    resolved_reviewed: int = 0
    reviews_persisted: int = 0
    contributions_updated: int = 0
    weights_changed: dict[str, tuple[float, float]] = field(default_factory=dict)
    mistakes_recorded: int = 0
    memories_written: int = 0
    memories_pruned: int = 0
    proposals: list[str] = field(default_factory=list)
    skipped_reason: str | None = None

    def render(self) -> str:
        if self.skipped_reason:
            return f"Learning cycle: {self.skipped_reason}"
        lines = [
            f"Learning cycle: reviewed {self.resolved_reviewed} resolved recommendation(s)",
            f"  after-action reviews stored : {self.reviews_persisted}",
            f"  contextual weights updated  : {self.contributions_updated}",
            f"  mistakes recorded           : {self.mistakes_recorded}",
            f"  memories written / pruned   : {self.memories_written} / {self.memories_pruned}",
        ]
        for agent, (old, new) in sorted(self.weights_changed.items()):
            lines.append(f"    {agent}: {old:.2f} → {new:.2f}")
        if self.proposals:
            lines.append(f"  experiments proposed        : {len(self.proposals)}")
            for p in self.proposals:
                lines.append(f"    · {p}")
            lines.append("  (proposals are inert until they pass the gates and a human approves)")
        return "\n".join(lines)


class ClosedLoop:
    """Runs the post-resolution half of the architecture."""

    def __init__(self, conn: sqlite3.Connection, router=None):
        self.conn = conn
        self.reviewer = AfterActionReviewer(conn)
        self.tracker = ContributionTracker(conn)
        self.memory = InstitutionalMemory(conn)
        self.router = router

    # ------------------------------------------------------------------

    def run(self, since_days: int = 30, propose: bool = True) -> LearningCycleResult:
        out = LearningCycleResult()
        rows = self._newly_resolved(since_days)
        if not rows:
            out.skipped_reason = "no newly resolved predictions to learn from"
            return out

        weights_before = self._weight_snapshot()

        # --- 1. after-action review, one per resolved recommendation ---
        for row in rows:
            try:
                aar = self._review_row(row)
            except Exception as exc:  # noqa: BLE001 — one bad row must not stop learning
                log.warning("after-action review failed for %s: %s", row["id"], exc)
                continue
            out.resolved_reviewed += 1
            persisted = self.reviewer.persist(aar)
            out.reviews_persisted += 1
            out.mistakes_recorded += int(bool(persisted.get("mistake_id")))
            out.memories_written += int(bool(aar.should_become_memory or aar.succeeded))

        # --- 2. contribution analysis over the same resolved population ---
        results = self.tracker.update_from_resolved(rows)
        out.contributions_updated = len(results)

        weights_after = self._weight_snapshot()
        for key, new in weights_after.items():
            old = weights_before.get(key)
            if old is None or abs(old - new) > 1e-9:
                out.weights_changed[key] = (old if old is not None else 1.0, new)

        # --- 3. housekeeping: expire memories past their TTL ---
        out.memories_pruned = sum(self.memory.prune().values())

        # --- 4. proposals, only when a failure mode has actually recurred ---
        if propose:
            out.proposals = self._maybe_propose()

        self.conn.commit()
        return out

    # ------------------------------------------------------------------

    def _newly_resolved(self, since_days: int) -> list[sqlite3.Row]:
        """Resolved predictions that have not yet been through a review.

        Joined against `after_action_reviews` so re-running the cycle is
        idempotent — learning twice from one outcome is double-counting, and
        double-counting outcomes is how a system convinces itself it has more
        evidence than it does.
        """
        cutoff = (dt.date.today() - dt.timedelta(days=since_days)).isoformat()
        self.conn.row_factory = sqlite3.Row
        return list(self.conn.execute(
            """SELECT p.* FROM predictions p
               WHERE p.resolved = 1
                 AND p.run_date >= ?
                 AND NOT EXISTS (
                     SELECT 1 FROM after_action_reviews a
                     WHERE a.recommendation_id = CAST(p.id AS TEXT)
                 )
               ORDER BY p.run_date""",
            (cutoff,)))

    def _review_row(self, row: sqlite3.Row):
        keys = row.keys()
        contributions = {}
        if "contributing_agents" in keys and row["contributing_agents"]:
            try:
                contributions = json.loads(row["contributing_agents"])
            except (json.JSONDecodeError, TypeError):
                contributions = {}

        def num(field, default):
            # Columns are nullable and older rows predate several of them, so
            # every read has to survive a NULL rather than assume a value.
            if field not in keys or row[field] is None:
                return default
            return row[field]

        rec = {
            "id": str(row["id"]),
            "subject": row["symbol"],
            "direction": num("direction", "?"),
            "probability": float(num("probability", 0.5)),
            "confidence": int(num("confidence", 50)),
        }
        realized = float(num("realized_r", 0.0))
        outcome = int(num("outcome", 0))
        return self.reviewer.review(rec, realized_r=realized, outcome=outcome,
                                    contributions=contributions)

    def _weight_snapshot(self) -> dict[str, float]:
        return {
            f"{r[0]}@{r[1]}": float(r[2])
            for r in self.conn.execute(
                "SELECT agent, context_key, weight FROM agent_context_scores")
        }

    def _maybe_propose(self) -> list[str]:
        """Ask the Research Scientist only when the evidence justifies it."""
        n_resolved = self.conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE resolved = 1").fetchone()[0]
        if n_resolved < MIN_RESOLVED_FOR_PROPOSALS:
            return []

        recurring = self.conn.execute(
            """SELECT taxonomy, COUNT(*) c FROM mistakes
               WHERE genuinely_unpredictable = 0
               GROUP BY taxonomy HAVING c >= ?""",
            (PROPOSAL_THRESHOLD,)).fetchall()
        if not recurring:
            return []

        from .experiments.lab import ExperimentLab
        from .experiments.scientist import ResearchScientist

        scientist = ResearchScientist(self.conn, router=self.router)
        proposals = scientist.propose()
        if not proposals:
            return []

        lab = ExperimentLab(self.conn)
        scientist.register_proposals(lab, proposals)
        # Registered, not promoted. `ResearchScientist` has no promote method
        # and this module deliberately does not call `lab.promote`.
        return [p.title for p in proposals]


def context_for(regime: str, event_type: str = "", setup: str = "") -> ContextKey:
    return ContextKey(regime=regime or "", event_type=event_type or "", setup=setup or "")
