"""The Research Scientist — bounded, propose-only.

Reads the system's own failures and proposes experiments to fix them. It has
exactly one hard constraint, enforced structurally rather than by instruction:

    it can propose and evaluate; it can NEVER promote.

`ResearchScientist` has no reference to the champion configuration and no
method that writes production state. Promotion lives on `ExperimentLab`, behind
gates and human approval. A scientist that could deploy its own hypotheses
would be an unsupervised release process.

Hypothesis generation is rule-driven first — the failure taxonomy already tells
you what to try when the same mistake recurs thirty times — with an optional
LLM pass for the cases where the pattern is not in the taxonomy.
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from dataclasses import asdict, dataclass, field

from ..memory.institutional import InstitutionalMemory, MistakeTaxonomy
from .lab import Challenger

log = logging.getLogger("marketswarm.experiments.scientist")


@dataclass
class ResearchProposal:
    title: str
    hypothesis: str
    rationale: str
    config_patch: dict
    kind: str
    expected_effect: str
    evidence: list[str] = field(default_factory=list)
    priority: int = 3                     # 1 highest
    created_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )

    def to_challenger(self) -> Challenger:
        return Challenger(
            name=self.title,
            hypothesis=self.hypothesis,
            config_patch=dict(self.config_patch),
            kind=self.kind,
            proposed_by="research_scientist",
        )

    def to_dict(self) -> dict:
        return asdict(self)


# Failure taxonomy → the change most likely to address it. Deterministic,
# auditable, and derived from what the system actually got wrong.
TAXONOMY_REMEDIES: dict[MistakeTaxonomy, dict] = {
    MistakeTaxonomy.OVERWEIGHTED_CORRELATED: {
        "title": "raise the correlation haircut",
        "hypothesis": "Confidence is inflated because correlated evidence is "
                      "counted as independent; a higher assumed correlation "
                      "should improve calibration.",
        "patch": {"signal_correlation": 0.55},
        "kind": "scoring",
        "effect": "lower confidence on correlated stacks; Brier should improve",
    },
    MistakeTaxonomy.OVERCONFIDENT: {
        "title": "tighten the confidence-to-action threshold",
        "hypothesis": "Acting at lower confidence produces negative expectancy; "
                      "raising the bar should improve mean R at the cost of volume.",
        "patch": {"min_confidence_for_action": 55},
        "kind": "threshold",
        "effect": "fewer trades, higher average quality",
    },
    MistakeTaxonomy.COST_UNDERESTIMATED: {
        "title": "raise the assumed friction",
        "hypothesis": "Realised costs exceed the modelled 6% of R; a higher "
                      "assumption should stop marginal ideas being published.",
        "patch": {"friction_r": 0.10},
        "kind": "scoring",
        "effect": "marginal ideas drop below the EV gate",
    },
    MistakeTaxonomy.REGIME_MISREAD: {
        "title": "require regime confidence before acting",
        "hypothesis": "Conditional base rates are applied even when the regime "
                      "label is uncertain; gating on regime confidence should help.",
        "patch": {"min_regime_confidence": 0.6},
        "kind": "routing",
        "effect": "conviction capped when the regime is unclear",
    },
    MistakeTaxonomy.STALE_INFORMATION: {
        "title": "shorten the evidence half-life",
        "hypothesis": "Stale evidence retains too much weight; faster timeliness "
                      "decay should reduce errors driven by old information.",
        "patch": {"evidence_half_life_hours": 6.0},
        "kind": "scoring",
        "effect": "older headlines contribute less",
    },
    MistakeTaxonomy.RED_TEAM_IGNORED: {
        "title": "increase the red-team confidence penalty",
        "hypothesis": "Ideas surviving high-severity objections underperform; a "
                      "larger penalty should remove them.",
        "patch": {"high_confidence_penalty": 40},
        "kind": "scoring",
        "effect": "more rejections, better surviving population",
    },
    MistakeTaxonomy.BAD_LEVELS: {
        "title": "widen stops relative to ATR",
        "hypothesis": "Stops are inside normal noise and are hit before the "
                      "thesis resolves; a wider stop should raise the hit rate.",
        "patch": {"stop_atr": 0.9},
        "kind": "features",
        "effect": "fewer noise stop-outs, larger per-trade risk",
    },
}

MIN_OCCURRENCES_TO_PROPOSE = 5


class ResearchScientist:
    """Analyses failures, proposes experiments. Cannot deploy anything."""

    def __init__(self, conn: sqlite3.Connection, router=None,
                 max_proposals_per_run: int = 3):
        self.conn = conn
        self.memory = InstitutionalMemory(conn)
        self.router = router
        self.max_proposals_per_run = max_proposals_per_run

    # ---------- analysis ----------

    def analyse_failures(self) -> dict:
        freq = self.memory.mistake_frequency()
        total = sum(freq.values())
        recurring = {k: v for k, v in freq.items() if v >= MIN_OCCURRENCES_TO_PROPOSE}

        avoidable = self.conn.execute(
            "SELECT COUNT(*) FROM mistakes WHERE genuinely_unpredictable=0").fetchone()[0]
        unavoidable = self.conn.execute(
            "SELECT COUNT(*) FROM mistakes WHERE genuinely_unpredictable=1").fetchone()[0]
        rt_ignored = self.conn.execute(
            "SELECT COUNT(*) FROM mistakes WHERE red_team_ignored=1").fetchone()[0]

        return {
            "total_mistakes": total,
            "frequency": freq,
            "recurring": recurring,
            "avoidable": avoidable,
            "unavoidable": unavoidable,
            "red_team_ignored": rt_ignored,
            "note": ("recurring avoidable failures are process defects and are "
                     "worth an experiment; unavoidable ones are not"),
        }

    def propose(self) -> list[ResearchProposal]:
        """Generate experiment proposals from measured failure patterns.

        Only recurring, avoidable failures produce proposals. One unlucky call
        is not a reason to change the system, and treating it as one is how a
        model learns to chase noise.
        """
        analysis = self.analyse_failures()
        proposals: list[ResearchProposal] = []

        for tax_value, count in analysis["recurring"].items():
            try:
                tax = MistakeTaxonomy(tax_value)
            except ValueError:
                continue
            if tax is MistakeTaxonomy.UNPREDICTABLE:
                continue
            remedy = TAXONOMY_REMEDIES.get(tax)
            if not remedy:
                continue

            examples = self.memory.mistakes_like(taxonomy=tax, limit=3)
            proposals.append(ResearchProposal(
                title=remedy["title"],
                hypothesis=remedy["hypothesis"],
                rationale=(f"'{tax.value}' has occurred {count} times, "
                           f"{count / max(analysis['total_mistakes'], 1):.0%} of all "
                           f"recorded failures. That is a process defect, not variance."),
                config_patch=dict(remedy["patch"]),
                kind=remedy["kind"],
                expected_effect=remedy["effect"],
                evidence=[e.get("lesson", "")[:160] for e in examples],
                priority=1 if count >= 10 else 2,
            ))

        proposals.sort(key=lambda p: (p.priority, -len(p.evidence)))
        return proposals[: self.max_proposals_per_run]

    def propose_from_calibration(self, calibration: dict) -> list[ResearchProposal]:
        """Propose from measured calibration rather than from individual errors."""
        out: list[ResearchProposal] = []
        n = calibration.get("n", 0)
        if n < 50:
            return out

        gap = (calibration.get("mean_forecast", 0.5)
               - calibration.get("hit_rate", calibration.get("observed_rate", 0.5)))
        if gap > 0.08:
            out.append(ResearchProposal(
                title="shrink published probabilities",
                hypothesis=(f"Forecasts run {gap:.0%} above observed frequency across "
                            f"{n} resolved calls; stronger recalibration should close it."),
                rationale="measured, systematic overconfidence",
                config_patch={"calibration_shrinkage": 0.75},
                kind="scoring",
                expected_effect="mean forecast falls toward the observed rate",
                evidence=[f"mean forecast {calibration.get('mean_forecast'):.1%} vs "
                          f"observed {calibration.get('hit_rate', 0):.1%} over {n}"],
                priority=1,
            ))

        if calibration.get("resolution", 1.0) < 0.005:
            out.append(ResearchProposal(
                title="drop non-contributing agents from routing",
                hypothesis=("Resolution is near zero, so the model barely separates "
                            "winners from losers; removing agents with negative measured "
                            "contribution should not hurt and will cut cost."),
                rationale="near-zero discrimination in the scored history",
                config_patch={"prune_negative_contributors": True},
                kind="routing",
                expected_effect="same or better skill at lower cost",
                evidence=[f"resolution {calibration.get('resolution'):.5f}"],
                priority=2,
            ))
        return out

    # ---------- registration (still not deployment) ----------

    def register_proposals(self, lab, proposals: list[ResearchProposal]) -> list[str]:
        """File proposals as challengers. They are inert until evaluated and
        promoted, and this method cannot promote."""
        ids = []
        for p in proposals:
            ids.append(lab.register(p.to_challenger()))
            self.conn.execute(
                """INSERT INTO memories
                   (id, created_at, category, subject, title, body, confidence,
                    evidence_n, source, active)
                   VALUES (?,?,?,?,?,?,?,?,?,1)""",
                (f"mem_prop_{ids[-1][-8:]}", p.created_at, "experiment", None,
                 f"proposed: {p.title}", p.hypothesis, 0.5, len(p.evidence),
                 "research_scientist"))
        self.conn.commit()
        return ids

    # Deliberately absent: promote(), deploy(), apply_to_production().
    # Promotion is ExperimentLab.promote, gated and human-approved.

    def report(self) -> str:
        analysis = self.analyse_failures()
        proposals = self.propose()
        lines = [
            "Research Scientist report",
            f"  mistakes recorded: {analysis['total_mistakes']} "
            f"({analysis['avoidable']} avoidable, {analysis['unavoidable']} not)",
        ]
        if analysis["recurring"]:
            lines.append("  recurring failure modes:")
            for k, v in analysis["recurring"].items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append("  no failure mode has recurred often enough to act on")
        if proposals:
            lines.append(f"  {len(proposals)} experiment(s) proposed:")
            for p in proposals:
                lines.append(f"    [{p.priority}] {p.title} — {p.expected_effect}")
        lines.append("  note: proposals are inert until they pass the promotion "
                     "gates and a human approves them")
        return "\n".join(lines)
