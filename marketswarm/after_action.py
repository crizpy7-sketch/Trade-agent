"""After-action review.

Once a recommendation has resolved, this reconstructs what happened and turns
it into something the system can use: a lesson, a mistake record with a
taxonomy, and — when a failure mode keeps recurring — an experiment proposal.

The distinction it works hardest to preserve is between *being wrong* and
*being wrong for a fixable reason*. A shock nobody could forecast is not a
lesson; treating it as one teaches the system to fear the wrong things. Only
avoidable failures become memories that change future behaviour.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from dataclasses import dataclass, field

from .memory.institutional import (
    InstitutionalMemory,
    Memory,
    MemoryCategory,
    Mistake,
    MistakeTaxonomy,
)

log = logging.getLogger("marketswarm.after_action")


@dataclass
class AfterActionReview:
    recommendation_id: str
    subject: str
    expected: str
    actual: str
    got_right: list[str] = field(default_factory=list)
    got_wrong: list[str] = field(default_factory=list)
    agents_helped: list[str] = field(default_factory=list)
    agents_hurt: list[str] = field(default_factory=list)
    evidence_mattered: list[str] = field(default_factory=list)
    evidence_noise: list[str] = field(default_factory=list)
    red_team_performance: str = ""
    calibration_error: float | None = None
    lesson: str = ""
    taxonomy: list[MistakeTaxonomy] = field(default_factory=list)
    succeeded: bool = False
    should_become_memory: bool = False
    should_trigger_experiment: bool = False
    created_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )

    def render(self) -> str:
        lines = [
            f"AFTER-ACTION REVIEW — {self.subject} ({self.recommendation_id})",
            f"  WHAT WE EXPECTED    {self.expected}",
            f"  WHAT HAPPENED       {self.actual}",
        ]
        if self.calibration_error is not None:
            lines.append(f"  CALIBRATION ERROR   {self.calibration_error:+.2f}")
        for label, items in (("GOT RIGHT", self.got_right),
                             ("GOT WRONG", self.got_wrong),
                             ("AGENTS HELPED", self.agents_helped),
                             ("AGENTS HURT", self.agents_hurt),
                             ("EVIDENCE THAT MATTERED", self.evidence_mattered),
                             ("EVIDENCE THAT WAS NOISE", self.evidence_noise)):
            if items:
                lines.append(f"  {label}")
                lines.extend(f"    - {i}" for i in items[:5])
        if self.red_team_performance:
            lines.append(f"  RED TEAM            {self.red_team_performance}")
        lines.append(f"  LESSON              {self.lesson}")
        lines.append(f"  BECOMES MEMORY      {'yes' if self.should_become_memory else 'no'}")
        lines.append(f"  TRIGGERS EXPERIMENT {'yes' if self.should_trigger_experiment else 'no'}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "recommendation_id": self.recommendation_id,
            "subject": self.subject,
            "expected": self.expected, "actual": self.actual,
            "got_right": self.got_right, "got_wrong": self.got_wrong,
            "agents_helped": self.agents_helped, "agents_hurt": self.agents_hurt,
            "evidence_mattered": self.evidence_mattered,
            "evidence_noise": self.evidence_noise,
            "red_team_performance": self.red_team_performance,
            "calibration_error": self.calibration_error,
            "lesson": self.lesson,
            "taxonomy": [t.value for t in self.taxonomy],
            "should_become_memory": self.should_become_memory,
            "should_trigger_experiment": self.should_trigger_experiment,
        }


RECURRENCE_THRESHOLD = 5


class AfterActionReviewer:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.memory = InstitutionalMemory(conn)

    def review(
        self,
        recommendation: dict,
        realized_r: float,
        outcome: int,
        contributions: dict[str, float] | None = None,
        evidence: list[dict] | None = None,
        red_team_findings: list[dict] | None = None,
    ) -> AfterActionReview:
        subject = recommendation.get("subject") or recommendation.get("symbol", "?")
        forecast = float(recommendation.get("forecast_probability")
                         or recommendation.get("probability") or 0.5)
        confidence = int(recommendation.get("confidence", 50))
        direction = recommendation.get("direction", "?")
        correct = bool(outcome)

        aar = AfterActionReview(
            recommendation_id=recommendation.get("id", "unknown"),
            subject=subject,
            expected=(f"{direction} with P={forecast:.0%} at {confidence}/100 confidence"),
            actual=(f"{'target reached' if correct else 'did not work'}, "
                    f"realised {realized_r:+.2f}R"),
            calibration_error=forecast - (1.0 if correct else 0.0),
            succeeded=correct,
        )

        # --- credit and blame, from measured contribution ---
        for agent, lo in (contributions or {}).items():
            pushed_up = lo > 0
            if abs(lo) < 0.05:
                continue
            if pushed_up == correct:
                aar.agents_helped.append(f"{agent} ({lo:+.2f})")
            else:
                aar.agents_hurt.append(f"{agent} ({lo:+.2f})")

        for ev in (evidence or [])[:10]:
            claim = ev.get("claim", "")[:100]
            (aar.evidence_mattered if correct else aar.evidence_noise).append(claim)

        # --- red team scoring ---
        had_findings = bool(red_team_findings)
        if not correct and had_findings:
            aar.red_team_performance = (
                f"raised {len(red_team_findings)} objection(s) and the call still "
                f"failed — the objections were correct")
            aar.taxonomy.append(MistakeTaxonomy.RED_TEAM_IGNORED)
        elif not correct and not had_findings:
            aar.red_team_performance = "raised no objection and the call failed — a miss"
        elif correct and had_findings:
            aar.red_team_performance = (
                "objected to a call that worked — the cost of caution, not an error")
        else:
            aar.red_team_performance = "no objections, call worked"

        # --- classify the failure ---
        if correct:
            aar.got_right = [f"direction ({direction})",
                             f"forecast of {forecast:.0%} was borne out"]
            aar.lesson = (f"{subject}: the {direction} read worked, returning "
                          f"{realized_r:+.2f}R. One success is not evidence of skill.")
        else:
            aar.got_wrong = [f"direction ({direction}) was wrong"]
            aar.taxonomy.extend(self._classify(recommendation, forecast, confidence,
                                               realized_r, evidence or []))
            aar.got_wrong.extend(t.value.replace("_", " ") for t in aar.taxonomy)
            aar.lesson = self._lesson(subject, aar.taxonomy, realized_r, forecast)

        avoidable = MistakeTaxonomy.UNPREDICTABLE not in aar.taxonomy
        aar.should_become_memory = (not correct) and avoidable
        aar.should_trigger_experiment = (
            aar.should_become_memory and self._is_recurring(aar.taxonomy))
        return aar

    def _classify(self, rec: dict, forecast: float, confidence: int,
                  realized_r: float, evidence: list[dict]) -> list[MistakeTaxonomy]:
        tax: list[MistakeTaxonomy] = []

        if forecast >= 0.65 and confidence >= 60:
            tax.append(MistakeTaxonomy.OVERCONFIDENT)

        independent = float(rec.get("effective_independent_evidence") or 0)
        if independent and independent < 2.5:
            tax.append(MistakeTaxonomy.OVERWEIGHTED_CORRELATED)

        clusters = {e.get("cluster") for e in evidence if e.get("cluster")}
        if len(clusters) == 1 and len(evidence) >= 4:
            tax.append(MistakeTaxonomy.OVERWEIGHTED_CORRELATED)

        if rec.get("contradicting_evidence"):
            tax.append(MistakeTaxonomy.IGNORED_CONTRADICTION)

        stale = [e for e in evidence
                 if (e.get("timeliness") is not None and e["timeliness"] < 0.25)]
        if len(stale) >= 2:
            tax.append(MistakeTaxonomy.STALE_INFORMATION)

        if (rec.get("regime_confidence") is not None
                and float(rec["regime_confidence"]) < 0.5):
            tax.append(MistakeTaxonomy.REGIME_MISREAD)

        # A loss of exactly -1R is a clean stop-out, which points at level
        # placement rather than at the thesis.
        if abs(realized_r + 1.0) < 0.05:
            tax.append(MistakeTaxonomy.BAD_LEVELS)

        # A modest forecast that failed is not necessarily a process error —
        # a 55% call failing 45% of the time is the model working correctly.
        if not tax and forecast < 0.60:
            tax.append(MistakeTaxonomy.UNPREDICTABLE)

        return list(dict.fromkeys(tax))

    def _lesson(self, subject: str, tax: list[MistakeTaxonomy],
                realized_r: float, forecast: float) -> str:
        if not tax:
            return f"{subject} failed ({realized_r:+.2f}R) with no identifiable process error."
        if MistakeTaxonomy.UNPREDICTABLE in tax:
            return (f"{subject} failed ({realized_r:+.2f}R) on a {forecast:.0%} forecast. "
                    f"Within the expected failure rate — no change warranted.")
        heads = {
            MistakeTaxonomy.OVERWEIGHTED_CORRELATED:
                "confidence rested on correlated evidence that looked like confirmation",
            MistakeTaxonomy.OVERCONFIDENT:
                "confidence exceeded what the evidence supported",
            MistakeTaxonomy.IGNORED_CONTRADICTION:
                "contradicting evidence was present and was not resolved",
            MistakeTaxonomy.STALE_INFORMATION:
                "the call leaned on information that was already old",
            MistakeTaxonomy.REGIME_MISREAD:
                "the regime was misclassified, so the base rates did not apply",
            MistakeTaxonomy.BAD_LEVELS:
                "a clean stop-out — the stop sat inside normal noise",
            MistakeTaxonomy.RED_TEAM_IGNORED:
                "the red team objected and was right",
        }
        reasons = [heads.get(t, t.value) for t in tax]
        return (f"{subject} failed ({realized_r:+.2f}R): " + "; ".join(reasons) + ".")

    def _is_recurring(self, tax: list[MistakeTaxonomy]) -> bool:
        freq = self.memory.mistake_frequency()
        return any(freq.get(t.value, 0) + 1 >= RECURRENCE_THRESHOLD for t in tax)

    # ------------------------------------------------------------------

    def persist(self, aar: AfterActionReview) -> dict:
        self.conn.execute(
            """INSERT INTO after_action_reviews
               (created_at, recommendation_id, expected, actual, got_right, got_wrong,
                agents_helped, agents_hurt, evidence_mattered, evidence_noise,
                red_team_performance, calibration_error, lesson, became_memory,
                triggered_experiment)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (aar.created_at, aar.recommendation_id, aar.expected, aar.actual,
             json.dumps(aar.got_right), json.dumps(aar.got_wrong),
             json.dumps(aar.agents_helped), json.dumps(aar.agents_hurt),
             json.dumps(aar.evidence_mattered), json.dumps(aar.evidence_noise),
             aar.red_team_performance, aar.calibration_error, aar.lesson,
             int(aar.should_become_memory), int(aar.should_trigger_experiment)))
        self.conn.commit()

        mistake_id = None
        if aar.should_become_memory:
            mistake_id = self.memory.record_mistake(Mistake(
                subject=aar.subject,
                predicted=aar.expected,
                actual=aar.actual,
                taxonomy=aar.taxonomy,
                lesson=aar.lesson,
                error_r=aar.calibration_error,
                recommendation_id=aar.recommendation_id,
                correlation_double_counted=(
                    MistakeTaxonomy.OVERWEIGHTED_CORRELATED in aar.taxonomy),
                stale_information=MistakeTaxonomy.STALE_INFORMATION in aar.taxonomy,
                regime_misread=MistakeTaxonomy.REGIME_MISREAD in aar.taxonomy,
                red_team_ignored=MistakeTaxonomy.RED_TEAM_IGNORED in aar.taxonomy,
                red_team_caught=bool(aar.red_team_performance
                                     and "were correct" in aar.red_team_performance),
                genuinely_unpredictable=MistakeTaxonomy.UNPREDICTABLE in aar.taxonomy,
            ))
        elif aar.succeeded and aar.subject:
            # Successes are worth remembering too, at lower confidence — a
            # store of failures alone teaches an unbalanced lesson.
            self.memory.remember(Memory(
                category=MemoryCategory.COMPANY,
                title=f"{aar.subject}: setup worked",
                body=aar.lesson,
                subject=aar.subject,
                confidence=0.4,
                evidence_n=1,
                source="after_action_review",
                related_entities=[aar.subject],
            ))

        return {"persisted": True, "mistake_id": mistake_id,
                "triggered_experiment": aar.should_trigger_experiment}
