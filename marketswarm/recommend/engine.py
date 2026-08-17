"""The Recommendation Engine.

1.x emitted predictions: a direction, a probability, a bracket. Every candidate
became long or short, because the vocabulary contained nothing else. A system
that cannot say "I don't know" will say something else instead, and that
something is noise wearing a number.

This engine produces *recommendations*, and the vocabulary includes ignorance:
`INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE` and `NO_EDGE` are first-class
outcomes reached by explicit rules, not failure states.

It separates the four things 1.x ran together:

    Observation      what was measured
    Interpretation   what we think it means
    Prediction       the probability, with its uncertainty
    Recommendation   what, if anything, follows

and it refuses to collapse them, because the reader needs to see which part
they disagree with.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum

from ..evidence.graph import EvidenceGraph, EvidenceNode


class Conviction(str, Enum):
    HIGH_CONVICTION = "HIGH_CONVICTION"
    MODERATE_CONVICTION = "MODERATE_CONVICTION"
    LOW_CONVICTION = "LOW_CONVICTION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    NO_ACTIONABLE_EDGE = "NO_ACTIONABLE_EDGE"

    @property
    def is_actionable(self) -> bool:
        return self in (Conviction.HIGH_CONVICTION, Conviction.MODERATE_CONVICTION)


class RecommendationType(str, Enum):
    WATCH = "WATCH"
    INVESTIGATE = "INVESTIGATE"
    FAVORABLE = "FAVORABLE"
    UNFAVORABLE = "UNFAVORABLE"
    AVOID = "AVOID"
    WAIT_FOR_CONFIRMATION = "WAIT_FOR_CONFIRMATION"
    NO_EDGE = "NO_EDGE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

    @property
    def implies_direction(self) -> bool:
        return self in (RecommendationType.FAVORABLE, RecommendationType.UNFAVORABLE)


@dataclass
class RecommendationInputs:
    """Everything the engine is allowed to consider. Explicit so the decision
    is reproducible and so a missing input is visible rather than assumed."""

    subject: str
    direction: str | None = None
    probability: float | None = None            # P(target before stop)
    confidence: int = 0                          # 0-100
    expected_r: float | None = None
    entry: float | None = None
    target: float | None = None
    stop: float | None = None
    effective_independent_evidence: float = 0.0
    n_contradictions: int = 0
    regime: str = "unknown"
    regime_confidence: float | None = None
    event_pending: bool = False                  # unresolved catalyst before the horizon
    data_quality: str = "ok"                     # ok | degraded | poor
    missing_critical_agents: list[str] = field(default_factory=list)
    red_team_severity: str | None = None
    review_status: str | None = None
    historical_calibration_gap: float | None = None   # forecast − observed
    prior_failures: list[str] = field(default_factory=list)
    liquidity_score: float = 0.5


@dataclass
class Recommendation:
    id: str
    subject: str
    rec_type: RecommendationType
    conviction: Conviction
    direction: str | None
    forecast_probability: float | None
    confidence: int
    original_confidence: int
    expected_r: float | None
    entry: float | None = None
    target: float | None = None
    stop: float | None = None
    horizon: str = "intraday"
    regime: str = "unknown"
    data_quality: str = "ok"

    observation: str = ""
    interpretation: str = ""
    prediction: str = ""
    rationale: str = ""
    key_risks: list[str] = field(default_factory=list)
    invalidation: list[str] = field(default_factory=list)
    change_our_mind: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    historical_analogues: list[str] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    agent_contributors: dict = field(default_factory=dict)
    review_status: str | None = None
    revision_count: int = 0
    investigation_id: str | None = None
    model_version: str = ""
    system_version: str = ""
    experiment_id: str | None = None

    # Lineage. Written once, never recomputed, so a published recommendation
    # can always be traced back to the candidate and review that produced it.
    candidate_id: str | None = None
    evidence_graph_id: str | None = None
    review_decision_id: str | None = None
    revision_parent_id: str | None = None

    # Presentation only. Carries the option strike, expiration and premium
    # zones the engine has no opinion about. Every decision-bearing field in
    # here is overwritten from this object by `publication._payload_from`, so
    # it can never become a second source of truth for confidence or levels.
    source_kind: str | None = None            # call | put | stock
    source_payload: dict = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )

    @property
    def actionable(self) -> bool:
        return self.conviction.is_actionable and self.rec_type.implies_direction

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rec_type"] = self.rec_type.value
        d["conviction"] = self.conviction.value
        return d

    def headline(self) -> str:
        arrow = f" {self.direction}" if self.direction else ""
        return (f"{self.subject}{arrow}: {self.rec_type.value} "
                f"[{self.conviction.value}] confidence {self.confidence}/100")


@dataclass
class EnginePolicy:
    min_independent_for_action: float = 2.5
    min_confidence_for_action: int = 40
    min_confidence_moderate: int = 30
    high_conviction_confidence: int = 65
    high_conviction_independent: float = 4.0
    min_expected_r: float = 0.0
    contradiction_limit: int = 2          # more than this ⇒ CONFLICTING_EVIDENCE
    poor_quality_blocks_action: bool = True


class RecommendationEngine:
    def __init__(self, policy: EnginePolicy | None = None,
                 system_version: str = "2.0.1"):
        self.policy = policy or EnginePolicy()
        self.system_version = system_version

    # ------------------------------------------------------------------

    def assess_conviction(self, inp: RecommendationInputs) -> tuple[Conviction, list[str]]:
        """Decide how much we actually know. Order matters: ignorance is checked
        before edge, because a confident number computed from nothing is the
        most dangerous output the system can produce."""
        p = self.policy
        notes: list[str] = []

        # 1. Is the evidence base even adequate?
        if inp.data_quality == "poor" or inp.missing_critical_agents:
            notes.append(
                "critical inputs missing: "
                + (", ".join(inp.missing_critical_agents) or "data quality poor")
            )
            return Conviction.INSUFFICIENT_EVIDENCE, notes

        if inp.effective_independent_evidence < 1.5:
            notes.append(
                f"only {inp.effective_independent_evidence:.1f} effectively independent "
                f"pieces of evidence — correlated signals are not confirmation"
            )
            return Conviction.INSUFFICIENT_EVIDENCE, notes

        # 2. Does the evidence disagree with itself?
        if inp.n_contradictions > p.contradiction_limit:
            notes.append(f"{inp.n_contradictions} unresolved contradictions in the evidence")
            return Conviction.CONFLICTING_EVIDENCE, notes

        # 3. Is there an edge worth acting on, after costs?
        if inp.expected_r is not None and inp.expected_r <= p.min_expected_r:
            notes.append(
                f"expected value {inp.expected_r:+.3f}R does not clear costs — "
                f"the analysis may be right and still not worth trading"
            )
            return Conviction.NO_ACTIONABLE_EDGE, notes

        if inp.probability is not None and 0.47 <= inp.probability <= 0.53:
            notes.append(
                f"P={inp.probability:.0%} is within noise of a coin flip; the outcome "
                f"would be decided by execution, not by the analysis"
            )
            return Conviction.NO_ACTIONABLE_EDGE, notes

        # 4. Graded conviction
        if inp.confidence < p.min_confidence_moderate:
            notes.append(f"confidence {inp.confidence}/100 is too low to act on")
            return Conviction.LOW_CONVICTION, notes

        if (inp.confidence >= p.high_conviction_confidence
                and inp.effective_independent_evidence >= p.high_conviction_independent
                and inp.n_contradictions == 0
                and not inp.event_pending):
            return Conviction.HIGH_CONVICTION, notes

        if inp.confidence >= p.min_confidence_for_action:
            if inp.event_pending:
                notes.append("an unresolved catalyst lands before the horizon — "
                             "conviction capped until it prints")
            return Conviction.MODERATE_CONVICTION, notes

        return Conviction.LOW_CONVICTION, notes

    def choose_type(self, inp: RecommendationInputs,
                    conviction: Conviction) -> RecommendationType:
        if conviction is Conviction.INSUFFICIENT_EVIDENCE:
            return RecommendationType.INSUFFICIENT_EVIDENCE
        if conviction is Conviction.CONFLICTING_EVIDENCE:
            return RecommendationType.INVESTIGATE
        if conviction is Conviction.NO_ACTIONABLE_EDGE:
            return RecommendationType.NO_EDGE

        if inp.data_quality == "degraded":
            return RecommendationType.WATCH
        if inp.red_team_severity in ("high", "critical"):
            return RecommendationType.WAIT_FOR_CONFIRMATION
        if inp.event_pending:
            return RecommendationType.WAIT_FOR_CONFIRMATION
        if inp.liquidity_score < 0.25:
            return RecommendationType.AVOID

        if conviction is Conviction.LOW_CONVICTION:
            return RecommendationType.WATCH
        if inp.direction == "long":
            return RecommendationType.FAVORABLE
        if inp.direction == "short":
            return RecommendationType.UNFAVORABLE
        return RecommendationType.WATCH

    # ------------------------------------------------------------------

    def build(
        self,
        inp: RecommendationInputs,
        graph: EvidenceGraph | None = None,
        thesis_node: EvidenceNode | None = None,
        analogues: list[str] | None = None,
        open_questions: list[str] | None = None,
        agent_contributors: dict | None = None,
        investigation_id: str | None = None,
        experiment_id: str | None = None,
    ) -> Recommendation:
        conviction, notes = self.assess_conviction(inp)
        rec_type = self.choose_type(inp, conviction)

        supporting: list[str] = []
        contradicting: list[str] = []
        change_mind: list[str] = []
        if graph is not None and thesis_node is not None:
            expl = graph.explain(thesis_node)
            supporting = expl["believe_because"]
            contradicting = expl["would_be_invalidated_by"]
            change_mind = list(contradicting)
        elif graph is not None:
            scored = sorted(graph.nodes.values(),
                            key=lambda n: -(n.score.composite() or 0))
            supporting = [n.describe() for n in scored[:6]]
            contradicting = [f"{a.claim} ⟂ {b.claim}" for a, b in graph.contradictions()[:4]]

        if not change_mind:
            change_mind = self._default_change_our_mind(inp)

        rec = Recommendation(
            id=f"rec_{uuid.uuid4().hex[:12]}",
            subject=inp.subject,
            rec_type=rec_type,
            conviction=conviction,
            direction=inp.direction if rec_type.implies_direction else None,
            forecast_probability=inp.probability,
            confidence=inp.confidence,
            original_confidence=inp.confidence,
            expected_r=inp.expected_r,
            entry=inp.entry, target=inp.target, stop=inp.stop,
            regime=inp.regime,
            data_quality=inp.data_quality,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            change_our_mind=change_mind,
            historical_analogues=list(analogues or []),
            uncertainty_notes=notes,
            open_questions=list(open_questions or []),
            agent_contributors=dict(agent_contributors or {}),
            review_status=inp.review_status,
            investigation_id=investigation_id,
            experiment_id=experiment_id,
            system_version=self.system_version,
        )

        rec.observation = self._observation(inp)
        rec.interpretation = self._interpretation(inp, conviction)
        rec.prediction = self._prediction(inp, conviction)
        rec.rationale = self._rationale(inp, conviction, rec_type)
        rec.key_risks = self._risks(inp, conviction)
        rec.invalidation = self._invalidation(inp)
        return rec

    # ------------------------------------------------------------------
    # narrative sections, generated deterministically from the inputs
    # ------------------------------------------------------------------

    def _observation(self, inp: RecommendationInputs) -> str:
        bits = [f"{inp.subject} assessed in a {inp.regime.replace('_', ' ')} regime"]
        if inp.regime_confidence is not None:
            bits[-1] += f" (regime confidence {inp.regime_confidence:.0%})"
        bits.append(f"{inp.effective_independent_evidence:.1f} effectively independent "
                    f"pieces of evidence")
        if inp.n_contradictions:
            bits.append(f"{inp.n_contradictions} contradiction(s) present")
        if inp.data_quality != "ok":
            bits.append(f"data quality {inp.data_quality}")
        return "; ".join(bits) + "."

    def _interpretation(self, inp: RecommendationInputs, c: Conviction) -> str:
        if c is Conviction.INSUFFICIENT_EVIDENCE:
            return ("The evidence base is too thin to support any interpretation. "
                    "This is a statement about our information, not about the security.")
        if c is Conviction.CONFLICTING_EVIDENCE:
            return ("Credible evidence points both ways and the conflict was not resolved. "
                    "Averaging it into a single direction would hide the disagreement.")
        if c is Conviction.NO_ACTIONABLE_EDGE:
            return ("A direction is discernible, but not by enough to survive trading "
                    "costs. Being right is not the same as being paid.")
        lean = ("higher" if inp.direction == "long" else "lower"
                if inp.direction == "short" else "unclear")
        return (f"The weight of independent evidence leans {lean} over the "
                f"{inp.regime.replace('_', ' ')} session.")

    def _prediction(self, inp: RecommendationInputs, c: Conviction) -> str:
        if inp.probability is None:
            return "No probability was produced — there is nothing to forecast here."
        base = (f"P(target before stop) = {inp.probability:.0%}"
                + (f", expected value {inp.expected_r:+.2f}R after costs"
                   if inp.expected_r is not None else ""))
        if inp.historical_calibration_gap is not None:
            gap = inp.historical_calibration_gap
            if abs(gap) > 0.05:
                base += (f". Historically this system has been "
                         f"{'over' if gap > 0 else 'under'}confident by "
                         f"{abs(gap):.0%} — read the number with that in mind")
        return base

    def _rationale(self, inp: RecommendationInputs, c: Conviction,
                   t: RecommendationType) -> str:
        if t is RecommendationType.INSUFFICIENT_EVIDENCE:
            return ("Not enough independent evidence was gathered to justify a view. "
                    "Reporting a direction anyway would manufacture confidence the "
                    "data does not support.")
        if t is RecommendationType.NO_EDGE:
            return ("The setup does not clear expected value after costs. Standing "
                    "aside is the position with the highest expected return today.")
        if t is RecommendationType.INVESTIGATE:
            return ("The evidence conflicts. The useful next step is resolving the "
                    "conflict, not taking a position on top of it.")
        if t is RecommendationType.WAIT_FOR_CONFIRMATION:
            reason = ("a red-team objection of "
                      f"{inp.red_team_severity} severity stands"
                      if inp.red_team_severity in ("high", "critical")
                      else "an unresolved catalyst lands before the horizon")
            return (f"The directional case is reasonable, but {reason}. "
                    f"Confirmation is cheaper than being early.")
        if t is RecommendationType.AVOID:
            return ("Liquidity is inadequate. Idea quality is irrelevant when the "
                    "fill destroys the edge.")
        if t is RecommendationType.WATCH:
            return ("There is a case here, but not a strong enough one to act on. "
                    "Worth monitoring rather than trading.")
        return (f"{inp.effective_independent_evidence:.1f} independent evidence "
                f"categories support the {inp.direction} case at "
                f"{inp.confidence}/100 confidence.")

    def _risks(self, inp: RecommendationInputs, c: Conviction) -> list[str]:
        risks: list[str] = []
        if inp.event_pending:
            risks.append("an unresolved scheduled catalyst can invalidate every "
                         "technical level in this analysis")
        if inp.regime in ("stress", "HIGH_VOLATILITY", "LIQUIDITY_STRESS"):
            risks.append(f"{inp.regime} regime — levels fail more often and stops slip")
        if inp.regime_confidence is not None and inp.regime_confidence < 0.5:
            risks.append(f"regime classification is itself uncertain "
                         f"({inp.regime_confidence:.0%}) — conditional base rates may not apply")
        if inp.n_contradictions:
            risks.append(f"{inp.n_contradictions} contradiction(s) remain unresolved")
        if inp.effective_independent_evidence < 3:
            risks.append("the evidence base is narrow; one wrong input moves the conclusion")
        if inp.data_quality != "ok":
            risks.append(f"data quality is {inp.data_quality}")
        for f in inp.prior_failures[:2]:
            risks.append(f"prior failure in similar conditions: {f}")
        if inp.historical_calibration_gap and inp.historical_calibration_gap > 0.08:
            risks.append(f"this system has historically been overconfident by "
                         f"{inp.historical_calibration_gap:.0%} on comparable calls")
        return risks

    def _invalidation(self, inp: RecommendationInputs) -> list[str]:
        out: list[str] = []
        if inp.stop is not None and inp.entry is not None:
            out.append(f"{inp.subject} trading through {inp.stop:.2f} on a 5-minute close")
        if inp.event_pending:
            out.append("the pending release printing materially away from consensus")
        out.append("a regime shift — VIX up more than 10% intraday invalidates the "
                   "base rates this forecast rests on")
        if inp.direction == "long":
            out.append("failure to hold above the pre-market high in the first 30 minutes")
        elif inp.direction == "short":
            out.append("failure to break the pre-market low in the first 30 minutes")
        return out

    def _default_change_our_mind(self, inp: RecommendationInputs) -> list[str]:
        out = ["a primary-source filing contradicting the current interpretation",
               "the move proving to be sector beta rather than company-specific"]
        if inp.direction:
            opposite = "selling" if inp.direction == "long" else "buying"
            out.append(f"sustained {opposite} pressure on rising volume after the open")
        if inp.effective_independent_evidence < 3:
            out.append("a genuinely independent second source either way — the current "
                       "evidence is too narrow to be robust")
        return out
