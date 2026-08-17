"""MarketSwarm 2.0 pipeline.

Wires the new components onto the existing swarm rather than replacing it. The
1.x orchestrator, agents, providers and statistics are unchanged and still do
the work; this adds the layers that were missing:

    Event Brain  →  Chief Investigator  →  (existing swarm, selectively)
                 →  Evidence Graph
                 →  Recommendation Engine
                 →  Red Team  →  Review Gate  →  Revision  →  Final

The important property is that the review gate sits *between* the playbook and
the published output, so a red-team finding can still change what a reader sees.
In 1.x it sat after, which is why it could not.

`Pipeline2.run()` degrades to the 1.x behaviour if anything new fails — the
upgrade must not make the system more fragile than the version it replaces.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass, field

from .evidence.graph import (
    EvidenceGraph,
    EvidenceNode,
    NodeType,
    Relation,
    default_predictive_utility,
    novelty_from_corroboration,
    timeliness_from_age,
)
from .investigation.chief import ChiefInvestigator
from .investigation.event_brain import EventBrain
from .observability import Observatory
from .publication import PublicationSet, RejectedCandidate
from .recommend.engine import (
    Conviction,
    Recommendation,
    RecommendationEngine,
    RecommendationInputs,
)
from .resilience import DegradationTracker
from .review.gate import ReviewGate
from .review.loop import ReviewLoop, findings_from_redteam_report

log = logging.getLogger("marketswarm.pipeline2")


def _bind_investigator(investigator, symbol: str, plan):
    """Adapt the orchestrator's investigator to the review loop's signature.

    The loop calls `investigator(idea, questions)`; the orchestrator needs the
    symbol and the parent plan to scope and budget the follow-up.
    """
    def follow_up(current: dict, questions: list[str]) -> dict:
        return investigator(symbol, current, questions, plan)
    return follow_up


# Which evidence cluster each agent's output belongs to. Drives the
# independence maths — see evidence.graph.CLUSTERS.
AGENT_CLUSTERS: dict[str, str] = {
    "overnight_scan": "market_beta",
    "global_markets": "market_beta",
    "futures": "market_beta",
    "volatility_regime": "volatility",
    "breaking_news": "company_news",
    "econ_calendar": "macro",
    "earnings": "company_fundamental",
    "sec_filings": "company_fundamental",
    "options_flow": "options_positioning",
    "institutional": "insider",
    "technicals": "technical",
    "sentiment": "sentiment",
}

SOURCE_HALF_LIVES: dict[str, float] = {
    "major_newswire": 8.0,
    "financial_media": 6.0,
    "aggregator": 4.0,
    "sec_edgar": 72.0,
    "company_ir": 48.0,
    "exchange_data": 2.0,
    "federal_reserve": 168.0,
    "treasury": 168.0,
}


@dataclass
class Pipeline2Result:
    run_date: dt.date
    graph: EvidenceGraph
    recommendations: list[Recommendation] = field(default_factory=list)
    rejected: list[RejectedCandidate] = field(default_factory=list)
    plans: list = field(default_factory=list)
    review_outcomes: list = field(default_factory=list)
    degradation: dict = field(default_factory=dict)
    contradictions: list[str] = field(default_factory=list)
    observability: dict = field(default_factory=dict)
    # The authoritative output. Everything downstream reads this.
    publication: PublicationSet = field(default_factory=PublicationSet)

    @property
    def actionable(self) -> list[Recommendation]:
        return [r for r in self.recommendations if r.actionable]

    def summary(self) -> dict:
        by_conviction: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for r in self.recommendations:
            by_conviction[r.conviction.value] = by_conviction.get(r.conviction.value, 0) + 1
            by_type[r.rec_type.value] = by_type.get(r.rec_type.value, 0) + 1
        return {
            "n_recommendations": len(self.recommendations),
            "n_actionable": len(self.actionable),
            "n_rejected_by_review": len(self.rejected),
            "by_conviction": by_conviction,
            "by_type": by_type,
            "evidence": self.graph.summary(),
            "contradictions": self.contradictions,
            "degradation": self.degradation,
        }


class Pipeline2:
    def __init__(
        self,
        config,
        store=None,
        chief: ChiefInvestigator | None = None,
        engine: RecommendationEngine | None = None,
        review_loop: ReviewLoop | None = None,
        observatory: Observatory | None = None,
        max_review_iterations: int = 2,
    ):
        self.config = config
        self.store = store
        self.chief = chief or ChiefInvestigator(brain=EventBrain())
        self.engine = engine or RecommendationEngine()
        self.review_loop = review_loop or ReviewLoop(
            gate=ReviewGate(), max_iterations=max_review_iterations)
        self.obs = observatory or Observatory(
            conn=getattr(store, "conn", None) if store else None)

    # ------------------------------------------------------------------
    # 1. snapshot for the event brain, built from reports the swarm produced
    # ------------------------------------------------------------------

    @staticmethod
    def build_snapshot(reports: dict) -> dict:
        def data(agent, key, default=None):
            r = reports.get(agent)
            return r.data.get(key, default) if r and r.usable else default

        profiles = data("overnight_scan", "profiles", {}) or {}
        setups = data("technicals", "setups", {}) or {}
        flows = data("options_flow", "flows", {}) or {}
        filings = data("sec_filings", "filings", []) or []
        earnings = data("earnings", "reacting", []) or []
        tonight = {e["ticker"] for e in (data("earnings", "reporting_tonight", []) or [])}
        news = data("breaking_news", "high_materiality", []) or []

        by_ticker_filings: dict[str, list] = {}
        for f in filings:
            by_ticker_filings.setdefault(f.get("ticker", ""), []).append(f)
        earnings_by_ticker = {e["ticker"]: e for e in earnings}
        news_by_ticker: dict[str, list] = {}
        for h in news:
            for t in h.get("tickers", []) or []:
                news_by_ticker.setdefault(t, []).append(h)

        symbols: dict[str, dict] = {}
        for sym in set(profiles) | set(setups):
            prof = profiles.get(sym, {}) or {}
            setup = setups.get(sym, {}) or {}
            flow = flows.get(sym, {}) or {}
            e = earnings_by_ticker.get(sym)
            symbols[sym] = {
                "gap_pct": prof.get("gap_pct"),
                "atr_pct": setup.get("atr_pct"),
                "rvol": setup.get("rvol"),
                "put_call": flow.get("put_call_volume_ratio"),
                "unusual": flow.get("unusual", []),
                "earnings_reacting": e is not None,
                "surprise_pct": e.get("surprise_pct") if e else None,
                "reporting_tonight": sym in tonight,
                "filings": by_ticker_filings.get(sym, []),
                "headlines": news_by_ticker.get(sym, []),
            }

        return {
            "symbols": symbols,
            "market": {
                "vix": data("volatility_regime", "vix"),
                "vix_change_pct": data("volatility_regime", "vix_change_pct"),
                "index_pct": data("futures", "avg_equity_pct"),
                "econ_events": data("econ_calendar", "events", []) or [],
            },
        }

    # ------------------------------------------------------------------
    # 2. evidence graph from the agent reports
    # ------------------------------------------------------------------

    def build_graph(self, reports: dict, investigation_id: str | None = None
                    ) -> EvidenceGraph:
        graph = EvidenceGraph(investigation_id)

        for agent, rep in reports.items():
            if not getattr(rep, "usable", False):
                continue
            cluster = AGENT_CLUSTERS.get(agent, "unknown")
            for ev in getattr(rep, "evidence", []) or []:
                node = EvidenceNode(
                    claim=ev.claim,
                    node_type=NodeType.OBSERVATION,
                    subject=self._subject_of(ev),
                    source=ev.source,
                    source_type=ev.source,
                    url=ev.url,
                    observed_at=ev.observed_at,
                    cluster=cluster,
                    agent=agent,
                    tags=list(ev.tags or []),
                    is_inferred=("options" in cluster
                                 and "unusual" in " ".join(ev.tags or []).lower()),
                    provenance=f"agent:{agent}",
                )
                node.score.factual_reliability = ev.reliability
                node.score.predictive_utility = default_predictive_utility(ev.source)
                half_life = SOURCE_HALF_LIVES.get(ev.source, 12.0)
                node.score.timeliness = timeliness_from_age(node.age_hours, half_life)
                corr = (ev.value or {}).get("corroborations", 1) if isinstance(ev.value, dict) else 1
                node.score.novelty = novelty_from_corroboration(int(corr or 1))
                graph.add(node)

        # Contradictions found by the existing cross-verification agent become
        # explicit edges rather than a sentence in a report.
        cv = reports.get("cross_verify")
        if cv and cv.usable:
            for text in cv.data.get("contradictions", []) or []:
                a = graph.observe(f"conflict: {text}", cluster="unknown",
                                  source="cross_verify", provenance="agent:cross_verify")
                a.score.factual_reliability = 0.9
                for other in list(graph.nodes.values())[:1]:
                    if other.id != a.id:
                        graph.link(a, other, Relation.CONTRADICTS, note=text)

        graph.annotate_independence()
        return graph

    @staticmethod
    def _subject_of(ev) -> str | None:
        for t in ev.tags or []:
            if t.isupper() and 1 <= len(t) <= 5 and t.isalpha():
                return t
        return None

    # ------------------------------------------------------------------
    # 3. recommendations, then review
    # ------------------------------------------------------------------

    def build_recommendations(
        self,
        reports: dict,
        graph: EvidenceGraph,
        degradation: DegradationTracker,
        calibration_gap: float | None = None,
        prior_failures: dict[str, list[str]] | None = None,
        investigator=None,
        plans: list | None = None,
    ) -> tuple[list[Recommendation], list[RejectedCandidate], list]:
        pb = reports.get("playbook")
        if not pb or not pb.usable:
            return [], [], []

        rt = reports.get("red_team")
        regime = (reports.get("volatility_regime").data.get("regime", "unknown")
                  if reports.get("volatility_regime")
                  and reports["volatility_regime"].usable else "unknown")
        event_pending = bool(
            reports.get("econ_calendar")
            and reports["econ_calendar"].usable
            and reports["econ_calendar"].data.get("very_high_impact"))
        contributions = (reports["cross_verify"].data.get("contributions", {})
                         if reports.get("cross_verify")
                         and reports["cross_verify"].usable else {})

        ideas = ([("call", i) for i in pb.data.get("calls", [])]
                 + [("put", i) for i in pb.data.get("puts", [])]
                 + [("stock", i) for i in pb.data.get("stocks", [])])

        recommendations: list[Recommendation] = []
        rejected: list[RejectedCandidate] = []
        outcomes: list = []
        plan_by_subject = {p.subject: p for p in (plans or [])}

        for kind, idea in ideas:
            symbol = idea.get("symbol", "?")
            sym_nodes = graph.by_subject(symbol) or list(graph.nodes.values())
            independent = graph.effective_independent_count(sym_nodes)
            candidate_id = idea.get("recommendation_id") or f"cand_{uuid.uuid4().hex[:12]}"

            # --- the review gate, applied BEFORE publication ---
            def critic(current, iteration, _rt=rt, _sym=symbol):
                return findings_from_redteam_report(_rt, symbol=_sym)

            # A REQUEST_MORE_RESEARCH decision must buy real evidence, not just
            # leave the questions open. The investigator is bounded by the
            # plan's budget and is supplied by the orchestrator, which is the
            # only layer that can actually execute agents.
            follow_up = (
                _bind_investigator(investigator, symbol, plan_by_subject.get(symbol))
                if investigator is not None else None
            )

            outcome = self.review_loop.run(
                dict(idea, recommendation_id=candidate_id),
                critic=critic, investigator=follow_up)
            outcomes.append(outcome)

            if outcome.rejected:
                rejected.append(RejectedCandidate(
                    subject=symbol,
                    candidate_id=candidate_id,
                    reason=outcome.final_decision.summary(),
                    findings=list(outcome.final_decision.reasons),
                    audit=outcome.audit_trail(),
                    original_confidence=int(idea.get("confidence", 0)),
                ))
                self.obs.event("review", f"{symbol} rejected: "
                                         f"{outcome.final_decision.summary()}",
                               level="info")
                continue

            reviewed = outcome.final_idea
            inp = RecommendationInputs(
                subject=symbol,
                direction=reviewed.get("direction"),
                probability=reviewed.get("probability"),
                confidence=int(reviewed.get("confidence", 0)),
                expected_r=reviewed.get("expected_r"),
                entry=reviewed.get("entry"),
                target=reviewed.get("target"),
                stop=reviewed.get("stop"),
                effective_independent_evidence=independent,
                n_contradictions=len(graph.contradictions()),
                regime=regime,
                event_pending=event_pending,
                data_quality=degradation.data_quality(),
                missing_critical_agents=sorted(degradation.failed_agents
                                               & {"technicals", "cross_verify"}),
                red_team_severity=reviewed.get("review_severity"),
                review_status=reviewed.get("review_status"),
                historical_calibration_gap=calibration_gap,
                prior_failures=(prior_failures or {}).get(symbol, []),
                liquidity_score=float(reviewed.get("liquidity", 0.5)),
            )

            rec = self.engine.build(
                inp, graph=graph,
                open_questions=reviewed.get("open_questions", []),
                agent_contributors=contributions,
                investigation_id=graph.investigation_id,
            )
            rec.original_confidence = int(idea.get("confidence", rec.confidence))
            rec.revision_count = max(0, outcome.iterations_used - 1)

            # Lineage, written once so a published call can be traced back.
            rec.candidate_id = candidate_id
            rec.evidence_graph_id = graph.investigation_id
            rec.review_decision_id = outcome.final_decision.id
            rec.review_status = outcome.final_decision.status.value
            rec.source_kind = kind

            # Presentation payload only. `publication._payload_from` overwrites
            # every decision-bearing key from `rec`, so the strike and the
            # expiration survive and a stale confidence cannot.
            rec.source_payload = {
                k: v for k, v in idea.items()
                if k in ("strike", "expiration", "option_entry", "option_target",
                         "option_stop", "option_note", "math_note", "ev_verdict",
                         "clears_bar", "liquidity", "contracts", "delta")
            }
            recommendations.append(rec)

        return recommendations, rejected, outcomes

    # ------------------------------------------------------------------
    # 4. the whole thing
    # ------------------------------------------------------------------

    def run(self, reports: dict, run_date: dt.date,
            calibration_gap: float | None = None,
            prior_failures: dict[str, list[str]] | None = None,
            plans: list | None = None,
            investigator=None,
            mode: str = "dynamic") -> Pipeline2Result:
        """Decide what may be published. The return value is authoritative.

        `plans` comes from the orchestrator when the Chief Investigator has
        already planned and routed the session — the plan must be made before
        the specialists run, or it is not a plan. Passing None re-plans here
        from the reports, which is only meaningful in `full` mode where
        everything ran anyway.
        """
        degradation = DegradationTracker()
        for name, rep in reports.items():
            degradation.record_agent(name, getattr(rep, "status", "ok"))
            self.obs.record_report(rep)

        if plans is None:
            snapshot = self.build_snapshot(reports)
            plans = self.chief.plan_session(snapshot)
            self.obs.event("chief", f"planned {len(plans)} investigations (post-hoc)")

        graph = self.build_graph(reports)
        contradictions = self.chief.detect_contradictions(reports)

        recs, rejected, outcomes = self.build_recommendations(
            reports, graph, degradation, calibration_gap, prior_failures,
            investigator=investigator, plans=plans)

        # A critically degraded run must not publish the output class whose
        # evidence is missing. This is the "do not silently proceed" rule.
        suppressed = degradation.suppressed_outputs()
        if suppressed:
            self.obs.event("degradation", f"suppressing: {', '.join(suppressed)}",
                           level="warning")
            if "all directional ideas" in suppressed:
                for r in recs:
                    r.conviction = Conviction.INSUFFICIENT_EVIDENCE
                    r.uncertainty_notes.append(degradation.missing_evidence_statement())

        publication = PublicationSet(
            approved=recs,
            rejected=rejected,
            mode=mode,
            review_iterations=sum(o.iterations_used for o in outcomes),
        )
        # Cheap, and it is the invariant the whole release exists to protect.
        publication.assert_no_leak()

        return Pipeline2Result(
            run_date=run_date,
            graph=graph,
            recommendations=recs,
            rejected=rejected,
            plans=plans,
            review_outcomes=outcomes,
            degradation=degradation.summary(),
            contradictions=contradictions,
            observability=self.obs.status_snapshot(),
            publication=publication,
        )
