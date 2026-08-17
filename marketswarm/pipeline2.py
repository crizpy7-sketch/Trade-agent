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
from .publication import RecordStatus
from .review.fallback import deterministic_review
from .review.gate import (Finding, ReviewExecutionStatus, ReviewGate, ReviewStatus,
                          Severity)
from .review.loop import (CriticResult, ReviewLoop, findings_from_redteam_report,
                          redteam_execution_status)

log = logging.getLogger("marketswarm.pipeline2")


class CandidateReview:
    """Per-candidate review state, refreshed between rounds.

    This class exists because of a specific defect. Before it, the evidence
    graph was built once for the whole session and the red-team report was
    captured once into a closure. Follow-up research genuinely ran agents and
    genuinely wrote new reports — and round 2 then re-derived its objections
    from the same captured report object, producing byte-identical findings.
    The loop's own no-progress guard then terminated, so the research was
    guaranteed to be wasted by construction.

    The fix is that a round reads the *current* state:

        critic()  →  refresh graph from live reports (if they changed)
                  →  re-derive findings from the current red-team report
                  →  report the execution status and graph version

    `reports` is the live dict the orchestrator mutates, so an agent that runs
    during follow-up is visible here immediately.
    """

    def __init__(self, pipeline: "Pipeline2", reports: dict, symbol: str,
                 graph: EvidenceGraph, plan, investigator=None):
        self.pipeline = pipeline
        self.reports = reports
        self.symbol = symbol
        self.graph = graph
        self.plan = plan
        self.investigator = investigator

        self.graph_version = 1
        self.dirty = False                      # new reports since the last graph build
        self.independent = self._independence()
        self.rounds: list[dict] = []
        self.last_status = ReviewExecutionStatus.COMPLETED
        self.followup_agents: list[str] = []
        self.followup_failures: list[str] = []
        self.nodes_before_followup = len(graph.nodes)
        self.nodes_after_followup = len(graph.nodes)
        self.candidate_id = ""

    # ------------------------------------------------------------------

    def _independence(self) -> float:
        nodes = self.graph.by_subject(self.symbol) or list(self.graph.nodes.values())
        return self.graph.effective_independent_count(nodes)

    def refresh_evidence(self) -> None:
        """Rebuild the graph from the current reports.

        Option B from the two the design allowed: a full rebuild rather than
        incremental ingestion. `build_graph` is already a pure function of the
        report set, so rebuilding is both simpler and safer than maintaining a
        second insertion path that could disagree with it. The cost is a few
        hundred node constructions on a session that asked for more research —
        which is not the hot path.
        """
        self.graph = self.pipeline.build_graph(self.reports)
        self.graph_version += 1
        self.independent = self._independence()
        self.nodes_after_followup = len(self.graph.nodes)
        self.dirty = False
        log.info("evidence refreshed for %s: graph v%d, %d nodes, "
                 "%.2f effective independent", self.symbol, self.graph_version,
                 len(self.graph.nodes), self.independent)

    # ------------------------------------------------------------------

    def critic(self, current: dict, iteration: int) -> CriticResult:
        """One adversarial pass against the current evidence state."""
        if self.dirty:
            self.refresh_evidence()

        rt = self.reports.get("red_team")
        status = redteam_execution_status(rt)

        if status.is_trustworthy:
            findings = findings_from_redteam_report(rt, symbol=self.symbol)
        else:
            # The agent did not complete. Structural review is still possible
            # from the numbers, and a degraded review beats none — but it is
            # labelled as degraded all the way to the reader.
            findings = deterministic_review(self.reports, symbol=self.symbol)
            if findings:
                status = ReviewExecutionStatus.COMPLETED_BY_FALLBACK
                log.warning("red team %s for %s — deterministic fallback review used",
                            redteam_execution_status(rt).value, self.symbol)

        # Research that was demanded and never arrived is itself an objection.
        # Without this the loop would treat an unanswered request as answered.
        if current.get("followup_failed"):
            findings = list(findings) + [Finding(
                objection=(f"Corroboration was demanded for {self.symbol} and could "
                           f"not be obtained: "
                           f"{current.get('followup_error', 'no agent could answer it')}."),
                severity=Severity.HIGH,
                test="The open question is still open. Treat the thesis as unsupported "
                     "on that point rather than assuming the answer.",
                category="unresolved_research",
                targets=[self.symbol],
            )]

        self.last_status = status
        self.rounds.append({
            "round": iteration + 1,
            "graph_version": self.graph_version,
            "evidence_nodes": len(self.graph.nodes),
            "effective_independent": round(self.independent, 3),
            "execution_status": status.value,
            "n_findings": len(findings),
            "objections": [f.objection[:160] for f in findings],
            "followup_agents": list(self.followup_agents),
        })
        return CriticResult(
            findings=findings,
            execution_status=status,
            graph_version=self.graph_version,
            evidence_nodes=len(self.graph.nodes),
        )

    def investigate(self, current: dict, questions: list[str]) -> dict:
        """Run bounded follow-up research, then mark the evidence stale.

        The orchestrator's investigator runs the agents *and* re-runs synthesis
        and the red team, so by the time this returns both the evidence and the
        adversary are new. Marking `dirty` makes the next round rebuild.
        """
        if self.investigator is None:
            current["followup_failed"] = True
            return current

        self.nodes_before_followup = len(self.graph.nodes)
        before = set(self.reports)
        updated = self.investigator(self.symbol, current, questions, self.plan)

        gained = sorted(set(self.reports) - before)
        refreshed = list(updated.get("followup_agents", []))
        self.followup_agents.extend(refreshed or gained)

        if not refreshed and not gained:
            # Nothing ran. Say so rather than letting the next round assume
            # the request was met.
            updated["followup_failed"] = True
            self.followup_failures.append("no agent produced new evidence")
        else:
            self.dirty = True
        return updated


def _bind_investigator(session: CandidateReview):
    """Adapt the review session to the loop's `investigator(idea, questions)`."""
    def follow_up(current: dict, questions: list[str]) -> dict:
        return session.investigate(current, questions)
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
    # Per-candidate review state: rounds, graph versions, follow-up agents.
    review_sessions: list = field(default_factory=list)

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
            # No candidates at all. That is not a review failure — there was
            # nothing to review — so `review_incomplete` stays False.
            return [], [], [], False, []

        # Note: the red-team report is deliberately *not* captured here.
        # `CandidateReview.critic` re-reads it every round, which is what makes
        # a post-follow-up round see a fresh adversary instead of a stale one.
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
        sessions: list[CandidateReview] = []
        review_incomplete = False
        plan_by_subject = {p.subject: p for p in (plans or [])}

        for kind, idea in ideas:
            symbol = idea.get("symbol", "?")
            candidate_id = idea.get("recommendation_id") or f"cand_{uuid.uuid4().hex[:12]}"

            # --- the review gate, applied BEFORE publication ---
            # The session owns the evidence state across rounds: a follow-up
            # rebuilds the graph and re-runs the adversary, so round 2 argues
            # against what round 1 bought rather than against a cached report.
            session = CandidateReview(
                self, reports, symbol, graph,
                plan_by_subject.get(symbol), investigator)
            session.candidate_id = candidate_id
            sessions.append(session)

            outcome = self.review_loop.run(
                dict(idea, recommendation_id=candidate_id),
                critic=session.critic,
                investigator=_bind_investigator(session) if investigator else None)
            outcomes.append(outcome)
            independent = session.independent

            if outcome.rejected:
                incomplete = (outcome.final_decision.status
                              is ReviewStatus.REVIEW_INCOMPLETE)
                rejected.append(RejectedCandidate(
                    subject=symbol,
                    candidate_id=candidate_id,
                    reason=outcome.final_decision.summary(),
                    findings=list(outcome.final_decision.reasons),
                    audit=outcome.audit_trail(),
                    original_confidence=int(idea.get("confidence", 0)),
                    status=(RecordStatus.SUPPRESSED if incomplete
                            else RecordStatus.REJECTED),
                    review_rounds=list(session.rounds),
                ))
                self.obs.event("review", f"{symbol} "
                                         f"{'suppressed' if incomplete else 'rejected'}: "
                                         f"{outcome.final_decision.summary()}",
                               level="warning" if incomplete else "info")
                if incomplete:
                    review_incomplete = True
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
            # The graph the *final* round actually saw, not the session's
            # opening one — after a follow-up these differ, and citing the
            # wrong version would make the audit trail lie.
            rec.evidence_graph_id = session.graph.investigation_id
            rec.graph_version = session.graph_version
            rec.review_decision_id = outcome.final_decision.id
            rec.review_status = outcome.final_decision.status.value
            rec.review_execution_status = session.last_status.value
            rec.review_rounds = list(session.rounds)
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

        return recommendations, rejected, outcomes, review_incomplete, sessions

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

        recs, rejected, outcomes, review_incomplete, sessions = \
            self.build_recommendations(
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

        # An active reviewed recommendation requires a successful adversarial
        # review. If the reviewer did not complete — and the deterministic
        # fallback could not stand in — nothing is published, and the reason
        # says so specifically rather than being laundered into "market
        # uncertainty". Infrastructure failure and genuine ignorance are
        # different states and the reader is entitled to know which one.
        if review_incomplete:
            statuses = {s.last_status.value for s in sessions
                        if not s.last_status.is_trustworthy}
            publication = PublicationSet.suppressed_set(
                "adversarial review did not complete "
                f"({', '.join(sorted(statuses)) or 'unknown'}) — an unreviewed "
                "recommendation is not publishable",
                mode=mode, rejected=rejected)
            self.obs.event("review", "publication suppressed: review incomplete",
                           level="error")
        else:
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
            review_sessions=sessions,
        )
