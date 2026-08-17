"""End-to-end tests for the MarketSwarm 2.0 layers.

The acceptance criteria this file enforces:
  * red-team findings change the published output
  * the event brain fires on anomalies and stays quiet otherwise
  * the chief selects a smaller team for quiet names than for events
  * evidence carries provenance and correlated evidence is discounted
  * the system can say NO_EDGE and INSUFFICIENT_EVIDENCE
  * a critically degraded run suppresses the output it cannot support
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from marketswarm.evidence.graph import (
    EvidenceGraph,
    EvidenceNode,
    NodeType,
    Relation,
)
from marketswarm.investigation.chief import ChiefInvestigator
from marketswarm.investigation.event_brain import EventBrain, EventPriority, EventType
from marketswarm.investigation.plan import PlanBudget, StopReason
from marketswarm.investigation.registry import REGISTRY, CapabilityRegistry
from marketswarm.pipeline2 import Pipeline2
from marketswarm.recommend.engine import (
    Conviction,
    RecommendationEngine,
    RecommendationInputs,
    RecommendationType,
)
from marketswarm.resilience import (
    BreakerState,
    CircuitBreaker,
    DegradationLevel,
    DegradationTracker,
)
from marketswarm.review.gate import Finding, Severity

from .test_pipeline import build_context, run_swarm


# ---------------- event brain ----------------

def test_brain_fires_on_a_large_gap_and_ignores_a_small_one():
    brain = EventBrain()
    big = brain.detect_gap("NVDA", -7.2, 2.4)
    assert big and big.priority is EventPriority.HIGH
    assert brain.detect_gap("AAPL", 0.2, 1.4) is None


def test_gap_is_normalised_by_the_symbols_own_volatility():
    """2% is an event on a quiet name and noise on a volatile one."""
    brain = EventBrain()
    quiet = brain.detect_gap("KO", 2.0, 0.8)      # 2.5 ATR
    wild = brain.detect_gap("TSLA", 2.0, 4.0)     # 0.5 ATR
    assert quiet is not None and quiet.priority.rank >= EventPriority.HIGH.rank
    assert wild is None


def test_brain_detects_the_full_event_set():
    brain = EventBrain()
    snap = {
        "market": {"vix": 31.0, "vix_change_pct": 22.0, "index_pct": -1.8,
                   "econ_events": [{"name": "CPI", "time_et": "08:30",
                                    "impact": "very high", "before_open": True}]},
        "symbols": {
            "NVDA": {"gap_pct": -7.0, "atr_pct": 2.2, "rvol": 5.0,
                     "put_call": 2.1,
                     "unusual": [{"kind": "put", "strike": 200, "volume": 9000,
                                  "open_interest": 1000, "turnover": 9.0,
                                  "notional_estimate": 5_000_000}],
                     "filings": [{"form": "8-K", "items": ["2.02"], "age_hours": 6}],
                     "earnings_reacting": True, "surprise_pct": -12.0},
            "AAPL": {"gap_pct": 0.1, "atr_pct": 1.5, "rvol": 1.0},
        },
    }
    tri = brain.triage(snap)
    kinds = {e.event_type for e in tri["events"]}
    for expected in (EventType.PRICE_GAP, EventType.VOLATILITY_SPIKE,
                     EventType.MACRO_RELEASE, EventType.OPTIONS_ANOMALY,
                     EventType.SEC_FILING, EventType.EARNINGS_REACTION,
                     EventType.RELATIVE_VOLUME):
        assert expected in kinds, f"missed {expected}"
    assert tri["highest"] is EventPriority.CRITICAL

    # AAPL is flat while the index is -1.8%, which is a genuine divergence
    # rather than quiet — being unchanged on a down day is information.
    aapl = tri["by_subject"].get("AAPL", [])
    assert [e.event_type for e in aapl] == [EventType.SECTOR_DIVERGENCE]
    assert "AAPL" not in tri["quiet"]


def test_quiet_market_produces_no_events():
    brain = EventBrain()
    tri = brain.triage({
        "market": {"vix": 14.0, "vix_change_pct": 0.5, "index_pct": 0.05},
        "symbols": {s: {"gap_pct": 0.1, "atr_pct": 1.5, "rvol": 1.0}
                    for s in ("AAPL", "MSFT", "KO")},
    })
    assert tri["n_events"] == 0
    assert len(tri["quiet"]) == 3


def test_options_anomaly_is_labelled_inferred():
    """Never claim inferred activity is observed trade flow."""
    brain = EventBrain()
    ev = brain.detect_options_anomaly(
        "NVDA", 1.9, [{"kind": "put", "strike": 100, "volume": 8000,
                       "open_interest": 900, "turnover": 8.9,
                       "notional_estimate": 3_000_000}])
    assert ev.evidence["inferred"] is True
    assert "not observed trade flow" in ev.evidence["caveat"]


# ---------------- chief investigator ----------------

def test_chief_spends_more_on_a_critical_event_than_a_quiet_name():
    chief = ChiefInvestigator()
    snap = {
        "market": {"vix": 15.0, "vix_change_pct": 1.0, "index_pct": 0.2},
        "symbols": {
            "NVDA": {"gap_pct": -8.0, "atr_pct": 2.0, "rvol": 6.0,
                     "filings": [{"form": "8-K", "items": ["2.02"], "age_hours": 4}]},
            "KO": {"gap_pct": 0.05, "atr_pct": 1.0, "rvol": 0.9},
        },
    }
    plans = {p.subject: p for p in chief.plan_session(snap)}
    nvda = plans["NVDA"]
    quiet = next(p for k, p in plans.items() if k.startswith("QUIET"))

    assert nvda.priority.rank > quiet.priority.rank
    assert nvda.budget.max_cost > quiet.budget.max_cost
    assert len(nvda.all_agents) > len(quiet.all_agents)
    assert "sec_filings" in nvda.all_agents
    assert "sec_filings" not in quiet.all_agents


def test_chief_always_includes_the_foundational_agents():
    chief = ChiefInvestigator()
    for plan in chief.plan_session({"market": {}, "symbols": {"KO": {"gap_pct": 0.0}}}):
        for essential in ("technicals", "cross_verify", "playbook", "red_team"):
            assert essential in plan.all_agents


def test_chief_prunes_impossible_hypotheses():
    """An earnings event rules out 'no identifiable catalyst'."""
    chief = ChiefInvestigator()
    plans = chief.plan_session({
        "market": {},
        "symbols": {"JPM": {"gap_pct": 3.0, "atr_pct": 1.5,
                            "earnings_reacting": True, "surprise_pct": 9.0}},
    })
    jpm = next(p for p in plans if p.subject == "JPM")
    assert not any("no identifiable catalyst" in h for h in jpm.hypotheses)


def test_budget_limits_are_enforced():
    b = PlanBudget(max_iterations=2, max_agents=3, max_cost=5.0, max_seconds=60)
    assert b.exceeded() is None
    b.iterations_used = 2
    assert b.exceeded() is StopReason.ITERATION_LIMIT
    b2 = PlanBudget(max_cost=5.0)
    b2.charge(cost=6.0)
    assert b2.exceeded() is StopReason.COST_LIMIT
    with pytest.raises(ValueError):
        PlanBudget(max_iterations=0)


def test_followup_depth_is_bounded():
    chief = ChiefInvestigator(max_followup_depth=1)
    plans = chief.plan_session({"market": {}, "symbols": {"KO": {"gap_pct": 0.0}}})
    parent = plans[0]
    assert chief.followup_plan(parent, ["check the filing"], depth=1) is not None
    assert chief.followup_plan(parent, ["check the filing"], depth=2) is None


def test_followup_routes_questions_to_the_right_agents():
    chief = ChiefInvestigator()
    plans = chief.plan_session({"market": {}, "symbols": {"KO": {"gap_pct": 0.0}}})
    child = chief.followup_plan(plans[0], ["was there an 8-K filing overnight?"])
    assert "sec_filings" in child.optional_agents


def test_registry_selects_a_smaller_team_for_fewer_event_types():
    reg = CapabilityRegistry()
    many, _ = reg.select({EventType.PRICE_GAP, EventType.EARNINGS_REACTION,
                          EventType.SEC_FILING, EventType.OPTIONS_ANOMALY,
                          EventType.MACRO_RELEASE})
    few, skipped = reg.select({EventType.QUIET})
    assert len(few) < len(many)
    assert skipped


def test_registry_pulls_in_declared_dependencies():
    reg = CapabilityRegistry()
    chosen, _ = reg.select({EventType.VOLATILITY_SPIKE})
    if "sentiment" in chosen:
        assert "options_flow" in chosen and "volatility_regime" in chosen


# ---------------- evidence graph ----------------

def test_correlated_evidence_is_discounted():
    """Five market-beta reads are not five independent confirmations."""
    g = EvidenceGraph()
    for i in range(5):
        g.add(EvidenceNode(claim=f"beta read {i}", cluster="market_beta"))
    correlated = g.effective_independent_count()

    g2 = EvidenceGraph()
    for cluster in ("market_beta", "company_fundamental", "macro",
                    "options_positioning", "insider"):
        g2.add(EvidenceNode(claim=f"{cluster} read", cluster=cluster))
    diverse = g2.effective_independent_count()

    assert correlated < 2.0, "five correlated reads should collapse"
    assert diverse > correlated * 1.5
    assert diverse <= 5.0


def test_provenance_chain_is_walkable():
    g = EvidenceGraph()
    raw = g.observe("SPY gapped +0.4%", cluster="market_beta", source="exchange_data")
    claim = g.add(EvidenceNode(claim="risk appetite is positive",
                               node_type=NodeType.CLAIM, cluster="market_beta"))
    g.link(claim, raw, Relation.DERIVES_FROM)
    chain = g.provenance_chain(claim)
    assert raw in chain


def test_contradictions_are_symmetric_and_reported():
    g = EvidenceGraph()
    a = g.observe("futures up", cluster="market_beta")
    b = g.observe("breadth negative", cluster="market_beta")
    g.link(a, b, Relation.CONTRADICTS)
    assert b in g.contradicting(a)
    assert a in g.contradicting(b)
    assert len(g.contradictions()) == 1


def test_explain_answers_both_questions():
    g = EvidenceGraph()
    thesis = g.hypothesize("NVDA continues higher", cluster="company_fundamental")
    sup = g.observe("earnings beat", cluster="company_fundamental", source="company_ir")
    con = g.observe("guidance cut", cluster="company_fundamental", source="sec_edgar")
    g.link(sup, thesis, Relation.SUPPORTS)
    g.link(con, thesis, Relation.CONTRADICTS)
    e = g.explain(thesis)
    assert any("earnings beat" in s for s in e["believe_because"])
    assert any("guidance cut" in s for s in e["would_be_invalidated_by"])


def test_no_counter_evidence_is_stated_not_implied():
    g = EvidenceGraph()
    t = g.hypothesize("thesis", cluster="technical")
    g.link(g.observe("support", cluster="technical"), t, Relation.SUPPORTS)
    e = g.explain(t)
    assert "absence of counter-evidence is not evidence of absence" in \
        " ".join(e["would_be_invalidated_by"])


def test_unknown_score_stays_unknown():
    n = EvidenceNode(claim="something with no source")
    assert n.score.composite() is None, "must not fabricate a default score"


def test_scores_stay_separate_until_used():
    n = EvidenceNode(claim="SEC filing")
    n.score.factual_reliability = 0.98
    n.score.predictive_utility = 0.30
    assert n.score.factual_reliability != n.score.predictive_utility
    c = n.score.composite()
    assert 0.30 < c < 0.98, "composite must sit between, not collapse to one"


# ---------------- recommendation engine ----------------

def _inputs(**kw):
    base = dict(subject="NVDA", direction="long", probability=0.58, confidence=60,
                expected_r=0.15, entry=100.0, target=103.0, stop=98.0,
                effective_independent_evidence=4.0)
    base.update(kw)
    return RecommendationInputs(**base)


def test_engine_says_insufficient_evidence():
    e = RecommendationEngine()
    rec = e.build(_inputs(effective_independent_evidence=1.0))
    assert rec.conviction is Conviction.INSUFFICIENT_EVIDENCE
    assert rec.rec_type is RecommendationType.INSUFFICIENT_EVIDENCE
    assert rec.direction is None
    assert not rec.actionable


def test_engine_says_no_edge_on_negative_expectancy():
    e = RecommendationEngine()
    rec = e.build(_inputs(expected_r=-0.05))
    assert rec.conviction is Conviction.NO_ACTIONABLE_EDGE
    assert rec.rec_type is RecommendationType.NO_EDGE


def test_engine_says_no_edge_on_a_coin_flip():
    e = RecommendationEngine()
    rec = e.build(_inputs(probability=0.50, expected_r=0.02))
    assert rec.conviction is Conviction.NO_ACTIONABLE_EDGE


def test_engine_flags_conflicting_evidence():
    e = RecommendationEngine()
    rec = e.build(_inputs(n_contradictions=4))
    assert rec.conviction is Conviction.CONFLICTING_EVIDENCE
    assert rec.rec_type is RecommendationType.INVESTIGATE


def test_engine_reaches_high_conviction_only_with_broad_evidence():
    e = RecommendationEngine()
    thin = e.build(_inputs(confidence=80, effective_independent_evidence=2.0))
    broad = e.build(_inputs(confidence=80, effective_independent_evidence=5.0))
    assert thin.conviction is not Conviction.HIGH_CONVICTION
    assert broad.conviction is Conviction.HIGH_CONVICTION


def test_missing_critical_agent_forces_insufficient_evidence():
    e = RecommendationEngine()
    rec = e.build(_inputs(missing_critical_agents=["technicals"]))
    assert rec.conviction is Conviction.INSUFFICIENT_EVIDENCE


def test_high_severity_red_team_downgrades_to_wait():
    e = RecommendationEngine()
    rec = e.build(_inputs(red_team_severity="high"))
    assert rec.rec_type is RecommendationType.WAIT_FOR_CONFIRMATION


def test_recommendation_separates_the_four_layers():
    rec = RecommendationEngine().build(_inputs())
    assert rec.observation and rec.interpretation and rec.prediction and rec.rationale
    assert rec.observation != rec.interpretation != rec.prediction
    assert rec.invalidation and rec.change_our_mind


def test_calibration_gap_is_disclosed_in_the_prediction():
    rec = RecommendationEngine().build(_inputs(historical_calibration_gap=0.12))
    assert "overconfident" in rec.prediction


# ---------------- resilience ----------------

def test_circuit_breaker_opens_and_recovers():
    b = CircuitBreaker("provider", failure_threshold=3, recovery_seconds=0.01,
                       half_open_successes=1)
    for _ in range(3):
        b.record_failure()
    assert b.state is BreakerState.OPEN
    assert not b.allow()

    import time
    time.sleep(0.02)
    assert b.allow()                      # half-open probe
    b.record_success()
    assert b.state is BreakerState.CLOSED


def test_losing_a_critical_agent_makes_the_run_unusable():
    d = DegradationTracker()
    d.record_agent("technicals", "failed")
    assert d.level is DegradationLevel.UNUSABLE
    assert not d.level.can_publish_ideas
    assert "all directional ideas" in d.suppressed_outputs()


def test_losing_options_suppresses_only_option_ideas():
    d = DegradationTracker()
    d.record_agent("options_flow", "failed")
    assert d.level is DegradationLevel.CRITICALLY_DEGRADED
    assert "option ideas" in d.suppressed_outputs()
    assert "all directional ideas" not in d.suppressed_outputs()


def test_missing_evidence_is_stated_explicitly():
    d = DegradationTracker()
    d.record_agent("breaking_news", "failed")
    d.record_provider("rss")
    stmt = d.missing_evidence_statement()
    assert "breaking_news" in stmt and "rss" in stmt
    assert d.data_quality() == "degraded"


def test_healthy_run_says_nothing():
    assert DegradationTracker().missing_evidence_statement() == ""


# ---------------- full integration ----------------

@pytest.fixture
def swarm_reports(tmp_path):
    ctx = build_context(tmp_path, gaps={"NVDA": 4.2, "SPY": 0.5, "AAPL": -0.3})
    return asyncio.run(run_swarm(ctx)), ctx


def test_pipeline2_runs_end_to_end(swarm_reports):
    reports, ctx = swarm_reports
    result = Pipeline2(ctx.config).run(reports, ctx.run_date)

    assert result.graph.summary()["n_nodes"] > 5
    assert result.plans, "the chief produced no plans"
    assert result.recommendations or result.rejected, "no output at all"
    summary = result.summary()
    assert "by_conviction" in summary
    assert summary["evidence"]["effective_independent"] <= summary["evidence"]["n_nodes"]


def test_pipeline2_evidence_carries_provenance(swarm_reports):
    reports, ctx = swarm_reports
    result = Pipeline2(ctx.config).run(reports, ctx.run_date)
    sourced = [n for n in result.graph.nodes.values() if n.source]
    assert sourced
    assert all(n.provenance for n in sourced)
    assert all(n.score.independence is not None for n in result.graph.nodes.values())


def test_red_team_findings_reach_the_published_output(swarm_reports):
    """The 1.x defect, now covered: criticism must be able to change the result."""
    reports, ctx = swarm_reports
    result = Pipeline2(ctx.config).run(reports, ctx.run_date)

    changed = [o for o in result.review_outcomes
               if o.final_decision.confidence_delta != 0 or o.rejected]
    assert changed, "the red team ran but changed nothing — this is the 1.x bug"

    for rec in result.recommendations:
        assert rec.review_status is not None


def test_severe_finding_rejects_an_idea_in_the_full_pipeline(swarm_reports):
    reports, ctx = swarm_reports

    class _Critical:
        agent = "red_team"
        status = "ok"
        evidence: list = []
        signals: list = []
        duration_ms = 1
        data = {"objections": [{
            "severity": "critical",
            "objection": "the underlying already reported and gapped the other way",
            "test": "check the tape",
        }]}
        usable = True

    reports = dict(reports)
    reports["red_team"] = _Critical()
    result = Pipeline2(ctx.config).run(reports, ctx.run_date)

    assert result.rejected, "a critical objection must remove ideas"
    assert not result.recommendations or all(
        not r.actionable for r in result.recommendations)


def test_critically_degraded_run_suppresses_unsupported_output(swarm_reports):
    reports, ctx = swarm_reports
    reports = dict(reports)

    class _Dead:
        agent = "technicals"
        status = "failed"
        usable = False
        data: dict = {}
        evidence: list = []
        signals: list = []
        error = "provider down"
        duration_ms = 0
    reports["technicals"] = _Dead()

    result = Pipeline2(ctx.config).run(reports, ctx.run_date)
    assert result.degradation["level"] == DegradationLevel.UNUSABLE.value
    assert not result.degradation["can_publish_ideas"]
    for rec in result.recommendations:
        assert rec.conviction is Conviction.INSUFFICIENT_EVIDENCE
