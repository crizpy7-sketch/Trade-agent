"""End-to-end control-path tests.

Unit tests prove a component is correct. These prove it is *reached*. Every
test here drives the real `Swarm.run()` with fake providers and asserts on what
the production path actually did — which agents were instantiated, what reached
the report, what reached the database.

The distinction matters because the defect these tests exist to prevent was
never a broken component. The review gate worked perfectly and was bypassed;
the Chief Investigator planned correctly, after every agent had already run.
Both passed their unit tests throughout.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json

import pytest

import marketswarm.orchestrator as orch
from marketswarm.agents import ALL_AGENTS
from marketswarm.config import Config
from marketswarm.publication import PublicationError, PublicationSet, RecordStatus
from marketswarm.recommend.engine import Conviction, Recommendation, RecommendationType
from marketswarm.report import render_html, render_markdown

from .fakes import FakeEarnings, FakeEcon, FakeEdgar, FakeMarket, FakeNews, FakeOptions

UNIVERSE = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMD"]
RUN_DATE = dt.date(2026, 8, 17)          # a Monday

QUIET = None
EVENTFUL = {"NVDA": 3.4, "AAPL": -2.9, "SPY": 0.3, "QQQ": 0.4, "MSFT": 1.1, "AMD": -2.2}


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

@pytest.fixture
def swarm_factory(tmp_path, monkeypatch):
    """Build a real `Swarm` whose providers are fakes.

    Everything else — the orchestrator, the routing, the review gate, the
    persistence — is the production code path.
    """
    def build(gaps=QUIET, **cfg_kwargs):
        market = FakeMarket(UNIVERSE, gaps=gaps)
        monkeypatch.setattr(orch, "MarketData", lambda c: market)
        monkeypatch.setattr(orch, "OptionsData", lambda c: FakeOptions(market))
        monkeypatch.setattr(orch, "NewsData", lambda c, universe=None: FakeNews())
        monkeypatch.setattr(orch, "EconData", lambda c, k: FakeEcon())
        monkeypatch.setattr(orch, "EdgarData", lambda c, ua: FakeEdgar())
        monkeypatch.setattr(orch, "EarningsData", lambda c: FakeEarnings())

        cfg = Config()
        cfg.universe = list(UNIVERSE)
        cfg.index_symbols = ["SPY", "QQQ"]
        cfg.data_dir = tmp_path
        cfg.report_dir = tmp_path / "reports"
        for k, v in cfg_kwargs.items():
            setattr(cfg, k, v)
        cfg.ensure_dirs()
        return orch.Swarm(cfg)

    return build


@pytest.fixture
def instantiated(monkeypatch):
    """Records every agent class actually constructed during a run.

    Instantiation, not planning: a test that only inspected the plan would have
    passed against the broken code, which is precisely the failure mode.
    """
    seen: list[str] = []
    for cls in ALL_AGENTS:
        original = cls.__init__

        def spy(self, *a, _orig=original, _name=cls.name, **kw):
            seen.append(_name)
            _orig(self, *a, **kw)

        monkeypatch.setattr(cls, "__init__", spy)
    return seen


def force_critical_objection(monkeypatch, symbol: str = "NVDA"):
    """Make the red team raise a CRITICAL finding against one symbol."""
    from marketswarm.agents.redteam import RedTeamAgent

    original = RedTeamAgent.run

    async def with_critical(self, ctx):
        rep = await original(self, ctx)
        rep.data.setdefault("objections", []).insert(0, {
            "severity": "critical",
            "objection": f"{symbol} thesis rests on a data error — the gap is a "
                         f"stale quote, not a real move.",
            "test": "Compare the pre-market print against the consolidated tape.",
        })
        rep.data["high_severity"] = rep.data.get("high_severity", 0) + 1
        return rep

    monkeypatch.setattr(RedTeamAgent, "run", with_critical)


def run(swarm, **kw):
    return asyncio.run(swarm.run(RUN_DATE, force=True, **kw))


# ==========================================================================
# TEST 1 — a rejected recommendation cannot re-enter publication
# ==========================================================================

def test_rejected_recommendation_can_never_reenter_publication_path(
        swarm_factory, monkeypatch):
    """The single most important invariant in the system.

    A candidate the review gate rejects must be absent from every active
    output while remaining visible as audit history. This test fails against
    the pre-fix orchestrator, which rebuilt `result.ideas` from the raw
    playbook after Pipeline2 had already removed the rejection.
    """
    force_critical_objection(monkeypatch, "NVDA")
    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm)

    pub = result.publication
    assert pub.rejected, "the forced CRITICAL objection rejected nothing"
    rejected_subjects = pub.rejected_subjects()

    # 1. not in the authoritative set
    assert not (rejected_subjects & pub.active_subjects())

    # 2. not in the legacy ideas view
    for bucket in result.ideas.values():
        for idea in bucket:
            assert idea["symbol"] not in rejected_subjects

    # 3. not rendered as an idea card in the Markdown report
    md = render_markdown(result)
    cards = [line for line in md.splitlines() if line.startswith("### ")]
    for sym in rejected_subjects:
        assert not any(sym in c for c in cards), \
            f"{sym} was rejected but rendered an idea card"

    # 4. nor as an HTML idea card (the review-gate audit section legitimately
    #    names them — that is where rejected candidates are supposed to appear)
    html = render_html(result)
    before_audit = html.split("Review gate")[0]
    for sym in rejected_subjects:
        assert f"<h3>{sym}" not in before_audit
        assert sym in html, "the rejection must still be auditable in the report"

    # 5. not in the JSON active recommendations
    payload = json.loads(json.dumps(
        {"ideas": result.ideas,
         "recommendations": [r.to_dict() for r in pub.active()]}, default=str))
    for rec in payload["recommendations"]:
        assert rec["subject"] not in rejected_subjects

    # 6. not persisted as an active prediction — the monitor and the scorer
    #    both read this table, so this covers them too
    rows = swarm.store.conn.execute(
        "SELECT symbol FROM predictions WHERE run_date = ?",
        (RUN_DATE.isoformat(),)).fetchall()
    persisted = {r[0] for r in rows}
    assert not (persisted & rejected_subjects)

    # 7. not an active recommendation row
    active_rows = swarm.store.conn.execute(
        "SELECT subject FROM recommendations WHERE status IN ('APPROVED','MODIFIED')"
    ).fetchall()
    assert not ({r[0] for r in active_rows} & rejected_subjects)

    # 8. the webhook payload
    from marketswarm.notify import summarize
    text = summarize(result)
    for sym in rejected_subjects:
        assert f"• {sym}" not in text

    # ---- but the audit trail must exist ----
    audit = swarm.store.conn.execute(
        "SELECT subject, status, rejection_reason, review_audit FROM recommendations "
        "WHERE status = 'REJECTED'").fetchall()
    assert audit, "a rejected candidate must remain auditable"
    assert {r[0] for r in audit} == rejected_subjects
    for _, status, reason, trail in audit:
        assert status == RecordStatus.REJECTED.value
        assert reason
        assert json.loads(trail), "the review audit trail must be stored"


def test_publication_set_refuses_to_hold_a_rejected_candidate():
    """The invariant is also enforced structurally, not only by the caller."""
    from marketswarm.publication import RejectedCandidate

    rec = Recommendation(
        id="cand_1", subject="NVDA", rec_type=RecommendationType.FAVORABLE,
        conviction=Conviction.HIGH_CONVICTION, direction="long",
        forecast_probability=0.6, confidence=70, original_confidence=70,
        expected_r=0.3, source_kind="stock")
    pub = PublicationSet(
        approved=[rec],
        rejected=[RejectedCandidate(subject="NVDA", candidate_id="cand_1",
                                    reason="critical finding")])
    with pytest.raises(PublicationError):
        pub.assert_no_leak()


def test_result_ideas_cannot_be_assigned():
    """`result.ideas` is derived. The pre-fix bug was a single assignment to
    it, so the attribute must not accept one."""
    result = orch.SwarmResult(run_date=RUN_DATE, market_open=True)
    with pytest.raises(AttributeError):
        result.ideas = {"calls": [{"symbol": "NVDA"}], "puts": [], "stocks": []}


# ==========================================================================
# TEST 2 — reduced confidence survives to every consumer
# ==========================================================================

def test_reduced_confidence_reaches_every_downstream_consumer():
    """A gate that reduces confidence 80 → 55 must be believed everywhere."""
    rec = Recommendation(
        id="rec_reduced", subject="MSFT", rec_type=RecommendationType.FAVORABLE,
        conviction=Conviction.MODERATE_CONVICTION, direction="long",
        forecast_probability=0.58, confidence=55, original_confidence=80,
        expected_r=0.25, entry=100.0, target=103.0, stop=98.5,
        rationale="reduced by review", invalidation=["through 98.50"],
        review_status="APPROVE_WITH_REDUCED_CONFIDENCE", source_kind="stock",
        # A stale pre-review confidence planted in the presentation payload:
        # the derived view must overwrite it, not carry it.
        source_payload={"confidence": 80, "probability": 0.72, "clears_bar": True},
    )
    pub = PublicationSet(approved=[rec])
    pub.assert_no_leak()

    idea = pub.ideas_view()["stocks"][0]
    assert idea["confidence"] == 55
    assert idea["probability"] == pytest.approx(0.58)
    assert idea["original_confidence"] == 80

    result = orch.SwarmResult(run_date=RUN_DATE, market_open=True, publication=pub)
    assert result.ideas["stocks"][0]["confidence"] == 55

    md = render_markdown(result)
    assert "55/100" in md, "the reduced confidence never reached the report"
    assert "80/100" not in md, "the pre-review confidence leaked into the report"

    from marketswarm.notify import summarize
    assert "P 58%" in summarize(result)


def test_gate_confidence_reduction_is_applied_in_the_real_pipeline(swarm_factory):
    """Not a hand-built object: the real run must show a reduction too."""
    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm)
    assert result.v2 is not None

    # Every candidate is either reduced or rejected — the red team objects on
    # this fixture, so nothing may pass through at full confidence.
    outcomes = result.v2.review_outcomes
    assert outcomes, "no candidate reached the review gate"
    for outcome in outcomes:
        assert outcome.rejected or outcome.confidence_change < 0, (
            "a candidate survived a red-team objection with its confidence intact")

    for rec in result.publication.active():
        assert rec.confidence <= rec.original_confidence


# ==========================================================================
# TEST 3 — a modified recommendation replaces the original
# ==========================================================================

def test_modified_recommendation_replaces_the_original(swarm_factory, monkeypatch):
    """FAVORABLE downgraded to WATCH must leave no FAVORABLE behind."""
    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm)
    pub = result.publication

    # Every symbol appears exactly once in the active set.
    subjects = [r.subject for r in pub.active()]
    assert len(subjects) == len(set(subjects)), \
        "the same subject survived twice — original and modified both published"

    # Anything downgraded away from a directional type is not in the trade view.
    non_directional = {r.subject for r in pub.active() if not r.actionable}
    traded = {i["symbol"] for b in result.ideas.values() for i in b}
    assert not (non_directional & traded)


# ==========================================================================
# TEST 4 & 7 — dynamic routing actually prevents execution
# ==========================================================================

def test_dynamic_routing_skips_agents_that_never_execute(swarm_factory, instantiated):
    """A quiet market must not construct the expensive specialists.

    Asserts on instantiation, so a plan that merely *says* it skipped an agent
    cannot satisfy this test.
    """
    swarm = swarm_factory(gaps=QUIET)
    result = run(swarm)

    executed = set(instantiated)
    assert result.trace.orchestration_mode == "dynamic"
    assert result.trace.agents_skipped > 0, "nothing was skipped on a quiet market"

    # The foundational scan and the decision agents always run.
    for name in orch.SCAN_AGENTS + orch.DECISION_AGENTS:
        assert name in executed, f"{name} must always run"

    # Whatever the trace claims was skipped was genuinely never constructed.
    skipped = {n for n, why in result.trace.agent_execution_reason.items()
               if why.startswith("skipped")}
    assert skipped, "no agent was reported skipped"
    assert not (skipped & executed), \
        f"agents reported skipped but actually executed: {skipped & executed}"

    # And the expensive ones are among them on a quiet tape.
    assert {"sec_filings", "institutional"} & skipped


def test_quiet_symbol_costs_materially_less_than_full_swarm(swarm_factory, instantiated):
    """Dynamic mode must actually save work, not merely reorder it."""
    swarm = swarm_factory(gaps=QUIET)
    quiet = run(swarm)
    dynamic_agents = len(set(instantiated))

    instantiated.clear()
    swarm_full = swarm_factory(gaps=QUIET)
    full = run(swarm_full, mode="full")
    full_agents = len(set(instantiated))

    assert dynamic_agents < full_agents, \
        f"dynamic ran {dynamic_agents}, full ran {full_agents} — no saving"
    assert full_agents == len(ALL_AGENTS)

    from marketswarm.investigation.registry import REGISTRY
    dyn_cost = REGISTRY.estimated_cost(
        [n for n, w in quiet.trace.agent_execution_reason.items()
         if not w.startswith("skipped")])
    all_cost = REGISTRY.estimated_cost([a.name for a in ALL_AGENTS])
    assert dyn_cost < all_cost * 0.8, \
        f"dynamic cost {dyn_cost:.1f} vs full {all_cost:.1f} — not a material saving"
    assert full.trace.agents_skipped == 0


# ==========================================================================
# TEST 5 — a critical event expands the swarm, within budget
# ==========================================================================

def test_critical_event_expands_the_swarm_within_budget(swarm_factory, instantiated):
    swarm_quiet = swarm_factory(gaps=QUIET)
    run(swarm_quiet)
    quiet_agents = set(instantiated)

    instantiated.clear()
    swarm_busy = swarm_factory(gaps=EVENTFUL)
    busy = run(swarm_busy)
    busy_agents = set(instantiated)

    assert len(busy_agents) > len(quiet_agents), \
        "a market full of gaps recruited no additional specialists"
    assert busy.trace.event_count > 0
    assert busy.trace.investigation_count > 0

    # Budgets hold: the swarm can grow, but not without limit.
    assert len(busy_agents) <= len(ALL_AGENTS)
    from marketswarm.investigation.chief import PRIORITY_BUDGETS
    ceiling = max(b["max_agents"] for b in PRIORITY_BUDGETS.values())
    assert busy.trace.agents_selected <= max(ceiling, len(ALL_AGENTS))


def test_chief_plans_before_specialists_execute(swarm_factory, monkeypatch):
    """Ordering, asserted directly. The pre-fix code planned last."""
    order: list[str] = []

    from marketswarm.investigation.chief import ChiefInvestigator
    original_plan = ChiefInvestigator.plan_session

    def spy_plan(self, snapshot, agent_weights=None):
        order.append("PLAN")
        return original_plan(self, snapshot, agent_weights)

    monkeypatch.setattr(ChiefInvestigator, "plan_session", spy_plan)

    from marketswarm.agents.news_agents import SECFilingsAgent
    original_run = SECFilingsAgent.run

    async def spy_run(self, ctx):
        order.append("SPECIALIST")
        return await original_run(self, ctx)

    monkeypatch.setattr(SECFilingsAgent, "run", spy_run)

    run(swarm_factory(gaps=EVENTFUL))
    assert "PLAN" in order, "the chief never planned"
    if "SPECIALIST" in order:
        assert order.index("PLAN") < order.index("SPECIALIST"), \
            "a routed specialist ran before the plan that selects it"


# ==========================================================================
# TEST 6 — request-more-research performs a real bounded follow-up
# ==========================================================================

def test_request_more_research_executes_a_real_followup(swarm_factory, monkeypatch):
    """The gate asks for evidence; agents must actually run to get it."""
    from marketswarm.review.gate import (ReviewDecision, ReviewExecutionStatus,
                                         ReviewGate, ReviewStatus, Severity)

    calls = {"n": 0}
    original_review = ReviewGate.review

    def demanding(self, findings, original_confidence, iteration=0,
                  max_iterations=2, evidence_already_requested=False,
                  execution_status=ReviewExecutionStatus.COMPLETED):
        # First pass always demands research, so the follow-up path is taken.
        if iteration == 0 and not evidence_already_requested:
            calls["n"] += 1
            return ReviewDecision(
                status=ReviewStatus.REQUEST_MORE_RESEARCH,
                original_confidence=original_confidence,
                revised_confidence=max(0, original_confidence - 10),
                severity=Severity.MEDIUM,
                reasons=["needs corroboration from the filing record"],
                required_followups=["check the SEC filing record for an 8-K",
                                    "corroborate the headline with a second source"],
                iteration=iteration)
        return original_review(self, findings, original_confidence, iteration,
                               max_iterations, evidence_already_requested,
                               execution_status)

    monkeypatch.setattr(ReviewGate, "review", demanding)

    # Quiet market, so sec_filings and breaking_news are NOT routed in
    # normally — any execution of them proves the follow-up did it.
    swarm = swarm_factory(gaps=QUIET)
    result = run(swarm)

    assert calls["n"] > 0, "the gate never demanded research"

    followed_up = [n for n, why in result.trace.agent_execution_reason.items()
                   if why.startswith("follow-up")]
    assert followed_up, "research was demanded but no agent ran to answer it"

    # The evidence graph grew because of it.
    assert result.v2 is not None
    assert len(result.v2.graph.nodes) > 0

    # The iteration cap held.
    assert result.trace.review_iterations <= 2 * max(
        1, result.trace.recommendations_candidate)

    # And the recommendations that came back carry the open questions.
    assert any(r.open_questions for r in result.publication.active()) or \
        result.publication.rejected


# ==========================================================================
# TEST 8 — production learning changes the next run
# ==========================================================================

def test_learning_from_session_a_changes_session_b(swarm_factory):
    """Two runs with a scoring pass between them, asserting on behaviour.

    Not "a row was written": the assertion is that the weight the next run
    actually resolves for an agent has moved, and moved in the direction the
    evidence implies.
    """
    from marketswarm.closed_loop import ClosedLoop
    from marketswarm.memory.contribution import MIN_OBSERVATIONS
    from marketswarm.memory.resolver import ContextualWeightResolver
    from marketswarm.memory import Prediction

    swarm = swarm_factory(gaps=EVENTFUL)
    conn = swarm.store.conn

    before = ContextualWeightResolver(conn).get_agent_weight(
        "technicals", regime="quiet_trend")

    # Session A: resolved history in which `technicals` consistently pushed
    # the probability the right way and `sentiment` consistently pushed it the
    # wrong way.
    run_id = swarm.store.start_run(RUN_DATE.isoformat(), "premarket")
    for i in range(MIN_OBSERVATIONS * 3):
        win = i % 2 == 0
        pid = swarm.store.record_prediction(Prediction(
            run_date=RUN_DATE.isoformat(), kind="stock_setup", symbol="SPY",
            direction="long", probability=0.62 if win else 0.38,
            entry=100, target=103, stop=98,
            features={"regime": "quiet_trend", "setup": "stock_setup"},
            contributing_agents={"technicals": 0.45 if win else -0.45,
                                 "sentiment": -0.40 if win else 0.40},
        ), run_id)
        swarm.store.resolve(pid, 1 if win else 0, 1.0 if win else -1.0)

    cycle = ClosedLoop(conn).run()
    assert cycle.resolved_reviewed > 0, "the closed loop reviewed nothing"
    assert cycle.contributions_updated > 0, "no contextual weight was updated"

    # Session B: the resolver now answers from measured contribution.
    resolver = ContextualWeightResolver(conn)
    after = resolver.get_agent_weight("technicals", regime="quiet_trend")
    sentiment_after = resolver.get_agent_weight("sentiment", regime="quiet_trend")

    assert after != before, "the next run would resolve an unchanged weight"
    assert after > sentiment_after, (
        f"the agent that helped ({after:.2f}) is not weighted above the one "
        f"that hurt ({sentiment_after:.2f})")
    assert "contextual" in resolver.provenance()["technicals"]

    # And the orchestrator consumes exactly that resolver on its next run.
    weights = swarm._resolve_agent_weights()
    assert weights["technicals"] == pytest.approx(
        resolver.get_agent_weight("technicals", regime=swarm._last_regime()))


def test_after_action_review_runs_as_part_of_scoring(swarm_factory):
    """It must not be a manual side-car."""
    from marketswarm.closed_loop import ClosedLoop
    from marketswarm.memory import Prediction

    swarm = swarm_factory()
    conn = swarm.store.conn
    run_id = swarm.store.start_run(RUN_DATE.isoformat(), "premarket")
    pid = swarm.store.record_prediction(Prediction(
        run_date=RUN_DATE.isoformat(), kind="stock_setup", symbol="NVDA",
        direction="long", probability=0.78, confidence=80,
        entry=100, target=103, stop=98,
        features={"regime": "high_vol"},
        contributing_agents={"options_flow": 0.5}), run_id)
    swarm.store.resolve(pid, 0, -1.0)

    assert conn.execute("SELECT COUNT(*) FROM after_action_reviews").fetchone()[0] == 0
    result = ClosedLoop(conn).run()
    assert result.resolved_reviewed == 1
    assert conn.execute("SELECT COUNT(*) FROM after_action_reviews").fetchone()[0] == 1

    # Running twice must not learn from the same outcome twice.
    again = ClosedLoop(conn).run()
    assert again.resolved_reviewed == 0
    assert conn.execute("SELECT COUNT(*) FROM after_action_reviews").fetchone()[0] == 1


# ==========================================================================
# TEST 9 — mistake memory is retrieved for a future investigation
# ==========================================================================

def test_prior_mistake_is_retrieved_for_a_later_investigation(swarm_factory):
    from marketswarm.memory.institutional import (InstitutionalMemory, Memory,
                                                  MemoryCategory, MistakeTaxonomy)

    swarm = swarm_factory(gaps=EVENTFUL)
    mem = InstitutionalMemory(swarm.store.conn)
    mem.remember(Memory(
        category=MemoryCategory.FAILURE,
        title="NVDA: double-counted correlated signals",
        body=("Four bullish signals were really one — index beta counted four "
              "times. Confidence was inflated and the trade lost."),
        subject="NVDA",
        confidence=0.8, evidence_n=6,
        source="after_action_review",
        related_entities=["NVDA"],
        provenance=MistakeTaxonomy.OVERWEIGHTED_CORRELATED.value,
    ))

    recalled = mem.recall(subject="NVDA", limit=3)
    assert recalled, "the memory store cannot recall its own entry"

    # The orchestrator supplies it to the decision layer.
    from marketswarm.agents import SwarmContext
    ctx = SwarmContext(run_date=RUN_DATE, prev_session=RUN_DATE,
                       universe=UNIVERSE, index_symbols=["SPY"],
                       market=None, options=None, news=None, econ=None,
                       edgar=None, earnings=None, config=swarm.config)
    priors = swarm._prior_failures(ctx)
    assert "NVDA" in priors
    assert any("double-counted" in p for p in priors["NVDA"])

    # And it reaches the recommendation as a stated risk.
    result = run(swarm)
    nvda = [r for r in result.publication.active() if r.subject == "NVDA"]
    if nvda:
        assert any("prior failure" in risk.lower() for risk in nvda[0].key_risks), \
            "a recorded prior failure did not surface in the recommendation's risks"


# ==========================================================================
# TEST 10 — the Research Scientist cannot self-deploy
# ==========================================================================

def test_research_scientist_cannot_promote_itself():
    """Enforced structurally: the class has no promotion method at all."""
    from marketswarm.experiments.scientist import ResearchScientist

    forbidden = ("promote", "deploy", "apply_to_production", "set_champion",
                 "activate", "install")
    for name in forbidden:
        assert not hasattr(ResearchScientist, name), \
            f"ResearchScientist.{name} exists — it must not be able to deploy"

    # No executable line may call a promotion method.
    import inspect
    for line in inspect.getsource(ResearchScientist).splitlines():
        code = line.split("#")[0]
        if '"""' in code or code.strip().startswith(("'", '"')):
            continue
        assert ".promote(" not in code, f"the scientist promotes: {line.strip()}"


def test_closed_loop_registers_challengers_but_never_promotes(swarm_factory):
    from marketswarm.closed_loop import ClosedLoop
    from marketswarm.experiments.lab import ExperimentLab
    from marketswarm.memory import Prediction

    swarm = swarm_factory()
    conn = swarm.store.conn
    run_id = swarm.store.start_run(RUN_DATE.isoformat(), "premarket")

    # Enough recurring, avoidable failures to justify a proposal.
    for i in range(40):
        pid = swarm.store.record_prediction(Prediction(
            run_date=RUN_DATE.isoformat(), kind="stock_setup", symbol="AMD",
            direction="long", probability=0.85, confidence=85,
            entry=100, target=103, stop=98,
            features={"regime": "high_vol"},
            contributing_agents={"sentiment": 0.6}), run_id)
        swarm.store.resolve(pid, 0, -1.0)

    ClosedLoop(conn).run()

    lab = ExperimentLab(conn)
    champion_before = _champion_of(conn)
    promoted = conn.execute(
        "SELECT COUNT(*) FROM experiments WHERE status = 'PROMOTED'").fetchone()[0]
    assert promoted == 0, "the closed loop promoted an experiment without a human"
    assert _champion_of(conn) == champion_before
    assert hasattr(lab, "promote"), "promotion must exist — on the lab, gated"


def _champion_of(conn):
    row = conn.execute(
        "SELECT COUNT(*) FROM experiments WHERE status='PROMOTED'").fetchone()
    return row[0] if row else 0


# ==========================================================================
# FINAL ADVERSARIAL GAUNTLET
# ==========================================================================

def test_scenario_a_red_team_rejects_everything(swarm_factory, monkeypatch):
    """Nothing may leak when every candidate is rejected."""
    from marketswarm.review.gate import (ReviewDecision, ReviewExecutionStatus,
                                         ReviewGate, ReviewStatus, Severity)

    def reject_all(self, findings, original_confidence, iteration=0,
                   max_iterations=2, evidence_already_requested=False,
                   execution_status=None):
        return ReviewDecision(
            status=ReviewStatus.REJECT, original_confidence=original_confidence,
            revised_confidence=0, severity=Severity.CRITICAL,
            reasons=["rejected for the purposes of this test"], iteration=iteration)

    monkeypatch.setattr(ReviewGate, "review", reject_all)
    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm)

    assert result.publication.approved == []
    assert result.ideas == {"calls": [], "puts": [], "stocks": []}
    assert result.publication.rejected
    assert swarm.store.conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE run_date = ?",
        (RUN_DATE.isoformat(),)).fetchone()[0] == 0

    md = render_markdown(result)
    assert "Review gate" in md
    from marketswarm.notify import summarize
    assert "•" not in summarize(result)


def test_scenario_b_pipeline_crash_publishes_nothing(swarm_factory, monkeypatch):
    """A crash mid-review must not fall back to unreviewed 1.x ideas.

    This is the fallback rule: a broken decision layer means no recommendation,
    never the pre-review one.
    """
    from marketswarm.pipeline2 import Pipeline2

    def explode(self, *a, **kw):
        raise RuntimeError("simulated failure inside the review layer")

    monkeypatch.setattr(Pipeline2, "run", explode)
    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm)

    assert result.v2 is None
    assert result.publication.suppressed
    assert "review layer failed" in result.publication.suppression_reason
    assert result.ideas == {"calls": [], "puts": [], "stocks": []}
    assert result.publication.active() == []

    # Nothing persisted as an active idea, including the index call.
    assert swarm.store.conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE run_date = ?",
        (RUN_DATE.isoformat(),)).fetchone()[0] == 0

    # The report says so out loud rather than looking like a quiet day.
    md = render_markdown(result)
    assert "Recommendations withheld" in md
    assert "No recommendation is published" in md


def test_scenario_c_selected_three_agents_means_three_agents(swarm_factory,
                                                             instantiated,
                                                             monkeypatch):
    """When the plan is minimal, the swarm must be minimal."""
    from marketswarm.investigation.chief import ChiefInvestigator

    def minimal(self, snapshot, agent_weights=None):
        return []

    monkeypatch.setattr(ChiefInvestigator, "plan_session", minimal)
    swarm = swarm_factory(gaps=QUIET)
    result = run(swarm)

    executed = set(instantiated)
    allowed = set(orch.SCAN_AGENTS) | set(orch.DECISION_AGENTS)
    assert executed <= allowed, f"agents ran with no plan selecting them: {executed - allowed}"
    assert result.trace.agents_skipped == len(ALL_AGENTS) - len(allowed)


def test_scenario_d_contextual_learning_separates_regimes(swarm_factory):
    """An agent good in one regime and bad in another must not be averaged."""
    from marketswarm.closed_loop import ClosedLoop
    from marketswarm.memory import Prediction
    from marketswarm.memory.contribution import MIN_OBSERVATIONS
    from marketswarm.memory.resolver import ContextualWeightResolver

    swarm = swarm_factory()
    conn = swarm.store.conn
    run_id = swarm.store.start_run(RUN_DATE.isoformat(), "premarket")

    for regime, helps in (("quiet_trend", True), ("high_vol", False)):
        for i in range(MIN_OBSERVATIONS * 3):
            win = i % 2 == 0
            lo = (0.5 if win else -0.5) if helps else (-0.5 if win else 0.5)
            pid = swarm.store.record_prediction(Prediction(
                run_date=RUN_DATE.isoformat(), kind="stock_setup", symbol="SPY",
                direction="long", probability=0.6 if win else 0.4,
                entry=100, target=103, stop=98,
                features={"regime": regime},
                contributing_agents={"options_flow": lo}), run_id)
            swarm.store.resolve(pid, 1 if win else 0, 1.0 if win else -1.0)

    ClosedLoop(conn).run()
    r = ContextualWeightResolver(conn)
    quiet = r.get_agent_weight("options_flow", regime="quiet_trend")
    stressed = r.get_agent_weight("options_flow", regime="high_vol")
    assert quiet != stressed, (
        "one weight for both regimes — contextual learning collapsed to a global "
        f"average ({quiet:.3f})")
    assert quiet > stressed


def test_scenario_g_no_llm_still_produces_a_decision(swarm_factory):
    """Every gate with teeth is deterministic; no API key must change nothing."""
    swarm = swarm_factory(gaps=EVENTFUL, anthropic_api_key=None, llm_enabled=False)
    result = run(swarm)
    assert result.trace.agents_executed > 0
    assert result.v2 is not None
    assert not result.publication.suppressed
    assert result.trace.recommendations_candidate > 0


def test_scenario_h_critical_provider_failure_suppresses(swarm_factory, monkeypatch):
    """Losing critical evidence must suppress ideas, not quietly thin them."""
    from marketswarm.agents.market_agents import TechnicalAgent
    from marketswarm.agents.synthesis import CrossVerificationAgent

    async def fail(self, ctx):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(TechnicalAgent, "run", fail)
    monkeypatch.setattr(CrossVerificationAgent, "run", fail)

    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm)

    assert result.v2 is not None
    level = (result.v2.degradation or {}).get("level")
    assert level in ("degraded", "critical", "unusable"), f"degradation not detected: {level}"
    for rec in result.publication.active():
        assert not rec.actionable, \
            "an actionable idea survived the loss of a critical agent"


def test_scenario_i_rejected_idea_is_invisible_to_monitor_and_api(
        swarm_factory, monkeypatch):
    """The monitor, the scorer and the API all read from the same tables."""
    force_critical_objection(monkeypatch, "NVDA")
    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm)
    rejected = result.publication.rejected_subjects()
    assert rejected

    from marketswarm.monitor import InvalidationMonitor
    monitored = {row["symbol"] for row in
                 InvalidationMonitor(swarm.store).todays_ideas(RUN_DATE)}
    assert not (monitored & rejected), "the monitor is tracking a rejected idea"

    scoring_queue = {row["symbol"] for row in swarm.store.unresolved_predictions()}
    assert not (scoring_queue & rejected), "a rejected idea entered the scoring queue"

    from marketswarm.api import ReadOnlyAPI
    api = ReadOnlyAPI(swarm.store.conn)
    active = api.recommendations(RUN_DATE.isoformat())
    assert not ({r.get("subject") for r in active} & rejected), \
        "the API is serving a rejected candidate as active"


# ==========================================================================
# observability and modes
# ==========================================================================

def test_control_path_trace_is_persisted_and_proves_the_architecture(swarm_factory):
    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm)

    row = swarm.store.conn.execute(
        "SELECT orchestration_mode, agents_available, agents_executed, "
        "agents_skipped, agent_execution_reason, legacy_fallback_used, "
        "recommendations_candidate, recommendations_rejected "
        "FROM run_control_path WHERE run_id = ?", (result.run_id,)).fetchone()
    assert row is not None, "no control-path trace was written"
    mode, available, executed, skipped, reasons, legacy, cand, rej = row

    assert mode == "dynamic"
    assert available == len(ALL_AGENTS)
    assert executed + skipped == available
    assert legacy == 0
    reasons = json.loads(reasons)
    assert len(reasons) == available, "every agent must have a recorded reason"
    assert cand == len(result.publication.approved) + len(result.publication.rejected)
    assert rej == len(result.publication.rejected)


def test_legacy_mode_cannot_publish_unreviewed_ideas_by_default(swarm_factory):
    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm, mode="legacy")

    assert result.trace.legacy_fallback_used
    assert result.publication.suppressed
    assert result.ideas == {"calls": [], "puts": [], "stocks": []}
    assert "allow_unreviewed_publication" in result.publication.suppression_reason


def test_legacy_mode_publishes_only_when_explicitly_allowed(swarm_factory):
    swarm = swarm_factory(gaps=EVENTFUL, allow_unreviewed_publication=True)
    result = run(swarm, mode="legacy")

    assert not result.publication.suppressed
    for rec in result.publication.active():
        assert rec.review_status == "NOT_REVIEWED"
        assert any("did not pass through" in n for n in rec.uncertainty_notes)


def test_full_mode_still_routes_through_the_review_gate(swarm_factory):
    """Benchmark mode changes routing, never the publication authority."""
    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm, mode="full")

    assert result.trace.agents_executed == len(ALL_AGENTS)
    assert not result.trace.legacy_fallback_used
    assert result.v2 is not None
    for rec in result.publication.active():
        assert rec.review_status is not None
        assert rec.review_status != "NOT_REVIEWED"


def test_unknown_mode_falls_back_to_dynamic(swarm_factory):
    swarm = swarm_factory()
    result = run(swarm, mode="turbo")
    assert result.trace.orchestration_mode == "dynamic"


def test_no_trading_capability_was_introduced():
    """The hard boundary, re-asserted against the new modules."""
    from marketswarm.security import FORBIDDEN_CAPABILITIES, assert_no_shell_execution

    assert_no_shell_execution()
    for cap in ("broker_order", "funds_transfer", "portfolio_allocate"):
        assert cap in FORBIDDEN_CAPABILITIES

    import pathlib
    banned = ("place_order", "submit_order", "buy(", "sell(", "broker.")
    for path in pathlib.Path("marketswarm").rglob("*.py"):
        text = path.read_text()
        for token in banned:
            assert token not in text, f"{path} contains {token!r}"
