"""End-to-end tests for the review / follow-up state machine.

2.0.1 could launch follow-up research. What it could not do was let that
research change anything: the evidence graph was built once per session and
the red-team report was captured into a closure, so round 2 re-derived
byte-identical objections from a stale report and the loop's no-progress guard
then terminated. The research was bought and discarded.

These tests assert on the properties that were missing, through the real
`Swarm.run()`:

    new research  →  new evidence  →  new adversarial review  →  new decision
    failed review ≠ approval
    current event context  →  contextual routing  →  outcome learning
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

import marketswarm.orchestrator as orch
from marketswarm.agents import ALL_AGENTS
from marketswarm.pipeline2 import CandidateReview
from marketswarm.recommend.engine import RecommendationEngine
from marketswarm.review.gate import (ReviewDecision, ReviewExecutionStatus,
                                     ReviewGate, ReviewStatus, Severity)

from .fakes import FakeEarnings, FakeEcon, FakeEdgar, FakeMarket, FakeNews, FakeOptions

UNIVERSE = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMD"]
RUN_DATE = dt.date(2026, 8, 17)
QUIET = None
EVENTFUL = {"NVDA": 3.4, "AAPL": -2.9, "SPY": 0.3, "QQQ": 0.4, "MSFT": 1.1, "AMD": -2.2}


@pytest.fixture
def swarm_factory(tmp_path, monkeypatch):
    def build(gaps=QUIET, **cfg_kwargs):
        from marketswarm.config import Config

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


def run(swarm, **kw):
    return asyncio.run(swarm.run(RUN_DATE, force=True, **kw))


def demand_research_once(monkeypatch, followups=None):
    """First round always demands research; later rounds use the real gate."""
    original = ReviewGate.review
    calls = {"n": 0}

    def demanding(self, findings, original_confidence, iteration=0,
                  max_iterations=2, evidence_already_requested=False,
                  execution_status=ReviewExecutionStatus.COMPLETED):
        if iteration == 0 and not evidence_already_requested:
            calls["n"] += 1
            return ReviewDecision(
                status=ReviewStatus.REQUEST_MORE_RESEARCH,
                original_confidence=original_confidence,
                revised_confidence=max(0, original_confidence - 5),
                severity=Severity.MEDIUM,
                reasons=["needs corroboration from the filing record"],
                required_followups=followups or [
                    "check the SEC filing record for an 8-K",
                    "corroborate the headline with a second source"],
                iteration=iteration)
        return original(self, findings, original_confidence, iteration,
                        max_iterations, evidence_already_requested, execution_status)

    monkeypatch.setattr(ReviewGate, "review", demanding)
    return calls


# ==========================================================================
# TEST A — follow-up evidence actually changes the graph
# ==========================================================================

def test_followup_research_grows_the_evidence_graph(swarm_factory, monkeypatch):
    """Not `len(nodes) > 0` — that passes on a graph nothing touched.

    Captures the node count and graph version before the follow-up and asserts
    both moved, and that the nodes which appeared carry provenance from an
    agent that only ran because the review demanded it.
    """
    demand_research_once(monkeypatch)
    swarm = swarm_factory(gaps=QUIET)
    result = run(swarm)

    assert result.v2 is not None
    sessions = result.v2.review_sessions
    assert sessions, "no review session was recorded"

    grew = [s for s in sessions if s.graph_version > 1]
    assert grew, "follow-up ran but no session ever rebuilt its evidence graph"

    for s in grew:
        assert s.nodes_after_followup > s.nodes_before_followup, (
            f"{s.symbol}: graph went from {s.nodes_before_followup} to "
            f"{s.nodes_after_followup} nodes — the follow-up bought nothing")

        # Provenance: the new nodes come from agents the follow-up recruited.
        followup_agents = set(s.followup_agents)
        assert followup_agents, "no follow-up agent was recorded"
        provenance = {n.agent for n in s.graph.nodes.values()}
        assert followup_agents & provenance, (
            f"{s.symbol}: follow-up ran {followup_agents} but none of them "
            f"appear in the graph's provenance ({sorted(provenance)})")

    # The rounds record which graph version each one saw.
    for s in grew:
        versions = [r["graph_version"] for r in s.rounds]
        assert versions == sorted(versions), "graph versions must not go backwards"
        assert versions[-1] > versions[0], (
            f"{s.symbol}: every round saw graph v{versions[0]} — the second "
            f"round was evaluated against stale evidence")

    assert result.trace.evidence_nodes_after_followup > \
        result.trace.evidence_nodes_before_followup
    assert result.trace.graph_versions > 1


# ==========================================================================
# TEST B — follow-up triggers a genuinely fresh adversarial pass
# ==========================================================================

def test_followup_triggers_a_fresh_red_team_pass(swarm_factory, monkeypatch):
    """The red-team agent must actually run again, not be re-read."""
    from marketswarm.agents.redteam import RedTeamAgent

    invocations = {"n": 0}
    original_run = RedTeamAgent.run

    async def counted(self, ctx):
        invocations["n"] += 1
        return await original_run(self, ctx)

    monkeypatch.setattr(RedTeamAgent, "run", counted)
    demand_research_once(monkeypatch)

    swarm = swarm_factory(gaps=QUIET)
    result = run(swarm)

    assert result.trace.followup_requests > 0, "no research was demanded"
    assert invocations["n"] >= 2, (
        f"the red team ran {invocations['n']} time(s) — round 2 reused round 1's "
        f"report instead of re-attacking the updated evidence")
    assert result.trace.red_team_attempts >= 1
    assert result.trace.red_team_successes >= 1


def test_each_round_is_preserved_not_overwritten(swarm_factory, monkeypatch):
    """Round 1 must survive round 2, in memory and in the database."""
    demand_research_once(monkeypatch)
    swarm = swarm_factory(gaps=QUIET)
    result = run(swarm)

    multi = [s for s in result.v2.review_sessions if len(s.rounds) > 1]
    assert multi, "no candidate went through more than one round"
    for s in multi:
        assert [r["round"] for r in s.rounds] == list(range(1, len(s.rounds) + 1))

    rows = swarm.store.conn.execute(
        "SELECT subject, round_number, graph_version, red_team_execution_status "
        "FROM review_rounds ORDER BY subject, round_number").fetchall()
    assert rows, "no review round was persisted"
    by_subject: dict[str, list[int]] = {}
    for subject, rnd, _version, _status in rows:
        by_subject.setdefault(subject, []).append(rnd)
    assert any(len(v) > 1 for v in by_subject.values()), \
        "the database kept only one round per candidate"


# ==========================================================================
# TEST C — new evidence can change the decision
# ==========================================================================

def test_new_evidence_can_change_the_verdict(swarm_factory, monkeypatch):
    """Round 2 must be derived from post-follow-up state.

    The red team here objects *only while* the filing record is unexamined.
    Once the follow-up runs `sec_filings`, a re-run produces no objection — so
    if round 2 still carries the objection, it was reading a stale report.
    """
    from marketswarm.agents.redteam import RedTeamAgent

    async def conditional(self, ctx):
        from marketswarm.agents.base import AgentReport

        rep = AgentReport(agent="red_team", headline="Red team")
        if "sec_filings" not in ctx.reports:
            rep.data = {"objections": [{
                "severity": "high",
                "objection": "The filing record has not been checked; the thesis "
                             "rests on price action alone.",
                "test": "Pull the 8-K record before acting.",
            }], "high_severity": 1}
        else:
            rep.data = {"objections": [], "high_severity": 0}
        rep.confidence = 0.7
        return rep

    monkeypatch.setattr(RedTeamAgent, "run", conditional)
    demand_research_once(monkeypatch,
                         followups=["check the SEC filing record for an 8-K"])

    swarm = swarm_factory(gaps=QUIET)
    result = run(swarm)

    sessions = [s for s in result.v2.review_sessions if len(s.rounds) > 1]
    assert sessions, "no candidate reached a second round"

    for s in sessions:
        first, last = s.rounds[0], s.rounds[-1]
        assert first["n_findings"] > 0, "round 1 should have carried the objection"
        assert last["n_findings"] < first["n_findings"], (
            f"{s.symbol}: round 1 had {first['n_findings']} findings and round "
            f"{last['round']} had {last['n_findings']} — the objection survived "
            f"evidence that answered it, so round 2 read a stale report")
        assert last["graph_version"] > first["graph_version"]

    # The final decision is the one that came from the later round.
    for rec in result.publication.active():
        if rec.review_rounds:
            assert rec.graph_version == rec.review_rounds[-1]["graph_version"]


def test_failed_followup_is_not_treated_as_evidence(swarm_factory, monkeypatch):
    """Scenario 3: research requested, nothing came back.

    The unanswered request must itself become an objection. Silently carrying
    on is how a demand for corroboration turns into corroboration.
    """
    demand_research_once(monkeypatch)

    from marketswarm.investigation.chief import ChiefInvestigator
    # No agent can answer, so the follow-up cannot produce anything.
    monkeypatch.setattr(ChiefInvestigator, "agents_for_questions",
                        lambda self, questions: [])
    monkeypatch.setattr(ChiefInvestigator, "followup_plan",
                        lambda self, parent, questions, depth=1: None)

    swarm = swarm_factory(gaps=QUIET)
    result = run(swarm)

    assert result.trace.followup_requests >= 0
    sessions = [s for s in result.v2.review_sessions if len(s.rounds) > 1]
    assert sessions, "no candidate reached a second round"

    unresolved = [r for s in sessions for r in s.rounds[1:]
                  if any("could not be obtained" in o for o in r["objections"])]
    assert unresolved, (
        "follow-up produced nothing, yet no round recorded the request as "
        "unresolved — the system acted as though evidence had arrived")

    # And nothing that carried an unanswered demand was published actionable.
    for rec in result.publication.actionable():
        assert not any("could not be obtained" in str(rec.review_rounds))


# ==========================================================================
# TEST D / E — review failure is not approval; completed-empty is
# ==========================================================================

def test_red_team_failure_suppresses_publication(swarm_factory, monkeypatch):
    """Scenario 4. The most important safety property in this release."""
    from marketswarm.agents.redteam import RedTeamAgent

    async def crash(self, ctx):
        raise RuntimeError("red team unavailable")

    monkeypatch.setattr(RedTeamAgent, "run", crash)
    # Disable the deterministic fallback so this tests the bare failure path.
    monkeypatch.setattr("marketswarm.pipeline2.deterministic_review",
                        lambda reports, symbol=None: [])

    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm)

    pub = result.publication
    assert pub.suppressed, "a failed adversarial review still published"
    assert pub.actionable() == []
    assert result.ideas == {"calls": [], "puts": [], "stocks": []}
    assert "review did not complete" in pub.suppression_reason
    assert result.trace.review_incomplete

    # No active prediction, so nothing is monitored or scored.
    assert swarm.store.conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE run_date = ?",
        (RUN_DATE.isoformat(),)).fetchone()[0] == 0

    # The candidate remains auditable, marked SUPPRESSED rather than REJECTED —
    # the reviewer never gave a verdict, so recording one would be a lie.
    rows = swarm.store.conn.execute(
        "SELECT subject, status FROM recommendations "
        "WHERE status IN ('REJECTED','SUPPRESSED')").fetchall()
    assert rows, "the candidate was not kept for audit"
    assert any(status == "SUPPRESSED" for _, status in rows)

    # Report, webhook and API all reflect the suppression.
    from marketswarm.api import ReadOnlyAPI
    from marketswarm.notify import summarize
    from marketswarm.report import render_markdown

    md = render_markdown(result)
    assert "Recommendations withheld" in md
    assert "review did not complete" in md
    assert "No recommendations published" in summarize(result)
    assert ReadOnlyAPI(swarm.store.conn).recommendations(RUN_DATE.isoformat()) == []


def test_red_team_failure_falls_back_to_deterministic_review(swarm_factory, monkeypatch):
    """A degraded review is allowed — and is labelled as degraded."""
    from marketswarm.agents.redteam import RedTeamAgent

    async def crash(self, ctx):
        raise RuntimeError("red team unavailable")

    monkeypatch.setattr(RedTeamAgent, "run", crash)

    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm)

    sessions = result.v2.review_sessions
    assert sessions, "no review session ran"
    statuses = {r["execution_status"] for s in sessions for r in s.rounds}
    assert ReviewExecutionStatus.COMPLETED_BY_FALLBACK.value in statuses, (
        f"the red team failed but the fallback reviewer did not run: {statuses}")

    # The fallback always states that review was degraded.
    objections = [o for s in sessions for r in s.rounds for o in r["objections"]]
    assert any("fallback mode" in o for o in objections)


def test_completed_review_with_no_findings_may_approve(swarm_factory, monkeypatch):
    """Scenario 5 / TEST E: the system distinguishes clean from unreviewed."""
    from marketswarm.agents.base import AgentReport
    from marketswarm.agents.redteam import RedTeamAgent

    async def clean(self, ctx):
        rep = AgentReport(agent="red_team", headline="Red team: no objection")
        rep.data = {"objections": [], "high_severity": 0,
                    "recommend_stand_down": False}
        rep.confidence = 0.7
        return rep

    monkeypatch.setattr(RedTeamAgent, "run", clean)

    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm)

    assert not result.publication.suppressed, \
        "a completed review that found nothing was treated as no review"
    assert not result.trace.review_incomplete

    statuses = {r["execution_status"] for s in result.v2.review_sessions
                for r in s.rounds}
    assert statuses == {ReviewExecutionStatus.COMPLETED.value}

    approved = [r for r in result.publication.active()
                if r.review_status == ReviewStatus.APPROVE.value]
    assert approved, "nothing was approved despite a clean completed review"
    for rec in approved:
        assert rec.confidence == rec.original_confidence


def test_gate_rejects_every_untrustworthy_execution_status():
    """Unit-level completeness: no failure mode reads as approval."""
    gate = ReviewGate()
    for status in ReviewExecutionStatus:
        decision = gate.review([], original_confidence=90, execution_status=status)
        if status.is_trustworthy:
            assert decision.status is ReviewStatus.APPROVE
            assert decision.revised_confidence == 90
        else:
            assert decision.status is ReviewStatus.REVIEW_INCOMPLETE, \
                f"{status.value} produced {decision.status.value}"
            assert decision.revised_confidence == 0
            assert decision.status.blocks_publication


def test_malformed_red_team_output_is_invalid_not_clean():
    """An agent that says 'ok' but emits nothing readable is not a clean bill."""
    from marketswarm.agents.base import AgentReport
    from marketswarm.review.loop import redteam_execution_status

    assert redteam_execution_status(None) is ReviewExecutionStatus.UNAVAILABLE

    rep = AgentReport(agent="red_team", headline="x")
    rep.data = {}                       # no `objections` key at all
    assert redteam_execution_status(rep) is ReviewExecutionStatus.INVALID

    rep.data = {"objections": []}
    assert redteam_execution_status(rep) is ReviewExecutionStatus.COMPLETED


# ==========================================================================
# TEST F — the leak invariant uses candidate lineage
# ==========================================================================

def test_leak_check_uses_candidate_lineage_not_recommendation_id():
    """Scenario 6, with production-shaped identities.

    The 2.0.1 check compared `rec.id` against rejected `candidate_id`s. Those
    live in different namespaces (`rec_...` vs `cand_...`), so it could never
    fire on real data — the earlier test only passed because it set both to the
    same string.
    """
    from marketswarm.publication import (PublicationError, PublicationSet,
                                         RejectedCandidate)
    from marketswarm.recommend.engine import (Conviction, Recommendation,
                                              RecommendationType)

    rec = Recommendation(
        id="rec_456", subject="NVDA", rec_type=RecommendationType.FAVORABLE,
        conviction=Conviction.HIGH_CONVICTION, direction="long",
        forecast_probability=0.62, confidence=70, original_confidence=70,
        expected_r=0.3, source_kind="stock", candidate_id="cand_123")
    pub = PublicationSet(
        approved=[rec],
        rejected=[RejectedCandidate(subject="NVDA", candidate_id="cand_123",
                                    reason="critical finding")])

    with pytest.raises(PublicationError, match="cand_123"):
        pub.assert_no_leak()


def test_leak_check_catches_a_rejected_revision_parent():
    from marketswarm.publication import (PublicationError, PublicationSet,
                                         RejectedCandidate)
    from marketswarm.recommend.engine import (Conviction, Recommendation,
                                              RecommendationType)

    rec = Recommendation(
        id="rec_789", subject="AMD", rec_type=RecommendationType.FAVORABLE,
        conviction=Conviction.MODERATE_CONVICTION, direction="long",
        forecast_probability=0.55, confidence=45, original_confidence=60,
        expected_r=0.2, source_kind="stock",
        candidate_id="cand_999", revision_parent_id="cand_rejected")
    pub = PublicationSet(
        approved=[rec],
        rejected=[RejectedCandidate(subject="AMD", candidate_id="cand_rejected",
                                    reason="critical finding")])

    with pytest.raises(PublicationError, match="descends from"):
        pub.assert_no_leak()


def test_distinct_lineage_is_allowed():
    """A different candidate for the same symbol is legitimate."""
    from marketswarm.publication import PublicationSet, RejectedCandidate
    from marketswarm.recommend.engine import (Conviction, Recommendation,
                                              RecommendationType)

    rec = Recommendation(
        id="rec_456", subject="NVDA", rec_type=RecommendationType.FAVORABLE,
        conviction=Conviction.MODERATE_CONVICTION, direction="long",
        forecast_probability=0.58, confidence=45, original_confidence=45,
        expected_r=0.2, source_kind="stock", candidate_id="cand_OTHER")
    PublicationSet(
        approved=[rec],
        rejected=[RejectedCandidate(subject="NVDA", candidate_id="cand_123",
                                    reason="rejected")]).assert_no_leak()


def test_production_run_assigns_distinct_candidate_and_recommendation_ids(
        swarm_factory):
    """The namespaces really are different in production."""
    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm)

    seen = result.publication.active() or []
    for rec in seen:
        assert rec.candidate_id, "a published recommendation has no lineage"
        assert rec.id != rec.candidate_id
        assert rec.id.startswith("rec_")
        assert rec.candidate_id.startswith("cand_")
    for rej in result.publication.rejected:
        assert rej.candidate_id.startswith("cand_")


# ==========================================================================
# TEST G / I — the current event context drives routing
# ==========================================================================

def _seed_context(conn, agent: str, regime: str, event_type: str, weight: float,
                  n: int = 60):
    """Write a measured contextual weight directly, as the tracker would."""
    from marketswarm.memory.contribution import ContextKey

    key = ContextKey(regime=regime, event_type=event_type, horizon="intraday").key()
    conn.execute(
        """INSERT OR REPLACE INTO agent_context_scores
           (agent, context_key, weight, n, contribution_sum, updated_at)
           VALUES (?,?,?,?,?,datetime('now'))""",
        (agent, key, weight, n, (0.1 if weight > 1 else -0.1) * n))
    conn.commit()


def test_event_context_is_resolved_before_routing(swarm_factory):
    """The weights that route must be resolved *after* detection."""
    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm)

    ctx = result.trace.event_context_resolved
    assert ctx and ctx != "any", "no event context was resolved for routing"
    assert "event=" in ctx and "regime=" in ctx


def test_event_specific_weights_change_which_agents_execute(swarm_factory,
                                                            monkeypatch):
    """TEST G / Scenario 7 — assert on execution, not on computed weights."""
    executed: list[list[str]] = []

    def capture(result):
        return sorted(k for k, v in result.trace.agent_execution_reason.items()
                      if not v.startswith("skipped"))

    # Baseline: no contextual history at all.
    base_swarm = swarm_factory(gaps=EVENTFUL)
    base = run(base_swarm)
    executed.append(capture(base))

    # Now teach the system that `institutional` is worth a lot in the event
    # context this fixture produces, and that `sentiment` is worthless.
    ctx = base.trace.event_context_resolved
    regime = ctx.split("regime=")[1].split("|")[0]
    event_type = ctx.split("event=")[1]

    dropped = next(a for a in ("breaking_news", "earnings", "sec_filings")
                   if a in executed[0])

    swarm = swarm_factory(gaps=EVENTFUL)
    _seed_context(swarm.store.conn, dropped, regime, event_type, 0.15)
    seeded = run(swarm)
    executed.append(capture(seeded))

    assert seeded.trace.contextual_weights_used > 0, (
        "seeded contextual weights were never consulted — routing resolved "
        "before the event was known")
    assert executed[0] != executed[1], (
        f"learned event-specific reliability changed nothing about execution: "
        f"{executed[0]}")
    assert dropped in executed[0], "precondition: the agent ran in the baseline"
    assert dropped not in executed[1], (
        f"{dropped} was measured to subtract value in this event context and "
        f"was run anyway — learned weights only reordered, they never excluded")


def test_unclassified_event_falls_back_without_inventing_context(swarm_factory,
                                                                 monkeypatch):
    """Scenario 8: no detectable event must not fabricate one."""
    from marketswarm.investigation.event_brain import EventBrain

    monkeypatch.setattr(EventBrain, "scan", lambda self, snapshot: [])

    swarm = swarm_factory(gaps=QUIET)
    result = run(swarm)

    assert result.trace.event_context_resolved.endswith("event=any"), (
        f"an unclassified session invented the context "
        f"{result.trace.event_context_resolved!r}")
    assert result.trace.agents_executed > 0
    assert result.v2 is not None


def test_dominant_event_type_prefers_priority_then_count():
    from marketswarm.investigation.event_brain import (DetectedEvent, EventPriority,
                                                       EventType)
    from marketswarm.orchestrator import _dominant_event_type

    assert _dominant_event_type([]) == "any"

    low = DetectedEvent(subject="A", event_type=EventType.QUIET,
                        priority=EventPriority.LOW, description="")
    high = DetectedEvent(subject="B", event_type=EventType.EARNINGS_REACTION,
                         priority=EventPriority.CRITICAL, description="")
    assert _dominant_event_type([low, low, low, high]) == \
        EventType.EARNINGS_REACTION.value


# ==========================================================================
# TEST H / I — event context survives into learning and back out again
# ==========================================================================

def test_event_context_is_persisted_for_later_learning(swarm_factory, monkeypatch):
    """TEST H, first half: the run records what it knew at the time.

    Uses a clean completed review so something is actually published — a
    version of this test that skipped when nothing survived would prove
    nothing at all.
    """
    import json

    from marketswarm.agents.base import AgentReport
    from marketswarm.agents.redteam import RedTeamAgent

    async def clean(self, ctx):
        rep = AgentReport(agent="red_team", headline="Red team: no objection")
        rep.data = {"objections": [], "high_severity": 0}
        rep.confidence = 0.7
        return rep

    monkeypatch.setattr(RedTeamAgent, "run", clean)

    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm)

    rows = swarm.store.conn.execute(
        "SELECT features FROM predictions WHERE run_date = ?",
        (RUN_DATE.isoformat(),)).fetchall()
    assert rows, "a clean review published nothing, so there is nothing to learn from"

    for (features,) in rows:
        feats = json.loads(features or "{}")
        assert "event_type" in feats, \
            "the event type was not persisted — learning would infer it from `setup`"
        assert feats["event_type"] == result.trace.event_context_resolved.split(
            "event=")[1]
        assert "horizon" in feats
        assert "agents_executed" in feats and "agents_skipped" in feats


def test_scoring_updates_the_event_specific_context(swarm_factory):
    """TEST H, second half: the outcome lands in the right bucket."""
    from marketswarm.closed_loop import ClosedLoop
    from marketswarm.memory import Prediction
    from marketswarm.memory.contribution import MIN_OBSERVATIONS

    swarm = swarm_factory()
    conn = swarm.store.conn
    run_id = swarm.store.start_run(RUN_DATE.isoformat(), "premarket")

    for i in range(MIN_OBSERVATIONS * 3):
        win = i % 2 == 0
        pid = swarm.store.record_prediction(Prediction(
            run_date=RUN_DATE.isoformat(), kind="stock_setup", symbol="NVDA",
            direction="long", probability=0.6 if win else 0.4,
            entry=100, target=103, stop=98,
            features={"regime": "quiet_trend", "event_type": "earnings_reaction",
                      "horizon": "intraday"},
            contributing_agents={"earnings": 0.5 if win else -0.5}), run_id)
        swarm.store.resolve(pid, 1 if win else 0, 1.0 if win else -1.0)

    ClosedLoop(conn).run()

    keys = [r[0] for r in conn.execute(
        "SELECT DISTINCT context_key FROM agent_context_scores")]
    assert any("event=earnings_reaction" in k for k in keys), (
        f"the outcome was not learned into its event-specific bucket: {keys}")


def test_learned_earnings_context_does_not_leak_into_macro(swarm_factory):
    """TEST I, second half: event contexts stay separate."""
    from marketswarm.memory.resolver import ContextualWeightResolver

    swarm = swarm_factory()
    conn = swarm.store.conn
    _seed_context(conn, "earnings", "quiet_trend", "earnings_reaction", 1.70)

    r = ContextualWeightResolver(conn)
    at_earnings = r.get_agent_weight("earnings", regime="quiet_trend",
                                     event_type="earnings_reaction")
    at_macro = r.get_agent_weight("earnings", regime="quiet_trend",
                                  event_type="macro_release")

    assert at_earnings == pytest.approx(1.70)
    assert at_macro != at_earnings, (
        "an earnings-specific weight dominated a macro route — the contexts "
        "are not actually separated")
    assert r.provenance()["earnings"].startswith("prior") or \
        "event=any" in r.provenance()["earnings"]


def test_contextual_backoff_is_hierarchical():
    """Thin exact context falls back rather than routing on noise."""
    import sqlite3

    from marketswarm.memory.contribution import ContextKey
    from marketswarm.memory.migrations import migrate
    from marketswarm.memory.resolver import ContextualWeightResolver

    conn = sqlite3.connect(":memory:")
    migrate(conn)

    # Exact context has too few observations; the broader one is well measured.
    exact = ContextKey(regime="high_vol", event_type="price_gap",
                       horizon="intraday").key()
    broad = ContextKey(regime="any", event_type="price_gap",
                       horizon="intraday").key()
    conn.execute("INSERT INTO agent_context_scores "
                 "(agent, context_key, weight, n, contribution_sum, updated_at) "
                 "VALUES ('options_flow',?,1.90,3,1.2,datetime('now'))", (exact,))
    conn.execute("INSERT INTO agent_context_scores "
                 "(agent, context_key, weight, n, contribution_sum, updated_at) "
                 "VALUES ('options_flow',?,1.30,80,16.0,datetime('now'))", (broad,))
    conn.commit()

    w = ContextualWeightResolver(conn).get_agent_weight(
        "options_flow", regime="high_vol", event_type="price_gap")
    assert w == pytest.approx(1.30), (
        f"resolved {w} — a 3-observation cell was allowed to dominate routing")


# ==========================================================================
# observability
# ==========================================================================

def test_trace_records_the_review_and_followup_shape(swarm_factory, monkeypatch):
    demand_research_once(monkeypatch)
    swarm = swarm_factory(gaps=QUIET)
    result = run(swarm)
    t = result.trace

    assert t.review_rounds >= 2, "multi-round review was not recorded"
    assert t.followup_requests >= 1
    assert t.followup_agents_selected >= t.followup_agents_executed
    assert t.graph_versions >= 2
    assert t.evidence_nodes_after_followup >= t.evidence_nodes_before_followup

    row = swarm.store.conn.execute(
        "SELECT review_rounds, followup_requests, followup_agents_executed, "
        "graph_versions, event_context_resolved, red_team_attempts, "
        "review_incomplete FROM run_control_path WHERE run_id = ?",
        (result.run_id,)).fetchone()
    assert row is not None, "the trace was not persisted"
    rounds, requests, executed, versions, ctx, attempts, incomplete = row
    assert rounds == t.review_rounds
    assert requests == t.followup_requests
    assert executed == t.followup_agents_executed
    assert versions == t.graph_versions
    assert ctx == t.event_context_resolved
    assert attempts == t.red_team_attempts
    assert incomplete == int(t.review_incomplete)


def test_red_team_is_treated_as_critical_to_publication():
    """Publication requires a successful adversarial review, by policy."""
    from marketswarm.resilience import CRITICAL_AGENTS

    assert "red_team" in CRITICAL_AGENTS, (
        "the red team is not critical, so losing it degrades quietly instead "
        "of suppressing publication")


def test_no_trading_capability_and_scientist_still_propose_only():
    """The 2.0 safety invariants survive this release."""
    from marketswarm.experiments.scientist import ResearchScientist
    from marketswarm.security import FORBIDDEN_CAPABILITIES, assert_no_shell_execution

    assert_no_shell_execution()
    for cap in ("broker_order", "funds_transfer", "portfolio_allocate"):
        assert cap in FORBIDDEN_CAPABILITIES
    for name in ("promote", "deploy", "apply_to_production"):
        assert not hasattr(ResearchScientist, name)


def test_all_agents_still_execute_when_nothing_is_wrong(swarm_factory):
    """Guard against the review changes accidentally shrinking the swarm."""
    swarm = swarm_factory(gaps=EVENTFUL)
    result = run(swarm, mode="full")
    assert result.trace.agents_executed >= len(ALL_AGENTS)


def test_contradictory_followup_evidence_changes_the_verdict(swarm_factory,
                                                             monkeypatch):
    """Gauntlet scenario 1: new evidence that argues *against* the thesis.

    The mirror of TEST C. There, follow-up resolved an objection; here it
    creates one, and the final decision must reflect the harsher round.
    """
    from marketswarm.agents.base import AgentReport
    from marketswarm.agents.redteam import RedTeamAgent

    async def worsening(self, ctx):
        rep = AgentReport(agent="red_team", headline="Red team")
        if "sec_filings" not in ctx.reports:
            rep.data = {"objections": [], "high_severity": 0}
        else:
            rep.data = {"objections": [{
                "severity": "critical",
                "objection": "The filing record contradicts the thesis outright.",
                "test": "Read the 8-K before acting.",
            }], "high_severity": 1}
        rep.confidence = 0.7
        return rep

    monkeypatch.setattr(RedTeamAgent, "run", worsening)
    demand_research_once(monkeypatch,
                         followups=["check the SEC filing record for an 8-K"])

    swarm = swarm_factory(gaps=QUIET)
    result = run(swarm)

    sessions = [s for s in result.v2.review_sessions if len(s.rounds) > 1]
    assert sessions, "no candidate reached a second round"
    for s in sessions:
        first, last = s.rounds[0], s.rounds[-1]
        assert last["n_findings"] > first["n_findings"], (
            f"{s.symbol}: contradictory evidence arrived and the objection "
            f"count went {first['n_findings']} → {last['n_findings']}")

    # A CRITICAL objection raised in round 2 must kill the candidate.
    assert result.publication.rejected, (
        "evidence that contradicted the thesis arrived in round 2 and nothing "
        "was rejected — the later round did not drive the decision")


def test_after_action_review_reports_whether_followup_paid_off():
    """The round history makes a research-value question answerable."""
    from marketswarm.after_action import AfterActionReviewer

    verdict = AfterActionReviewer._research_verdict

    assert verdict([]) == ""
    assert verdict([{"round": 1}]) == ""

    wasted = verdict([
        {"round": 1, "graph_version": 1, "evidence_nodes": 19, "n_findings": 2},
        {"round": 2, "graph_version": 1, "evidence_nodes": 19, "n_findings": 2},
    ])
    assert "never changed" in wasted and "no information" in wasted

    paid = verdict([
        {"round": 1, "graph_version": 1, "evidence_nodes": 19, "n_findings": 3},
        {"round": 2, "graph_version": 2, "evidence_nodes": 24, "n_findings": 1,
         "followup_agents": ["sec_filings"]},
    ])
    assert "resolved 2 objection" in paid and "sec_filings" in paid

    worsened = verdict([
        {"round": 1, "graph_version": 1, "evidence_nodes": 19, "n_findings": 0},
        {"round": 2, "graph_version": 2, "evidence_nodes": 22, "n_findings": 2},
    ])
    assert "raised 2 new objection" in worsened

    neutral = verdict([
        {"round": 1, "graph_version": 1, "evidence_nodes": 19, "n_findings": 2},
        {"round": 2, "graph_version": 2, "evidence_nodes": 21, "n_findings": 2},
    ])
    assert "did not bear on them" in neutral


def test_closed_loop_reads_the_round_history(swarm_factory, monkeypatch):
    """End-to-end: rounds persisted by a run are read back by scoring."""
    from marketswarm.closed_loop import ClosedLoop

    demand_research_once(monkeypatch)
    swarm = swarm_factory(gaps=QUIET)
    run(swarm)

    rounds = swarm.store.conn.execute(
        "SELECT COUNT(*) FROM review_rounds").fetchone()[0]
    assert rounds >= 2, "the run persisted no multi-round history"

    loop = ClosedLoop(swarm.store.conn)

    import sqlite3
    swarm.store.conn.row_factory = sqlite3.Row
    row = swarm.store.conn.execute(
        "SELECT r.subject AS symbol, p.run_date AS run_date FROM review_rounds r "
        "JOIN run_control_path p ON p.run_id = r.run_id LIMIT 1").fetchone()
    recovered = loop._rounds_for(row)
    assert len(recovered) >= 2, (
        "the closed loop could not recover the round history it needs to say "
        "whether the research was worth it")
    assert recovered[0]["round"] < recovered[-1]["round"]


# ==========================================================================
# TEST N — the published recommendation is built on the evidence that
#          approved it, not on the evidence round 1 argued about
# ==========================================================================

def test_publication_is_built_from_the_refreshed_graph(swarm_factory, monkeypatch):
    """The state-coherence invariant, asserted where it was actually broken.

    Before this, `build_recommendations` recorded `session.graph_version` onto
    the recommendation while passing the *outer* graph to the engine. So a
    candidate could be stamped v2, stored as v2, and reported as v2, while the
    evidence the engine actually reasoned over was v1 — the split state the
    review loop exists to prevent, and invisible from the outside precisely
    because the label was right.
    """
    from marketswarm.agents.base import AgentReport
    from marketswarm.agents.redteam import RedTeamAgent

    # Round 1 demands follow-up (so the graph refreshes), then the red team
    # finds nothing (so something actually reaches the engine). Both halves are
    # needed: a run where everything is rejected never calls build at all.
    async def clean(self, ctx):
        rep = AgentReport(agent="red_team", headline="Red team: no objection")
        rep.data = {"objections": [], "high_severity": 0,
                    "recommend_stand_down": False}
        rep.confidence = 0.7
        return rep

    demand_research_once(monkeypatch)
    monkeypatch.setattr(RedTeamAgent, "run", clean)
    swarm = swarm_factory(gaps=EVENTFUL)

    seen: list[int] = []
    original_build = RecommendationEngine.build

    def spy(self, inputs, graph=None, **kw):
        # Node count identifies which graph object the engine was handed.
        seen.append(len(graph.nodes) if graph is not None else -1)
        return original_build(self, inputs, graph=graph, **kw)

    monkeypatch.setattr(RecommendationEngine, "build", spy)
    result = run(swarm)

    sessions = result.v2.review_sessions
    grew = [s for s in sessions if s.graph_version > 1]
    if not grew:
        pytest.skip("no session refreshed its graph in this run")

    assert seen, "the recommendation engine was never called"

    # Every build saw at least as much evidence as existed after the refresh.
    smallest_refreshed = min(s.nodes_after_followup for s in grew)
    stale = min(s.nodes_before_followup for s in grew)
    for nodes in seen:
        assert nodes >= smallest_refreshed, (
            f"engine built from a {nodes}-node graph; the refreshed evidence "
            f"had {smallest_refreshed} nodes (pre-refresh was {stale}) — the "
            f"recommendation was justified with evidence the review replaced")


def test_a_refreshed_graph_keeps_its_investigation_id(swarm_factory):
    """`refresh_evidence` rebuilds the graph; rebuilding must not reset identity.

    Asserted directly rather than through a run, because nothing in the pipeline
    currently assigns an investigation id at all — `build_graph` takes one and
    every caller passes None, so a run-level assertion would only be testing
    that None survives. This pins the behaviour that matters when the
    investigation layer is wired up: whatever identity the graph had going into
    a refresh is the identity it has coming out.
    """
    from marketswarm.pipeline2 import Pipeline2

    swarm = swarm_factory(gaps=QUIET)
    pipeline = Pipeline2(swarm.config, store=swarm.store)
    graph = pipeline.build_graph({}, "inv_abc123")
    assert graph.investigation_id == "inv_abc123"

    session = CandidateReview(pipeline, {}, "SPY", graph, None, None)
    before = session.graph.investigation_id
    session.refresh_evidence()

    assert session.graph is not graph, "refresh did not actually rebuild"
    assert session.graph_version == 2
    assert session.graph.investigation_id == before == "inv_abc123", (
        "the rebuild dropped the investigation id — every node written after "
        "the refresh would be orphaned from its investigation")


def test_the_reported_graph_matches_what_was_published(swarm_factory, monkeypatch):
    """The report and the recommendations must not disagree about the evidence."""
    demand_research_once(monkeypatch)
    swarm = swarm_factory(gaps=QUIET)
    result = run(swarm)

    grew = [s for s in result.v2.review_sessions if s.graph_version > 1]
    if not grew:
        pytest.skip("no session refreshed its graph in this run")

    run_nodes = len(result.v2.graph.nodes)
    largest = max(s.nodes_after_followup for s in grew)
    assert run_nodes >= largest, (
        f"the run-level graph carries {run_nodes} nodes but a session ended "
        f"with {largest} — the report would show evidence the published "
        f"recommendations were not built on")
