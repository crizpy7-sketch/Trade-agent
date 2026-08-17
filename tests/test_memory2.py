"""Tests for migrations, institutional memory, contribution learning,
after-action review, the experiment lab, security and the read API."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3

import pytest

from marketswarm.after_action import AfterActionReviewer
from marketswarm.api import ReadOnlyAPI
from marketswarm.experiments.lab import (
    Challenger,
    ExperimentLab,
    ExperimentResult,
    PromotionGates,
)
from marketswarm.experiments.scientist import ResearchScientist
from marketswarm.llm_router import Budget, ModelRouter, Tier
from marketswarm.memory.contribution import (
    ContextKey,
    ContributionTracker,
    ablation_contribution,
)
from marketswarm.memory.institutional import (
    InstitutionalMemory,
    Memory,
    MemoryCategory,
    Mistake,
    MistakeTaxonomy,
)
from marketswarm.memory.migrations import LATEST_VERSION, current_version, migrate
from marketswarm.memory.store import MemoryStore, Prediction


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    migrate(s.conn)
    yield s
    s.close()


# ---------------- migrations ----------------

def test_migrations_apply_and_are_idempotent(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    first = migrate(s.conn)
    assert first and current_version(s.conn) == LATEST_VERSION
    assert migrate(s.conn) == [], "re-running must apply nothing"
    s.close()


def test_migrations_preserve_existing_1x_data(tmp_path):
    """The hard requirement: no existing history may be lost."""
    s = MemoryStore(tmp_path / "m.db")
    run_id = s.start_run("2026-01-05")
    pid = s.record_prediction(
        Prediction(run_date="2026-01-05", kind="stock_setup", symbol="SPY",
                   direction="long", probability=0.6, entry=100, target=102, stop=99),
        run_id)
    s.resolve(pid, 1, 2.0, "worked")
    before = s.performance_summary(365)

    migrate(s.conn)

    after = s.performance_summary(365)
    assert after["n"] == before["n"] == 1
    row = s.conn.execute("SELECT * FROM predictions WHERE id=?", (pid,)).fetchone()
    assert row["outcome"] == 1 and row["realized_r"] == 2.0
    s.close()


def test_newer_schema_is_not_downgraded(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    migrate(s.conn)
    s.conn.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?,?,?)",
        (999, "from_the_future", dt.datetime.now(dt.timezone.utc).isoformat()))
    s.conn.commit()
    assert migrate(s.conn) == []
    s.close()


def test_all_2x_tables_exist(store):
    names = {r[0] for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("investigations", "evidence_nodes", "evidence_edges", "recommendations",
              "recommendation_revisions", "agent_context_scores", "market_regimes",
              "memories", "mistakes", "experiments", "experiment_results",
              "calibration_results", "after_action_reviews", "agent_runs",
              "system_events"):
        assert t in names, f"missing table {t}"


# ---------------- institutional memory ----------------

def test_memory_roundtrip_and_recall(store):
    m = InstitutionalMemory(store.conn)
    m.remember(Memory(MemoryCategory.COMPANY, "NVDA gaps fade",
                      "large NVDA gaps have faded 6 of 9 times", subject="NVDA",
                      confidence=0.7, evidence_n=9))
    got = m.recall(MemoryCategory.COMPANY, subject="NVDA")
    assert len(got) == 1 and got[0].title == "NVDA gaps fade"


def test_stale_memory_is_excluded(store):
    m = InstitutionalMemory(store.conn)
    old = Memory(MemoryCategory.EPISODIC, "ancient", "long ago", subject="X")
    old.valid_until = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).isoformat()
    m.remember(old, ttl_days=None)
    assert m.recall(MemoryCategory.EPISODIC, subject="X") == []
    assert len(m.recall(MemoryCategory.EPISODIC, subject="X", include_stale=True)) == 1


def test_superseding_keeps_the_old_version(store):
    m = InstitutionalMemory(store.conn)
    first = Memory(MemoryCategory.SEMANTIC, "v1", "gaps fade")
    m.remember(first)
    m.supersede(first.id, Memory(MemoryCategory.SEMANTIC, "v2", "gaps actually persist"))

    active = m.recall(MemoryCategory.SEMANTIC)
    assert len(active) == 1 and active[0].title == "v2" and active[0].version == 2
    old = store.conn.execute("SELECT active FROM memories WHERE id=?", (first.id,)).fetchone()
    assert old["active"] == 0, "the old belief must remain readable, just inactive"


def test_contradiction_lowers_confidence_without_deleting(store):
    m = InstitutionalMemory(store.conn)
    mem = Memory(MemoryCategory.SEMANTIC, "claim", "body", confidence=0.8)
    m.remember(mem)
    m.record_contradiction(mem.id, "new evidence")
    row = store.conn.execute("SELECT confidence, contradicted_by, active FROM memories "
                             "WHERE id=?", (mem.id,)).fetchone()
    assert row["confidence"] < 0.8
    assert row["contradicted_by"] == "new evidence"
    assert row["active"] == 1


def test_mistake_recording_and_frequency(store):
    m = InstitutionalMemory(store.conn)
    for _ in range(3):
        m.record_mistake(Mistake(
            subject="NVDA", predicted="long", actual="fell",
            taxonomy=[MistakeTaxonomy.OVERWEIGHTED_CORRELATED],
            lesson="counted beta five times", error_r=-1.0))
    freq = m.mistake_frequency()
    assert freq["overweighted_correlated_evidence"] == 3
    assert len(m.mistakes_like(subject="NVDA")) == 3


def test_unpredictable_mistake_does_not_become_a_lesson(store):
    """Bad luck must not teach the system to fear the wrong thing."""
    m = InstitutionalMemory(store.conn)
    m.record_mistake(Mistake(
        subject="SPY", predicted="long", actual="geopolitical shock",
        taxonomy=[MistakeTaxonomy.UNPREDICTABLE], lesson="nobody could forecast this",
        genuinely_unpredictable=True))
    assert m.recall(MemoryCategory.FAILURE) == []


def test_memory_hygiene_prunes(store):
    m = InstitutionalMemory(store.conn)
    stale = Memory(MemoryCategory.EPISODIC, "old", "x")
    stale.valid_until = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).isoformat()
    m.remember(stale, ttl_days=None)
    m.remember(Memory(MemoryCategory.SEMANTIC, "keep", "durable", confidence=0.9,
                      evidence_n=50))
    stats = m.prune()
    assert stats["expired"] >= 1
    assert any(x.title == "keep" for x in m.recall(MemoryCategory.SEMANTIC))


def test_relevant_to_prefers_company_and_failure_memories(store):
    m = InstitutionalMemory(store.conn)
    m.remember(Memory(MemoryCategory.COMPANY, "NVDA specific", "b", subject="NVDA"))
    m.remember(Memory(MemoryCategory.SEMANTIC, "stress regime lesson",
                      "in stress regimes breakouts fail"))
    got = m.relevant_to("NVDA", regime="stress")
    titles = [g.title for g in got]
    assert "NVDA specific" in titles
    assert "stress regime lesson" in titles


# ---------------- contribution learning ----------------

def test_ablation_credits_the_agent_that_helped():
    contribs = {"futures": 0.8, "sentiment": -0.6}
    deltas = ablation_contribution(contribs, outcome=1)
    assert deltas["futures"] > 0, "pushed toward the right answer"
    assert deltas["sentiment"] < 0, "pushed away from it"


def test_ablation_punishes_an_agent_that_was_confidently_wrong():
    deltas = ablation_contribution({"loud": 1.5}, outcome=0)
    assert deltas["loud"] < 0


def test_small_samples_cannot_move_weights(store):
    """Three lucky calls must not produce a 2x weight."""
    t = ContributionTracker(store.conn)
    rows = []
    for i in range(3):
        pid = store.record_prediction(Prediction(
            run_date="2026-01-05", kind="stock_setup", symbol="X", direction="long",
            probability=0.6, raw_probability=0.6, entry=1, target=2, stop=0.5,
            contributing_agents={"lucky": 1.0}, features={"regime": "quiet_trend"}))
        store.resolve(pid, 1, 2.0)
    rows = store.scored_history()
    results = t.update_from_resolved(rows)
    for r in results:
        assert r.final_weight == 1.0, "held neutral below the observation floor"
        assert "observations" in " ".join(r.notes)


def test_weights_move_with_enough_evidence(store):
    t = ContributionTracker(store.conn)
    for i in range(40):
        win = i % 4 != 0                       # 75% hit rate
        pid = store.record_prediction(Prediction(
            run_date="2026-01-05", kind="stock_setup", symbol="X", direction="long",
            probability=0.65, raw_probability=0.65, entry=1, target=2, stop=0.5,
            contributing_agents={"good": 0.7 if win else 0.7,
                                 "bad": -0.7 if win else -0.7},
            features={"regime": "quiet_trend"}))
        store.resolve(pid, 1 if win else 0, 2.0 if win else -1.0)

    results = {r.agent: r for r in t.update_from_resolved(store.scored_history())}
    assert results["good"].final_weight > results["bad"].final_weight
    assert results["good"].mean_contribution > results["bad"].mean_contribution


def test_weight_steps_are_bounded(store):
    t = ContributionTracker(store.conn)
    ctx = ContextKey().key()
    store.conn.execute(
        "INSERT INTO agent_context_scores (agent, context_key, n, weight) VALUES (?,?,?,?)",
        ("a", ctx, 100, 1.0))
    store.conn.commit()
    for i in range(30):
        pid = store.record_prediction(Prediction(
            run_date="2026-01-05", kind="stock_setup", symbol="X", direction="long",
            probability=0.9, raw_probability=0.9, entry=1, target=2, stop=0.5,
            contributing_agents={"a": 2.0}, features={"regime": "any"}))
        store.resolve(pid, 1, 2.0)
    res = [r for r in t.update_from_resolved(store.scored_history()) if r.agent == "a"]
    assert all(abs(r.final_weight - 1.0) <= 0.26 for r in res), "step limit not applied"


def test_context_backoff_finds_a_broader_slice(store):
    t = ContributionTracker(store.conn)
    store.conn.execute(
        "INSERT INTO agent_context_scores (agent, context_key, n, weight) VALUES (?,?,?,?)",
        ("opt", ContextKey().key(), 50, 1.3))
    store.conn.commit()
    w = t.get_weight("opt", ContextKey(regime="stress", event_type="earnings"))
    assert w == 1.3, "should back off to the global slice"


# ---------------- after-action review ----------------

def test_aar_on_a_failed_overconfident_call(store):
    r = AfterActionReviewer(store.conn)
    aar = r.review(
        {"id": "rec_1", "subject": "NVDA", "forecast_probability": 0.75,
         "confidence": 70, "direction": "long",
         "effective_independent_evidence": 1.8},
        realized_r=-1.0, outcome=0,
        contributions={"futures": 0.8, "sentiment": -0.3},
        red_team_findings=[{"objection": "one-sided book"}],
    )
    assert MistakeTaxonomy.OVERCONFIDENT in aar.taxonomy
    assert MistakeTaxonomy.OVERWEIGHTED_CORRELATED in aar.taxonomy
    assert MistakeTaxonomy.RED_TEAM_IGNORED in aar.taxonomy
    assert aar.should_become_memory
    assert "futures" in " ".join(aar.agents_hurt)
    assert "AFTER-ACTION REVIEW" in aar.render()


def test_aar_does_not_blame_an_unforecastable_loss(store):
    r = AfterActionReviewer(store.conn)
    aar = r.review(
        {"id": "rec_2", "subject": "SPY", "forecast_probability": 0.55,
         "confidence": 40, "direction": "long",
         "effective_independent_evidence": 4.0},
        realized_r=-0.4, outcome=0)
    assert MistakeTaxonomy.UNPREDICTABLE in aar.taxonomy
    assert not aar.should_become_memory
    assert "no change warranted" in aar.lesson


def test_aar_persists_and_creates_a_mistake(store):
    r = AfterActionReviewer(store.conn)
    aar = r.review(
        {"id": "rec_3", "subject": "AMD", "forecast_probability": 0.8,
         "confidence": 75, "direction": "long",
         "effective_independent_evidence": 1.2},
        realized_r=-1.0, outcome=0)
    out = r.persist(aar)
    assert out["persisted"] and out["mistake_id"]
    assert store.conn.execute(
        "SELECT COUNT(*) FROM after_action_reviews").fetchone()[0] == 1
    assert InstitutionalMemory(store.conn).mistakes_like(subject="AMD")


def test_aar_records_a_win_as_a_low_confidence_memory(store):
    r = AfterActionReviewer(store.conn)
    aar = r.review({"id": "r", "subject": "MSFT", "forecast_probability": 0.6,
                    "confidence": 55, "direction": "long"},
                   realized_r=1.5, outcome=1)
    r.persist(aar)
    mems = InstitutionalMemory(store.conn).recall(MemoryCategory.COMPANY, subject="MSFT")
    assert mems and mems[0].confidence < 0.5


# ---------------- experiment lab ----------------

def test_champion_is_never_mutated_by_a_challenger(store):
    lab = ExperimentLab(store.conn)
    lab.set_initial_champion("baseline", {"signal_correlation": 0.35})
    champion_cfg = {"signal_correlation": 0.35, "friction_r": 0.06}

    ch = Challenger("higher correlation", "haircut is too small",
                    {"signal_correlation": 0.6})
    patched = ch.apply_to(champion_cfg)

    assert patched["signal_correlation"] == 0.6
    assert champion_cfg["signal_correlation"] == 0.35, "champion was mutated"
    assert patched is not champion_cfg


def test_promotion_requires_every_gate(store):
    lab = ExperimentLab(store.conn, require_human_approval=False)
    lab.set_initial_champion()
    ch = Challenger("c", "h", {"x": 1})
    lab.register(ch)

    champ = ExperimentResult("champ", "hist", 1000, 500, 0.01, 0.5, 0.24, 0.01, 0.5, -10)
    # Better mean R but a far worse drawdown: one metric improving is not enough.
    bad = ExperimentResult(ch.id, "hist", 1000, 500, 0.05, 0.55, 0.24, 0.02, 1.0, -30,
                           deflated_sharpe=0.99, pbo=0.1)
    verdict = lab.evaluate(bad, champ)
    assert not verdict.promoted
    assert any("drawdown" in f for f in verdict.failed)
    assert not lab.promote(ch.id, verdict)


def test_small_sample_challenger_is_blocked(store):
    lab = ExperimentLab(store.conn, require_human_approval=False)
    lab.set_initial_champion()
    ch = Challenger("tiny", "h", {})
    lab.register(ch)
    champ = ExperimentResult("champ", "hist", 1000, 500, 0.0, 0.5, 0.25, 0.0, 0.0, -10)
    tiny = ExperimentResult(ch.id, "hist", 10, 5, 0.5, 0.9, 0.10, 0.2, 3.0, -1,
                            deflated_sharpe=0.99, pbo=0.05)
    v = lab.evaluate(tiny, champ)
    assert not v.promoted
    assert any("sample size" in f for f in v.failed)


def test_a_genuinely_better_challenger_can_be_promoted(store):
    lab = ExperimentLab(store.conn, require_human_approval=False)
    lab.set_initial_champion()
    ch = Challenger("good", "h", {"x": 2})
    lab.register(ch)
    champ = ExperimentResult("champ", "hist", 1000, 500, 0.00, 0.50, 0.250, 0.00, 0.1, -10)
    good = ExperimentResult(ch.id, "hist", 1000, 500, 0.03, 0.55, 0.245, 0.02, 0.9, -10,
                            deflated_sharpe=0.96, pbo=0.2)
    v = lab.evaluate(good, champ)
    assert v.promoted, v.failed
    assert lab.promote(ch.id, v)
    assert lab.champion()["name"] == "good"


def test_human_approval_is_required_by_default(store):
    lab = ExperimentLab(store.conn)          # default: approval required
    lab.set_initial_champion()
    ch = Challenger("auto", "h", {})
    lab.register(ch)
    champ = ExperimentResult("champ", "h", 1000, 500, 0.0, 0.5, 0.25, 0.0, 0.1, -10)
    good = ExperimentResult(ch.id, "h", 1000, 500, 0.03, 0.55, 0.245, 0.02, 0.9, -10,
                            deflated_sharpe=0.96, pbo=0.2)
    v = lab.evaluate(good, champ)
    assert v.promoted
    assert not lab.promote(ch.id, v), "must not self-promote without a human"
    assert lab.champion()["name"] != "auto"
    assert lab.promote(ch.id, v, human_approved=True)


def test_research_scientist_cannot_promote(store):
    """The structural guarantee: self-improvement is not self-deployment."""
    sci = ResearchScientist(store.conn)
    for attr in ("promote", "deploy", "apply_to_production", "set_champion"):
        assert not hasattr(sci, attr), f"scientist must not expose {attr}"


def test_scientist_proposes_only_for_recurring_failures(store):
    m = InstitutionalMemory(store.conn)
    sci = ResearchScientist(store.conn)
    assert sci.propose() == [], "no failures yet"

    for _ in range(6):
        m.record_mistake(Mistake("X", "long", "fell",
                                 [MistakeTaxonomy.OVERWEIGHTED_CORRELATED],
                                 "counted beta repeatedly"))
    proposals = sci.propose()
    assert proposals
    assert any("correlation" in p.title for p in proposals)
    assert all(p.config_patch for p in proposals)


def test_scientist_proposals_register_as_inert_challengers(store):
    m = InstitutionalMemory(store.conn)
    for _ in range(6):
        m.record_mistake(Mistake("X", "l", "f", [MistakeTaxonomy.OVERCONFIDENT], "l"))
    sci = ResearchScientist(store.conn)
    lab = ExperimentLab(store.conn)
    ids = sci.register_proposals(lab, sci.propose())
    assert ids
    for i in ids:
        row = store.conn.execute("SELECT role, status FROM experiments WHERE id=?",
                                 (i,)).fetchone()
        assert row["role"] == "challenger" and row["status"] == "proposed"


# ---------------- model router ----------------

def test_deterministic_tasks_never_call_a_model():
    r = ModelRouter(api_key="sk-ant-test")
    for task in ("anomaly_detection", "probability_fusion", "review_gate", "scoring"):
        d = r.route(task)
        assert d.tier is Tier.DETERMINISTIC and not d.use_llm


def test_router_escalates_on_disagreement():
    r = ModelRouter(api_key="sk-ant-test")
    calm = r.route("headline_classification", importance=0.3, disagreement=0.1)
    fight = r.route("headline_classification", importance=0.9, disagreement=0.8)
    assert calm.tier is Tier.CHEAP
    assert fight.tier is Tier.STRONG


def test_router_degrades_when_the_budget_is_exhausted():
    r = ModelRouter(api_key="sk-ant-test", budget=Budget(daily_usd=0.0001,
                                                         per_run_usd=0.0001))
    d = r.route("narrative", importance=0.9)
    assert d.tier is Tier.DETERMINISTIC and d.downgraded


def test_router_without_a_key_is_fully_deterministic():
    r = ModelRouter(api_key=None)
    d = r.route("narrative", importance=1.0)
    assert d.tier is Tier.DETERMINISTIC
    assert r.complete(d, "sys", "user") is None


# ---------------- read-only API ----------------

def test_api_dashboard_is_safe_on_an_empty_database(store):
    api = ReadOnlyAPI(store.conn)
    dash = api.dashboard()
    for key in ("system", "investigations", "agents", "recommendations",
                "performance", "failures", "experiments", "costs", "memory"):
        assert key in dash


def test_api_exposes_no_mutating_methods(store):
    api = ReadOnlyAPI(store.conn)
    for name in dir(api):
        if name.startswith("_"):
            continue
        assert not any(name.startswith(v) for v in
                       ("write", "insert", "update", "delete", "create", "set_",
                        "promote", "place", "order")), f"{name} looks mutating"


def test_api_read_only_connection_rejects_writes(tmp_path):
    from marketswarm.api import open_api
    s = MemoryStore(tmp_path / "m.db")
    migrate(s.conn)
    s.close()
    api = open_api(tmp_path / "m.db")
    with pytest.raises(sqlite3.OperationalError):
        api.conn.execute("INSERT INTO memories (id, created_at, category, title, body) "
                         "VALUES ('x','now','semantic','t','b')")


def test_api_reports_schema_version(store):
    assert ReadOnlyAPI(store.conn).system_status()["schema_version"] == LATEST_VERSION
