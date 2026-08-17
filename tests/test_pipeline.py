"""Full-swarm integration test on synthetic data.

Exercises the path that matters most and that a live run cannot be relied on to
cover: every agent producing output, signals fusing, ideas being constructed
with real Monte-Carlo probabilities, and the report rendering.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from marketswarm import clock
from marketswarm.agents import ALL_AGENTS, SwarmContext
from marketswarm.config import Config
from marketswarm.memory import LearningEngine, MemoryStore, Prediction
from marketswarm.orchestrator import SwarmResult, stage_agents
from marketswarm.report import render_html, render_markdown
from marketswarm.stats.calibration import PlattCalibrator

from .fakes import FakeEarnings, FakeEcon, FakeEdgar, FakeMarket, FakeNews, FakeOptions


UNIVERSE = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMD"]


def build_context(tmp_path, gaps=None) -> SwarmContext:
    cfg = Config()
    cfg.universe = UNIVERSE
    cfg.index_symbols = ["SPY", "QQQ"]
    cfg.data_dir = tmp_path
    cfg.report_dir = tmp_path / "reports"
    cfg.calibrator = PlattCalibrator()
    cfg.ensure_dirs()

    market = FakeMarket(UNIVERSE, gaps=gaps)
    run_date = dt.date(2026, 8, 17)  # a Monday
    return SwarmContext(
        run_date=run_date,
        prev_session=clock.previous_trading_day(run_date),
        universe=UNIVERSE,
        index_symbols=cfg.index_symbols,
        market=market,
        options=FakeOptions(market),
        news=FakeNews(),
        econ=FakeEcon(),
        edgar=FakeEdgar(),
        earnings=FakeEarnings(),
        config=cfg,
    )


async def run_swarm(ctx: SwarmContext):
    for stage in stage_agents(ALL_AGENTS):
        reports = await asyncio.gather(*(a().execute(ctx) for a in stage))
        for r in reports:
            ctx.reports[r.agent] = r
    return ctx.reports


@pytest.fixture
def reports(tmp_path):
    ctx = build_context(tmp_path)
    return asyncio.run(run_swarm(ctx)), ctx


def test_every_agent_reports_usable(reports):
    reps, _ = reports
    assert len(reps) == len(ALL_AGENTS)
    failed = [r.agent for r in reps.values() if not r.usable]
    assert not failed, f"agents failed on synthetic data: {failed}"


def test_agents_produce_evidence_with_provenance(reports):
    reps, _ = reports
    evidence = [e for r in reps.values() for e in r.evidence]
    assert len(evidence) > 10
    assert all(e.source for e in evidence)
    assert all(0 <= e.reliability <= 1 for e in evidence)


def test_cross_verification_fuses_signals(reports):
    reps, _ = reports
    cv = reps["cross_verify"]
    assert cv.usable
    assert 0.0 < cv.data["probability"] < 1.0
    assert cv.data["raw_signal_count"] >= 3
    # Correlation haircut must bite: effective signals below the raw count.
    assert cv.data["effective_n"] < cv.data["raw_signal_count"]
    assert 1 <= cv.data["confidence_score"] <= 99


def test_risk_agent_sizes_down_from_base(reports):
    reps, _ = reports
    risk = reps["risk"]
    assert 0 < risk.data["suggested_risk_pct"] <= Config().base_risk_pct
    assert risk.data["portfolio_heat"]["effective"] > 0


def test_playbook_produces_bracketed_ideas(reports):
    reps, _ = reports
    pb = reps["playbook"]
    assert pb.usable
    ideas = pb.data["calls"] + pb.data["puts"] + pb.data["stocks"]
    assert ideas, "no ideas constructed from synthetic data"

    for i in ideas:
        assert i["symbol"] in UNIVERSE
        assert 0.0 < i["probability"] < 1.0
        # Ideas may be published with negative expectancy, but only when
        # explicitly flagged as not clearing the cost bar.
        assert i["clears_bar"] == (i["expected_r"] > 0)
        assert i["ev_verdict"]
        assert i["invalidation"], "every idea must carry invalidating conditions"
        assert i["rationale"]
        assert i["math_note"]
        if i["direction"] == "long":
            assert i["stop"] < i["entry"] < i["target"]
        else:
            assert i["target"] < i["entry"] < i["stop"]


def test_option_ideas_carry_strike_and_premium_levels(reports):
    reps, _ = reports
    for i in reps["playbook"].data["calls"] + reps["playbook"].data["puts"]:
        assert i["strike"] is not None
        assert i["expiration"]
        assert i["option_entry"] and i["option_target"] and i["option_stop"]


def test_ideas_are_ranked_by_composite_score(reports):
    reps, _ = reports
    stocks = reps["playbook"].data["stocks"]
    # Ideas clearing the cost bar always rank above those that do not.
    flags = [s["clears_bar"] for s in stocks]
    assert flags == sorted(flags, reverse=True)

    tradeable = [s for s in stocks if s["clears_bar"]]
    if len(tradeable) > 1:
        scores = [s["expected_r"] * (0.5 + 0.5 * s["confidence"] / 100) *
                  (0.4 + 0.6 * s["liquidity"]) for s in tradeable]
        assert scores == sorted(scores, reverse=True)

    watchlist = [s for s in stocks if not s["clears_bar"]]
    if len(watchlist) > 1:
        evs = [s["expected_r"] for s in watchlist]
        assert evs == sorted(evs, reverse=True)


def test_report_renders_with_playbook_and_disclaimer(reports, tmp_path):
    reps, ctx = reports
    result = SwarmResult(run_date=ctx.run_date, market_open=True, reports=reps)
    result.probability = reps["cross_verify"].data["probability"]
    result.confidence = reps["cross_verify"].data["confidence_score"]
    pb = reps["playbook"]
    result.ideas = {"calls": pb.data["calls"], "puts": pb.data["puts"], "stocks": pb.data["stocks"]}

    md = render_markdown(result)
    assert "# Day Trading Playbook" in md
    assert "1. Best Call Options" in md
    assert "2. Best Put Options" in md
    assert "3. Best Stocks for Day Trading" in md
    assert "not financial advice" in md
    assert "Invalidated if:" in md
    for section in ("Overnight Scan", "Options Flow", "Cross-Verification", "Risk Assessment"):
        assert section in md

    html = render_html(result)
    assert html.startswith("<!doctype html>")
    assert "prefers-color-scheme" in html
    assert "<script" not in html.lower()


def test_closed_market_report_is_short():
    result = SwarmResult(run_date=dt.date(2026, 7, 3), market_open=False,
                         closed_reason="Independence Day")
    md = render_markdown(result)
    assert "Market closed" in md
    assert "Day Trading Playbook" not in md


def test_gap_direction_flows_into_bias(tmp_path):
    """A large down gap on every name should not yield an all-long playbook."""
    ctx = build_context(tmp_path, gaps={s: -2.5 for s in UNIVERSE})
    reps = asyncio.run(run_swarm(ctx))
    pb = reps["playbook"]
    shorts = [i for i in pb.data["stocks"] if i["direction"] == "short"]
    assert shorts, "uniformly negative gaps produced no short setups"


# ---------------- memory & learning ----------------

def test_predictions_persist_and_resolve(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    run_id = store.start_run("2026-08-17")
    pid = store.record_prediction(
        Prediction(run_date="2026-08-17", kind="stock_setup", symbol="SPY", direction="long",
                   probability=0.58, entry=500.0, target=503.0, stop=498.0,
                   features={"regime": "quiet_trend"}, contributing_agents={"futures": 0.4}),
        run_id,
    )
    assert len(store.unresolved_predictions()) == 1

    engine = LearningEngine(store)
    bars = {"highs": [500.5, 501.2, 503.4], "lows": [499.5, 499.9, 501.0],
            "closes": [500.2, 500.9, 503.1]}
    outcome = engine.resolve_prediction(store.unresolved_predictions()[0], bars)
    assert outcome[0] == 1 and outcome[1] == pytest.approx(1.5)
    store.resolve(pid, *outcome)
    assert not store.unresolved_predictions()
    store.close()


def test_stop_before_target_is_a_loss(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    pid = store.record_prediction(
        Prediction(run_date="2026-08-17", kind="stock_setup", symbol="SPY", direction="long",
                   probability=0.6, entry=500.0, target=503.0, stop=498.0)
    )
    engine = LearningEngine(store)
    row = store.unresolved_predictions()[0]
    # Stop is touched on bar 1, target only on bar 2 — must score as a loss.
    bars = {"highs": [500.1, 503.5], "lows": [497.5, 502.0], "closes": [498.0, 503.2]}
    outcome = engine.resolve_prediction(row, bars)
    assert outcome[0] == 0 and outcome[1] == -1.0
    store.close()


def test_learning_updates_weights_and_calibrator(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    engine = LearningEngine(store)

    # 60 resolved calls where "futures" pushed correctly and "sentiment" did not.
    for i in range(60):
        win = i % 3 != 0            # 2/3 win rate
        pid = store.record_prediction(
            Prediction(run_date="2026-06-01", kind="stock_setup", symbol="SPY",
                       direction="long", probability=0.88, raw_probability=0.88,
                       entry=100, target=102, stop=99,
                       features={"regime": "quiet_trend", "setup": "stock_setup"},
                       contributing_agents={"futures": 0.5 if win else 0.5,
                                            "sentiment": -0.5 if win else -0.5})
        )
        store.resolve(pid, 1 if win else 0, 2.0 if win else -1.0)

    result = engine.score_and_learn()
    assert result.resolved == 60
    weights = store.get_agent_weights()
    assert weights["futures"] > weights["sentiment"]
    # Forecasts of 88% against a 67% hit rate must be pulled toward the middle.
    cal = engine.load_calibrator()
    assert cal.transform(0.88) < 0.88
    store.close()


def test_lessons_require_enough_evidence(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    engine = LearningEngine(store)
    for i in range(20):
        pid = store.record_prediction(
            Prediction(run_date="2026-06-01", kind="stock_setup", symbol="XYZ",
                       direction="long", probability=0.6, raw_probability=0.6,
                       entry=100, target=102, stop=99,
                       features={"regime": "stress", "setup": "stock_setup"})
        )
        store.resolve(pid, 0, -1.0)   # stress regime always loses
    for i in range(20):
        pid = store.record_prediction(
            Prediction(run_date="2026-06-02", kind="stock_setup", symbol="ABC",
                       direction="long", probability=0.6, raw_probability=0.6,
                       entry=100, target=102, stop=99,
                       features={"regime": "quiet_trend", "setup": "stock_setup"})
        )
        store.resolve(pid, 1, 2.0)

    engine.score_and_learn()
    lessons = store.active_lessons()
    assert any("regime=stress" in l["scope"] for l in lessons)
    assert all(l["evidence_n"] >= 12 for l in lessons)
    store.close()


def test_performance_summary_reports_calibration_gap(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    for i in range(10):
        pid = store.record_prediction(
            Prediction(run_date=dt.date.today().isoformat(), kind="call_idea", symbol="SPY",
                       direction="long", probability=0.9, entry=1, target=2, stop=0.5)
        )
        store.resolve(pid, 1 if i < 5 else 0, 1.0 if i < 5 else -1.0)
    perf = store.performance_summary(30)
    assert perf["by_kind"]["call_idea"]["hit_rate"] == pytest.approx(0.5)
    assert perf["by_kind"]["call_idea"]["calibration_gap"] == pytest.approx(0.4)
    store.close()


def test_red_team_objections_reach_the_report(reports, tmp_path):
    """The red team is worthless if its objections never get rendered."""
    reps, ctx = reports
    rt = reps["red_team"]
    assert rt.usable and rt.data["objections"], "red team produced nothing to render"

    result = SwarmResult(run_date=ctx.run_date, market_open=True, reports=reps)
    pb = reps["playbook"]
    result.ideas = {"calls": pb.data["calls"], "puts": pb.data["puts"], "stocks": pb.data["stocks"]}
    md = render_markdown(result)

    assert "Red Team" in md
    assert rt.data["objections"][0]["objection"][:40] in md
    assert "*Test:*" in md


def test_demo_flag_marks_every_idea_card(reports):
    """A sample report must be unmistakable even from a screenshot of one card —
    fake prices that look real are the most dangerous output this thing can make."""
    reps, ctx = reports
    result = SwarmResult(run_date=ctx.run_date, market_open=True, reports=reps)
    pb = reps["playbook"]
    result.ideas = {"calls": pb.data["calls"], "puts": pb.data["puts"], "stocks": pb.data["stocks"]}

    plain = render_markdown(result, demo=False)
    assert "SAMPLE" not in plain

    demo = render_markdown(result, demo=True)
    assert demo.count("SAMPLE REPORT — SYNTHETIC DATA") >= 2   # top and above the playbook
    for i in result.ideas["calls"] + result.ideas["puts"]:
        assert f"SAMPLE / FAKE PRICE — {i['symbol']}" in demo

    html = render_html(result, demo=True)
    assert 'class="demo-bar"' in html
    assert "position:sticky" in html
