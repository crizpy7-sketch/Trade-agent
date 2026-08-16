import math

import numpy as np
import pytest

from marketswarm.stats import bayes, calibration, distributions as dist, edge, regime, technicals as ta


# ---------------- bayes ----------------

def test_fuse_empty_returns_prior():
    r = bayes.fuse([], prior=0.6)
    assert r.probability == pytest.approx(0.6)


def test_fuse_agreeing_signals_move_past_each_individually():
    sigs = [bayes.Signal(f"s{i}", 0.60, 1.0) for i in range(4)]
    r = bayes.fuse(sigs, correlation=0.0)
    assert r.probability > 0.60
    assert r.effective_n == pytest.approx(4.0, abs=0.01)


def test_correlation_haircut_reduces_confidence():
    sigs = [bayes.Signal(f"s{i}", 0.65, 1.0) for i in range(8)]
    independent = bayes.fuse(sigs, correlation=0.0)
    correlated = bayes.fuse(sigs, correlation=0.6)
    assert correlated.probability < independent.probability
    assert correlated.effective_n < independent.effective_n


def test_opposing_signals_cancel():
    r = bayes.fuse([bayes.Signal("up", 0.70, 1.0), bayes.Signal("down", 0.30, 1.0)])
    assert r.probability == pytest.approx(0.5, abs=0.01)


def test_llr_is_capped():
    r = bayes.fuse([bayes.Signal("crazy", 0.99999, 1.0)], max_abs_llr=2.2)
    assert r.probability < 0.91  # sigmoid(2.2) ≈ 0.900


def test_beta_posterior_weight_respects_sample_size():
    lucky = bayes.BetaPosterior().update(4, 0)      # 4/4
    grinder = bayes.BetaPosterior().update(120, 80)  # 60% over 200
    assert grinder.reliability_weight() > lucky.reliability_weight()


def test_coin_flip_source_gets_no_weight():
    coin = bayes.BetaPosterior().update(100, 100)
    assert coin.reliability_weight() == pytest.approx(0.15, abs=0.02)


def test_hierarchical_shrinkage_pulls_small_samples_to_mean():
    rates = {"a": (5, 5), "b": (3, 3), "small": (2, 2), "hot": (4, 4)}
    out = bayes.hierarchical_shrink({"steady": (100, 200), "hot": (4, 4)})
    assert out["hot"] < 1.0  # not left at 100%
    assert 0.4 < out["steady"] < 0.6


def test_confidence_label_rewards_breadth():
    _, thin = bayes.probability_to_confidence_label(0.62, n_eff=1.2, dispersion=0.4)
    _, broad = bayes.probability_to_confidence_label(0.62, n_eff=8.0, dispersion=0.05)
    assert broad > thin


# ---------------- calibration ----------------

def test_brier_of_perfect_and_worst():
    assert calibration.brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert calibration.brier_score(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == 1.0


def test_brier_decomposition_identity():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.1, 0.9, 400)
    y = (rng.uniform(size=400) < p).astype(float)
    d = calibration.brier_decomposition(p, y, n_bins=10)
    # BS ≈ reliability - resolution + uncertainty (exact up to binning)
    assert abs(d.brier - (d.reliability - d.resolution + d.uncertainty)) < 0.02
    assert d.skill_score > 0


def test_platt_shrinks_overconfident_forecasts():
    rng = np.random.default_rng(11)
    true_p = rng.uniform(0.35, 0.65, 500)
    y = (rng.uniform(size=500) < true_p).astype(float)
    # Report forecasts far more extreme than reality.
    overconfident = np.clip(0.5 + (true_p - 0.5) * 3.0, 0.02, 0.98)
    cal = calibration.PlattCalibrator().fit(overconfident, y)
    assert cal.a < 1.0
    assert abs(cal.transform(0.9) - 0.5) < abs(0.9 - 0.5)


def test_platt_identity_on_tiny_sample():
    cal = calibration.PlattCalibrator().fit(np.array([0.6, 0.4]), np.array([1.0, 0.0]))
    assert cal.a == 1.0 and cal.b == 0.0
    assert cal.transform(0.7) == pytest.approx(0.7, abs=1e-6)


def test_sprt_detects_real_skill_and_no_skill():
    assert calibration.sequential_sprt(70, 100)[0] == "skill confirmed"
    assert calibration.sequential_sprt(40, 100)[0] == "no skill — stand down"
    assert calibration.sequential_sprt(6, 10)[0] == "continue"


def test_reliability_curve_bins_sum_to_n():
    p = np.linspace(0.01, 0.99, 100)
    y = (p > 0.5).astype(float)
    rows = calibration.reliability_curve(p, y, n_bins=10)
    assert sum(r["n"] for r in rows) == 100


# ---------------- distributions ----------------

def test_implied_move_from_straddle():
    im = dist.implied_move_from_straddle(underlying=100.0, straddle_price=2.0, days_to_expiry=1)
    assert im.implied_move_abs == pytest.approx(2.0 / 0.7979, rel=1e-3)
    assert im.implied_move_pct == pytest.approx(im.implied_move_abs / 100)
    assert im.implied_vol_annual > 0


def test_barrier_symmetric_bracket_is_near_coinflip():
    b = dist.barrier_probabilities(100, 101, 99, sigma_daily=0.02, n_paths=6000, seed=3)
    assert abs(b.p_target_first - b.p_stop_first) < 0.10
    assert b.p_target_first + b.p_stop_first + b.p_neither == pytest.approx(1.0, abs=1e-6)


def test_barrier_tight_target_hits_more_often():
    tight = dist.barrier_probabilities(100, 100.3, 98, sigma_daily=0.02, n_paths=6000, seed=5)
    wide = dist.barrier_probabilities(100, 104, 98, sigma_daily=0.02, n_paths=6000, seed=5)
    assert tight.p_target_first > wide.p_target_first


def test_barrier_drift_helps_the_long():
    up = dist.barrier_probabilities(100, 102, 98, sigma_daily=0.02, drift_daily=0.02, n_paths=6000, seed=9)
    flat = dist.barrier_probabilities(100, 102, 98, sigma_daily=0.02, drift_daily=0.0, n_paths=6000, seed=9)
    assert up.p_target_first > flat.p_target_first


def test_realized_and_parkinson_vol_agree_roughly():
    rng = np.random.default_rng(4)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    highs, lows = closes * 1.006, closes * 0.994
    rv = dist.realized_volatility(closes, 20)
    pk = dist.parkinson_volatility(highs, lows, 20)
    assert 0.002 < rv < 0.03 and 0.002 < pk < 0.03


def test_gap_conditional_stats_reports_thin_samples():
    history = np.array([[1.0, 0.5], [1.1, -0.2]])
    out = dist.gap_conditional_stats(history, 1.0)
    assert out["n"] < 3 and "insufficient" in out["note"]


def test_vol_of_vol_ratio_reads():
    assert "rich" in dist.vol_of_vol_ratio(0.30, 0.18)["read"]
    assert "cheap" in dist.vol_of_vol_ratio(0.14, 0.20)["read"]


# ---------------- technicals ----------------

def test_rsi_bounds_and_direction():
    up = np.arange(1, 60, dtype=float)
    assert ta.rsi(up) > 90
    assert ta.rsi(up[::-1]) < 10


def test_atr_positive():
    rng = np.random.default_rng(2)
    c = 100 + np.cumsum(rng.normal(0, 1, 60))
    assert ta.atr(c + 1, c - 1, c) > 0


def test_vwap_between_extremes():
    h = np.array([11.0, 12.0]); l = np.array([9.0, 10.0])
    c = np.array([10.0, 11.0]); v = np.array([100.0, 300.0])
    assert 9 < ta.vwap(h, l, c, v) < 12


def test_trend_state_detects_uptrend():
    closes = np.linspace(100, 130, 80)
    t = ta.trend_state(closes)
    assert t.direction == "up" and t.ema9 > t.ema21


def test_cluster_levels_finds_repeated_touches():
    base = np.concatenate([np.linspace(100, 110, 30), np.linspace(110, 100, 30),
                           np.linspace(100, 110, 30)])
    levels = ta.cluster_levels(base + 0.5, base - 0.5, base, current=105)
    assert levels
    assert all(0 <= lv.strength <= 1 for lv in levels)
    below, above = ta.nearest_levels(levels, 105)
    assert below is None or below.price < 105
    assert above is None or above.price > 105


def test_pivots_ordered():
    p = ta.pivot_levels(110, 90, 100)
    assert p["S2"] < p["S1"] < p["P"] < p["R1"] < p["R2"]


def test_relative_volume():
    assert ta.relative_volume(200, np.array([100.0] * 20)) == pytest.approx(2.0)


# ---------------- edge ----------------

def test_evaluate_trade_rejects_negative_expectancy():
    t = edge.evaluate_trade(p_win=0.30, entry=100, target=101, stop=99)
    assert not t.acceptable and t.expected_r < 0


def test_evaluate_trade_accepts_real_edge():
    t = edge.evaluate_trade(p_win=0.55, entry=100, target=103, stop=99)
    assert t.acceptable and t.edge_after_costs > 0
    assert t.reward_risk == pytest.approx(3.0)
    assert t.breakeven_p == pytest.approx(0.25)


def test_kelly_is_fractional_and_capped():
    t = edge.evaluate_trade(p_win=0.90, entry=100, target=110, stop=99, max_risk_pct=1.0)
    assert t.kelly_fraction > 0.8            # raw Kelly is huge
    assert t.suggested_risk_pct <= 1.0       # but sizing is capped


def test_position_size_respects_risk():
    out = edge.kelly_position_size(50_000, 1.0, entry=100, stop=98)
    assert out["units"] == 250
    assert out["risk_dollars"] == pytest.approx(500.0)


def test_portfolio_heat_flags_correlated_stack():
    heat = edge.portfolio_heat([1.5, 1.5, 1.5], correlation=0.8)
    assert heat["effective"] > 2.5
    assert heat["warning"]


def test_ranking_drops_negative_ev_and_favours_liquidity():
    ranked = edge.rank_opportunities([
        {"symbol": "A", "expected_r": -0.1, "confidence": 90, "liquidity_score": 1.0},
        {"symbol": "B", "expected_r": 0.4, "confidence": 70, "liquidity_score": 1.0},
        {"symbol": "C", "expected_r": 0.4, "confidence": 70, "liquidity_score": 0.1},
    ], top_n=3)
    assert [r["symbol"] for r in ranked] == ["B", "C"]


def test_theta_burn_scales_with_hold():
    assert edge.theta_burn_pct(2.0, 0.5, 6.5) > edge.theta_burn_pct(2.0, 0.5, 1.0)


# ---------------- regime ----------------

def test_classify_regime_labels():
    rng = np.random.default_rng(1)
    calm_trend = 100 * np.exp(np.cumsum(rng.normal(0.0015, 0.005, 300)))
    r = regime.classify_regime(calm_trend, vix=13.0)
    assert r.label in ("quiet_trend", "choppy")

    stressed = 100 * np.exp(np.cumsum(rng.normal(-0.002, 0.03, 300)))
    assert regime.classify_regime(stressed, vix=32.0).label == "stress"


def test_regime_needs_history():
    assert regime.classify_regime(np.array([100.0, 101.0])).label == "unknown"


def test_garch_forecast_shape():
    rng = np.random.default_rng(6)
    out = regime.garch_forecast(rng.normal(0, 0.01, 300))
    assert out["sigma_daily"] > 0 and out["sigma_annual"] > out["sigma_daily"]


def test_hurst_detects_trend_vs_noise():
    rng = np.random.default_rng(21)
    trending = np.cumsum(rng.normal(0.02, 0.004, 400))   # drift dominates noise
    random_walk = np.cumsum(rng.normal(0, 0.01, 400))
    mean_reverting = rng.normal(100, 1, 400)             # no accumulation at all
    assert regime.hurst_exponent(trending) > 0.5
    assert regime.hurst_exponent(mean_reverting) < 0.4
    assert regime.hurst_exponent(mean_reverting) < regime.hurst_exponent(random_walk)


def test_hurst_degenerate_series_returns_neutral():
    assert regime.hurst_exponent(np.ones(300)) == 0.5


def test_vix_term_structure_states():
    assert regime.vix_term_structure_signal(28, 24)["state"] == "backwardation"
    assert regime.vix_term_structure_signal(14, 18)["state"] == "steep contango"
    assert regime.vix_term_structure_signal(None, 18)["state"] == "unknown"
