"""Tests for the historical replay machinery.

The most important tests here are the negative ones: that the store refuses
lookahead, that purged folds do not leak, and that the tie-break rule for
ambiguous bars is the pessimistic one. A backtest harness that is merely
"working" but subtly optimistic is worse than none.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from marketswarm.backtest.datastore import LookaheadError, PointInTimeStore
from marketswarm.backtest.features import build_dataset, build_features, resolve_outcome
from marketswarm.backtest.fills import ILLIQUID_PRESET, LIQUID_PRESET, STANDARD_PRESET
from marketswarm.backtest.validation import (
    deflated_sharpe_ratio,
    max_drawdown,
    probability_of_backtest_overfitting,
    purged_walk_forward_splits,
    summarize_returns,
)
from marketswarm.models.logistic import LogisticModel


def synth_frame(symbols=("AAA", "BBB"), n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = dt.date(2020, 1, 1)
    for si, sym in enumerate(symbols):
        price = 100.0
        d = start
        for i in range(n):
            while d.weekday() >= 5:
                d += dt.timedelta(days=1)
            ret = rng.normal(0.0004, 0.013)
            o = price
            c = price * (1 + ret)
            h = max(o, c) * (1 + abs(rng.normal(0, 0.004)))
            l = min(o, c) * (1 - abs(rng.normal(0, 0.004)))
            rows.append({"symbol": sym, "date": d.isoformat(), "open": o, "high": h,
                         "low": l, "close": c, "volume": rng.uniform(1e6, 5e6)})
            price = c
            d += dt.timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture
def store(tmp_path):
    s = PointInTimeStore(tmp_path / "h.db")
    s.ingest_frame(synth_frame(), source="test")
    yield s
    s.close()


# ---------------- point-in-time guarantees ----------------

def test_history_is_strictly_before_the_decision_date(store):
    dates = store.trading_dates()
    asof = dates[100]
    h = store.history_before("AAA", asof, lookback=50)
    assert h is not None and len(h) == 50
    assert h.last_date < asof
    store.assert_no_lookahead(h, asof)


def test_lookahead_is_detected(store):
    dates = store.trading_dates()
    asof = dates[100]
    h = store.history_before("AAA", dates[105], lookback=50)
    with pytest.raises(LookaheadError):
        store.assert_no_lookahead(h, asof)


def test_open_price_is_available_but_not_the_rest_of_the_bar(store):
    """The open is knowable at the open; the close is not. Features may use the
    first and must never use the second."""
    dates = store.trading_dates()
    asof = dates[100]
    o = store.open_price("AAA", asof)
    bar = store.bar_on("AAA", asof)
    assert o == bar.open

    row = build_features(store, "AAA", asof)
    assert row is not None
    assert row.entry == o
    # No feature may equal the session's close, high or low.
    for name, v in row.features.items():
        assert v != bar.close, f"feature {name} leaked the close"


def test_bad_ohlc_rows_are_rejected(tmp_path):
    s = PointInTimeStore(tmp_path / "b.db")
    df = pd.DataFrame([
        {"symbol": "X", "date": "2020-01-02", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1},
        {"symbol": "X", "date": "2020-01-03", "open": 10, "high": 8, "low": 9, "close": 10.5, "volume": 1},
    ])
    s.ingest_frame(df)
    assert s.coverage()["bars"] == 1
    s.close()


def test_event_cannot_be_known_before_it_happens(store):
    with pytest.raises(ValueError):
        store.ingest_event("AAA", dt.date(2020, 6, 1),
                           dt.datetime(2020, 5, 30, 12, 0), "earnings")


def test_events_respect_the_asof_clock(store):
    store.ingest_event("AAA", dt.date(2020, 6, 1), dt.datetime(2020, 6, 1, 16, 30), "8-K")
    assert store.events_known_by(dt.datetime(2020, 6, 1, 9, 0)) == []
    assert len(store.events_known_by(dt.datetime(2020, 6, 2, 9, 0))) == 1


# ---------------- outcome resolution ----------------

def test_both_barriers_touched_scores_as_a_loss(tmp_path):
    """Daily bars do not reveal which came first; the tie-break must be the
    pessimistic one or every volatile day flatters the backtest."""
    s = PointInTimeStore(tmp_path / "r.db")
    s.ingest_frame(pd.DataFrame([{
        "symbol": "X", "date": "2020-01-02", "open": 100, "high": 105,
        "low": 95, "close": 101, "volume": 1e6}]))
    out = s and resolve_outcome(s, "X", dt.date(2020, 1, 2), 100, 103, 97)
    assert out[0] == 0 and out[1] == -1.0
    assert "both" in out[2]

    optimistic = resolve_outcome(s, "X", dt.date(2020, 1, 2), 100, 103, 97, conservative=False)
    assert optimistic[0] == 1
    s.close()


def test_clean_target_and_stop_resolution(tmp_path):
    s = PointInTimeStore(tmp_path / "r.db")
    s.ingest_frame(pd.DataFrame([
        {"symbol": "X", "date": "2020-01-02", "open": 100, "high": 104, "low": 99.5, "close": 103, "volume": 1},
        {"symbol": "X", "date": "2020-01-03", "open": 100, "high": 100.5, "low": 96, "close": 97, "volume": 1},
    ]))
    win = resolve_outcome(s, "X", dt.date(2020, 1, 2), 100, 103, 97)
    assert win[0] == 1 and win[1] == pytest.approx(1.0)
    loss = resolve_outcome(s, "X", dt.date(2020, 1, 3), 100, 103, 97)
    assert loss[0] == 0 and loss[1] == -1.0
    s.close()


def test_neither_barrier_marks_to_close(tmp_path):
    s = PointInTimeStore(tmp_path / "r.db")
    s.ingest_frame(pd.DataFrame([{
        "symbol": "X", "date": "2020-01-02", "open": 100, "high": 101, "low": 99.5,
        "close": 100.6, "volume": 1}]))
    out = resolve_outcome(s, "X", dt.date(2020, 1, 2), 100, 103, 97)
    assert out[0] == 1
    assert 0 < out[1] < 1
    assert "neither" in out[2]
    s.close()


# ---------------- purged walk-forward ----------------

def test_folds_never_train_on_the_future():
    dates = np.array([dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(600)])
    splits = purged_walk_forward_splits(dates, n_splits=4, embargo_days=5)
    assert splits
    for sp in splits:
        assert sp.train_range[1] < sp.test_range[0]
        assert max(dates[sp.train_idx]) < min(dates[sp.test_idx])


def test_embargo_creates_a_real_gap():
    dates = np.array([dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(600)])
    splits = purged_walk_forward_splits(dates, n_splits=4, embargo_days=10)
    for sp in splits:
        gap = (sp.test_range[0] - sp.train_range[1]).days
        assert gap > 10, f"embargo not enforced: {gap} day gap"
        assert sp.embargoed > 0


def test_no_index_appears_in_both_train_and_test():
    dates = np.array([dt.date(2020, 1, 1) + dt.timedelta(days=i // 3) for i in range(900)])
    for sp in purged_walk_forward_splits(dates, n_splits=3, embargo_days=3):
        assert not set(sp.train_idx.tolist()) & set(sp.test_idx.tolist())


# ---------------- deflation and PBO ----------------

def test_deflated_sharpe_punishes_many_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.03, 1.0, 500)
    one = deflated_sharpe_ratio(r, n_trials=1)
    many = deflated_sharpe_ratio(r, n_trials=500)
    assert many.deflated_sharpe < one.deflated_sharpe
    assert many.expected_max_sharpe > one.expected_max_sharpe


def test_pure_noise_is_not_significant():
    rng = np.random.default_rng(2)
    v = deflated_sharpe_ratio(rng.normal(0, 1, 400), n_trials=20)
    assert not v.significant


def test_strong_real_signal_survives_deflation():
    rng = np.random.default_rng(3)
    v = deflated_sharpe_ratio(rng.normal(0.35, 1.0, 800), n_trials=5)
    assert v.significant


def _mean_pbo(make_perf, seeds=range(6), n_partitions=8) -> float:
    """PBO on a single seed is noisy — average a few before asserting."""
    vals = []
    for s in seeds:
        out = probability_of_backtest_overfitting(make_perf(s), n_partitions=n_partitions)
        if not np.isnan(out["pbo"]):
            vals.append(out["pbo"])
    return float(np.mean(vals))


def test_pbo_separates_noise_selection_from_a_real_winner():
    """The property that matters: choosing among pure noise must score far
    worse than choosing when one configuration is genuinely better."""
    def noise(seed):
        return np.random.default_rng(seed).normal(0, 1, size=(600, 8))

    def real_winner(seed):
        p = np.random.default_rng(seed + 100).normal(0, 1, size=(600, 8))
        p[:, 3] += 1.2
        return p

    pbo_noise = _mean_pbo(noise)
    pbo_real = _mean_pbo(real_winner)
    assert pbo_real < 0.05, f"a genuine winner should survive selection (got {pbo_real:.3f})"
    assert pbo_noise > pbo_real + 0.2, (
        f"noise selection ({pbo_noise:.3f}) must score far worse than a real "
        f"winner ({pbo_real:.3f})"
    )


def test_pbo_needs_enough_observations():
    out = probability_of_backtest_overfitting(np.random.default_rng(0).normal(size=(10, 4)))
    assert np.isnan(out["pbo"])
    assert "need at least" in out["note"]


def test_summaries():
    r = np.array([1.0, -1.0, 2.0, -1.0, 0.5])
    s = summarize_returns(r)
    assert s["n"] == 5 and s["hit_rate"] == pytest.approx(0.6)
    assert s["profit_factor"] == pytest.approx(3.5 / 2.0)
    assert max_drawdown(np.array([1.0, 2.0, 0.5, 3.0]))["max_drawdown"] < 0


# ---------------- fills ----------------

def test_costs_are_ordered_by_liquidity():
    args = dict(entry=100.0, stop=99.0, shares=500, adv=2_000_000, daily_vol=0.015)
    liq = LIQUID_PRESET.round_trip_cost_r(**args)
    std = STANDARD_PRESET.round_trip_cost_r(**args)
    ill = ILLIQUID_PRESET.round_trip_cost_r(**args)
    assert liq < std < ill
    assert 0.01 < liq < 0.10, "liquid round-trip should be a few percent of R"


def test_impact_scales_with_size_and_volatility():
    small = STANDARD_PRESET.fill_equity(100, 1_000, 5_000_000, "buy", daily_vol=0.015)
    large = STANDARD_PRESET.fill_equity(100, 500_000, 5_000_000, "buy", daily_vol=0.015)
    assert large.cost_bps > small.cost_bps
    calm = STANDARD_PRESET.fill_equity(100, 100_000, 5_000_000, "buy", daily_vol=0.008)
    wild = STANDARD_PRESET.fill_equity(100, 100_000, 5_000_000, "buy", daily_vol=0.05)
    assert wild.cost_bps > calm.cost_bps


def test_buy_fills_above_and_sell_below():
    b = STANDARD_PRESET.fill_equity(100, 1000, 1e6, "buy")
    s = STANDARD_PRESET.fill_equity(100, 1000, 1e6, "sell")
    assert b.filled > 100 > s.filled


def test_option_fill_pays_part_of_the_spread():
    f = STANDARD_PRESET.fill_option(bid=1.00, ask=1.20, contracts=5, side="buy")
    assert 1.10 < f.filled <= 1.20
    assert f.cost_bps > 0


def test_gap_through_stop_fills_at_the_open():
    f = STANDARD_PRESET.stop_exit(stop_price=99.0, next_open=96.0, side="sell")
    assert f.filled == 96.0
    assert "gapped" in f.note


# ---------------- model ----------------

def test_logistic_learns_a_separable_signal():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(800, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.4, 800) > 0).astype(float)
    m = LogisticModel(feature_names=list("abcd")).fit(X, y)
    acc = ((m.predict_proba(X) > 0.5) == (y > 0.5)).mean()
    assert acc > 0.85
    top = m.importances(2)
    assert top[0][0] == "a"


def test_logistic_on_pure_noise_stays_near_the_base_rate():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(600, 5))
    y = (rng.random(600) < 0.5).astype(float)
    m = LogisticModel().fit(X, y)
    p = m.predict_proba(X)
    assert 0.3 < p.mean() < 0.7
    assert p.std() < 0.2, "model is confidently fitting noise"


def test_model_roundtrips_through_json(tmp_path):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(float)
    m = LogisticModel(feature_names=["a", "b", "c"]).fit(X, y)
    p = tmp_path / "m.json"
    m.save(p)
    loaded = LogisticModel.load(p)
    assert loaded is not None
    assert np.allclose(loaded.predict_proba(X), m.predict_proba(X))


def test_constant_feature_does_not_explode():
    X = np.column_stack([np.ones(300), np.random.default_rng(9).normal(size=300)])
    y = (X[:, 1] > 0).astype(float)
    m = LogisticModel().fit(X, y)
    assert np.all(np.isfinite(m.predict_proba(X)))


# ---------------- end-to-end ----------------

def test_dataset_builds_and_labels_are_binary(store):
    dates = store.trading_dates()[100:160]
    X, y, meta = build_dataset(store, ["AAA", "BBB"], dates, progress_every=0)
    assert len(X) == len(y) == len(meta) > 50
    assert set(np.unique(y)).issubset({0.0, 1.0})
    assert X.shape[1] == 20
    assert np.all(np.isfinite(X))
    for m in meta:
        assert m["date"] in dates
