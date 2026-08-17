"""Point-in-time feature extraction.

Every feature here is computed from bars strictly before the decision date,
plus the opening print of the decision date itself — which a trader entering
at or after the open genuinely knows.

That single exception is the one place lookahead could creep in, so it is
isolated in `gap_pct` and nowhere else. Nothing in this module may touch the
decision day's high, low, close or volume.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass

import numpy as np

from ..stats import distributions as dist
from ..stats import regime as rg
from ..stats import technicals as ta
from .datastore import History, PointInTimeStore

FEATURE_NAMES = [
    "gap_pct",
    "trend_score",
    "ema9_dist",
    "ema21_dist",
    "ema50_dist",
    "rsi",
    "atr_pct",
    "rvol",
    "percent_b",
    "bandwidth",
    "dist_support",
    "dist_resistance",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "vol_ratio",
    "hurst",
    "squeeze",
    "dow",
]


@dataclass
class FeatureRow:
    symbol: str
    date: dt.date
    entry: float
    atr: float
    sigma_daily: float
    support: float | None
    resistance: float | None
    regime: str
    features: dict[str, float]

    def vector(self) -> np.ndarray:
        return np.array([self.features.get(n, 0.0) for n in FEATURE_NAMES], dtype=float)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        return d


def _safe(x: float, default: float = 0.0) -> float:
    return default if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else float(x)


def build_features(
    store: PointInTimeStore,
    symbol: str,
    date: dt.date,
    market_hist: History | None = None,
    lookback: int = 250,
    min_bars: int = 60,
) -> FeatureRow | None:
    """Feature vector for one symbol on one session, as knowable at the open."""
    hist = store.history_before(symbol, date, lookback)
    if hist is None or len(hist) < min_bars:
        return None
    store.assert_no_lookahead(hist, date)

    entry = store.open_price(symbol, date)
    if not entry or entry <= 0:
        return None

    closes, highs, lows, vols = hist.closes, hist.highs, hist.lows, hist.volumes
    prev_close = float(closes[-1])

    atr = _safe(ta.atr(highs, lows, closes), entry * 0.015)
    if atr <= 0:
        return None
    sigma = _safe(dist.realized_volatility(closes, 20), 0.015)
    if sigma <= 0:
        sigma = 0.015

    trend = ta.trend_state(closes)
    bb = ta.bollinger(closes)
    levels = ta.cluster_levels(highs, lows, closes, entry)
    support, resistance = ta.nearest_levels(levels, entry)
    regime = rg.classify_regime(market_hist.closes if market_hist is not None else closes)

    def ret(n: int) -> float:
        return float(closes[-1] / closes[-1 - n] - 1) * 100 if closes.size > n else 0.0

    vol60 = _safe(dist.realized_volatility(closes, 60), sigma)

    feats = {
        "gap_pct": (entry / prev_close - 1) * 100 if prev_close else 0.0,
        "trend_score": trend.strength * (1 if trend.direction == "up" else -1 if trend.direction == "down" else 0),
        "ema9_dist": (entry / _safe(trend.ema9, entry) - 1) * 100 if _safe(trend.ema9) else 0.0,
        "ema21_dist": (entry / _safe(trend.ema21, entry) - 1) * 100 if _safe(trend.ema21) else 0.0,
        "ema50_dist": (entry / _safe(trend.ema50, entry) - 1) * 100 if _safe(trend.ema50) else 0.0,
        "rsi": _safe(ta.rsi(closes), 50.0),
        "atr_pct": atr / entry * 100,
        "rvol": _safe(ta.relative_volume(float(vols[-1]), vols[-20:]), 1.0),
        "percent_b": _safe(bb.get("percent_b"), 0.5),
        "bandwidth": _safe(bb.get("bandwidth"), 0.05) * 100,
        "dist_support": (entry / support.price - 1) * 100 if support else 5.0,
        "dist_resistance": (resistance.price / entry - 1) * 100 if resistance else 5.0,
        "ret_1d": ret(1),
        "ret_5d": ret(5),
        "ret_20d": ret(20),
        "vol_20d": sigma * 100,
        "vol_ratio": sigma / vol60 if vol60 > 0 else 1.0,
        "hurst": rg.hurst_exponent(closes),
        "squeeze": 1.0 if ta.keltner_squeeze(highs, lows, closes) else 0.0,
        "dow": float(date.weekday()),
    }

    return FeatureRow(
        symbol=symbol.upper(),
        date=date,
        entry=float(entry),
        atr=float(atr),
        sigma_daily=float(sigma),
        support=support.price if support else None,
        resistance=resistance.price if resistance else None,
        regime=regime.label,
        features={k: _safe(v) for k, v in feats.items()},
    )


def resolve_outcome(
    store: PointInTimeStore,
    symbol: str,
    date: dt.date,
    entry: float,
    target: float,
    stop: float,
    conservative: bool = True,
) -> tuple[int, float, str] | None:
    """Score a bracket against the session's actual bar.

    Daily OHLC does not record whether the high or the low came first. When a
    session touched both barriers the honest options are to discard it or to
    assume the worst; this assumes the worst (`conservative=True`).

    That biases every measured result downward. It is the right bias: an
    optimistic tie-break makes a strategy look profitable precisely on the
    volatile days where real execution is hardest, which is how backtests come
    to promise money they cannot deliver.
    """
    bar = store.bar_on(symbol, date)
    if bar is None:
        return None

    long_side = target > entry
    risk = abs(entry - stop)
    if risk <= 0:
        return None

    hit_target = bar.high >= target if long_side else bar.low <= target
    hit_stop = bar.low <= stop if long_side else bar.high >= stop

    if hit_target and hit_stop:
        if conservative:
            return 0, -1.0, "both barriers touched; scored as a loss (bar order unknown)"
        return 1, abs(target - entry) / risk, "both touched; optimistic tie-break"
    if hit_target:
        return 1, abs(target - entry) / risk, "target reached"
    if hit_stop:
        return 0, -1.0, "stop hit"

    pnl = (bar.close - entry) if long_side else (entry - bar.close)
    r = pnl / risk
    return (1 if r > 0 else 0), float(r), f"neither touched; marked to close ({r:+.2f}R)"


def build_dataset(
    store: PointInTimeStore,
    symbols: list[str],
    dates: list[dt.date],
    target_atr: float = 1.0,
    stop_atr: float = 0.6,
    market_symbol: str | None = "SPY",
    min_bars: int = 60,
    progress_every: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build the full point-in-time training set.

    A fixed ATR-multiple bracket is used here rather than the live structural
    bracket, deliberately: the learning target should be "does this symbol move
    up before it moves down, given today's state", not "was one particular
    stop placement lucky". The structural bracket is applied later, at replay
    time, on top of the probability this data teaches.
    """
    import logging

    log = logging.getLogger("marketswarm.backtest.features")

    X: list[np.ndarray] = []
    y: list[int] = []
    meta: list[dict] = []

    market_cache: dict[dt.date, History | None] = {}

    for i, date in enumerate(dates):
        if market_symbol:
            if date not in market_cache:
                market_cache[date] = store.history_before(market_symbol, date, 250)
            mhist = market_cache[date]
        else:
            mhist = None

        for sym in symbols:
            row = build_features(store, sym, date, mhist, min_bars=min_bars)
            if row is None:
                continue
            target = row.entry + target_atr * row.atr
            stop = row.entry - stop_atr * row.atr
            outcome = resolve_outcome(store, sym, date, row.entry, target, stop)
            if outcome is None:
                continue

            X.append(row.vector())
            y.append(outcome[0])
            meta.append({
                "symbol": sym, "date": date, "entry": row.entry, "atr": row.atr,
                "sigma": row.sigma_daily, "regime": row.regime,
                "realized_r": outcome[1], "note": outcome[2],
                "support": row.support, "resistance": row.resistance,
            })

        if progress_every and i and i % progress_every == 0:
            log.info("features: %d/%d dates, %d samples", i, len(dates), len(X))

    if not X:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0), []
    return np.vstack(X), np.array(y, dtype=float), meta
