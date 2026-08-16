"""Technical structure: levels, trend, momentum, volatility bands.

Pure numpy/pandas so it runs anywhere. Every function takes plain arrays; the
agent layer is responsible for feeding it clean OHLCV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def ema(values: np.ndarray, span: int) -> np.ndarray:
    v = np.asarray(values, float)
    if v.size == 0:
        return v
    alpha = 2 / (span + 1)
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, v.size):
        out[i] = alpha * v[i] + (1 - alpha) * out[i - 1]
    return out


def sma(values: np.ndarray, window: int) -> np.ndarray:
    v = np.asarray(values, float)
    if v.size < window:
        return np.full_like(v, np.nan)
    c = np.cumsum(np.insert(v, 0, 0.0))
    out = np.full_like(v, np.nan)
    out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def rsi(closes: np.ndarray, period: int = 14) -> float:
    c = np.asarray(closes, float)
    if c.size <= period:
        return float("nan")
    d = np.diff(c)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    ag, al = gain[:period].mean(), loss[:period].mean()
    for i in range(period, d.size):  # Wilder smoothing
        ag = (ag * (period - 1) + gain[i]) / period
        al = (al * (period - 1) + loss[i]) / period
    if al == 0:
        return 100.0
    return float(100 - 100 / (1 + ag / al))


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    h, l, c = (np.asarray(x, float) for x in (highs, lows, closes))
    if c.size < 2:
        return float("nan")
    prev = c[:-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev), np.abs(l[1:] - prev)))
    if tr.size < period:
        return float(tr.mean())
    a = tr[:period].mean()
    for x in tr[period:]:
        a = (a * (period - 1) + x) / period
    return float(a)


def vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float:
    h, l, c, v = (np.asarray(x, float) for x in (highs, lows, closes, volumes))
    tp = (h + l + c) / 3
    tot = v.sum()
    return float((tp * v).sum() / tot) if tot > 0 else float(c[-1] if c.size else np.nan)


def anchored_vwap(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray, anchor_idx: int
) -> float:
    s = slice(max(0, anchor_idx), None)
    return vwap(np.asarray(highs)[s], np.asarray(lows)[s], np.asarray(closes)[s], np.asarray(volumes)[s])


def bollinger(closes: np.ndarray, window: int = 20, k: float = 2.0) -> dict:
    c = np.asarray(closes, float)
    if c.size < window:
        return {"mid": float("nan"), "upper": float("nan"), "lower": float("nan"), "bandwidth": float("nan"), "percent_b": float("nan")}
    seg = c[-window:]
    mid, sd = float(seg.mean()), float(seg.std(ddof=1))
    up, lo = mid + k * sd, mid - k * sd
    return {
        "mid": mid,
        "upper": up,
        "lower": lo,
        "bandwidth": (up - lo) / mid if mid else float("nan"),
        "percent_b": (float(c[-1]) - lo) / (up - lo) if up > lo else float("nan"),
    }


def keltner_squeeze(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int = 20
) -> bool:
    """Bollinger bands inside Keltner channels — compressed volatility that
    historically precedes expansion. A setup filter, not a direction call."""
    bb = bollinger(closes, window)
    if math.isnan(bb["upper"]):
        return False
    mid = float(np.asarray(closes, float)[-window:].mean())
    a = atr(highs, lows, closes, window)
    if math.isnan(a):
        return False
    return bb["upper"] < mid + 1.5 * a and bb["lower"] > mid - 1.5 * a


def pivot_levels(prev_high: float, prev_low: float, prev_close: float) -> dict:
    """Classic floor-trader pivots off the prior session."""
    p = (prev_high + prev_low + prev_close) / 3
    r = prev_high - prev_low
    return {
        "P": p,
        "R1": 2 * p - prev_low,
        "S1": 2 * p - prev_high,
        "R2": p + r,
        "S2": p - r,
        "R3": prev_high + 2 * (p - prev_low),
        "S3": prev_low - 2 * (prev_high - p),
    }


@dataclass
class Level:
    price: float
    kind: str            # "support" | "resistance"
    strength: float      # 0-1, from touch count and recency
    touches: int
    sources: list[str] = field(default_factory=list)


def cluster_levels(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    current: float,
    tolerance_pct: float = 0.0035,
    max_levels: int = 6,
) -> list[Level]:
    """Find price levels the market has repeatedly respected.

    Swing pivots are clustered within a tolerance band; clusters with more
    touches and more recent touches score higher. Recency is weighted because a
    level from 200 sessions ago carries far less information than last week's.
    """
    h, l = np.asarray(highs, float), np.asarray(lows, float)
    n = h.size
    if n < 10:
        return []

    pts: list[tuple[float, str, float]] = []  # (price, kind, recency weight)
    for i in range(2, n - 2):
        w = (i + 1) / n
        if h[i] == max(h[i - 2 : i + 3]):
            pts.append((float(h[i]), "resistance", w))
        if l[i] == min(l[i - 2 : i + 3]):
            pts.append((float(l[i]), "support", w))
    if not pts:
        return []

    pts.sort(key=lambda x: x[0])
    clusters: list[list[tuple[float, str, float]]] = [[pts[0]]]
    for p in pts[1:]:
        if abs(p[0] - clusters[-1][-1][0]) / max(clusters[-1][-1][0], 1e-9) <= tolerance_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    levels: list[Level] = []
    for cl in clusters:
        price = float(np.average([c[0] for c in cl], weights=[c[2] for c in cl]))
        touches = len(cl)
        recency = float(np.mean([c[2] for c in cl]))
        strength = min(1.0, (touches / 5) * 0.6 + recency * 0.4)
        kind = "resistance" if price > current else "support"
        levels.append(Level(price, kind, strength, touches, ["swing pivots"]))

    levels.sort(key=lambda lv: (-lv.strength, abs(lv.price - current)))
    return levels[:max_levels]


def nearest_levels(levels: list[Level], price: float) -> tuple[Level | None, Level | None]:
    below = [lv for lv in levels if lv.price < price]
    above = [lv for lv in levels if lv.price > price]
    return (
        max(below, key=lambda lv: lv.price) if below else None,
        min(above, key=lambda lv: lv.price) if above else None,
    )


@dataclass
class TrendState:
    direction: str        # "up" | "down" | "sideways"
    strength: float       # 0-1
    ema9: float
    ema21: float
    ema50: float
    above_all: bool
    slope_pct: float
    notes: list[str] = field(default_factory=list)


def trend_state(closes: np.ndarray) -> TrendState:
    c = np.asarray(closes, float)
    if c.size < 21:
        return TrendState("sideways", 0.0, float("nan"), float("nan"), float("nan"), False, 0.0,
                          ["insufficient history"])
    e9, e21 = ema(c, 9)[-1], ema(c, 21)[-1]
    e50 = ema(c, 50)[-1] if c.size >= 50 else float("nan")
    last = float(c[-1])
    lookback = min(10, c.size - 1)
    slope = (last / c[-1 - lookback] - 1) * 100

    notes = []
    score = 0.0
    if e9 > e21:
        score += 0.35
        notes.append("9EMA above 21EMA")
    else:
        score -= 0.35
        notes.append("9EMA below 21EMA")
    if not math.isnan(e50):
        if last > e50:
            score += 0.25
            notes.append("price above 50EMA")
        else:
            score -= 0.25
            notes.append("price below 50EMA")
    score += max(-0.4, min(0.4, slope / 5 * 0.4))

    direction = "up" if score > 0.2 else "down" if score < -0.2 else "sideways"
    return TrendState(
        direction=direction,
        strength=min(1.0, abs(score)),
        ema9=float(e9),
        ema21=float(e21),
        ema50=float(e50),
        above_all=bool(last > e9 > e21) if not math.isnan(e9) else False,
        slope_pct=float(slope),
        notes=notes,
    )


def opening_range_levels(gap_pct: float, prev_close: float, premarket_high: float, premarket_low: float) -> dict:
    """Pre-market extremes are the first intraday levels that matter, because
    they are where overnight positioning actually transacted."""
    return {
        "premarket_high": premarket_high,
        "premarket_low": premarket_low,
        "prev_close": prev_close,
        "gap_pct": gap_pct,
        "gap_fill_target": prev_close,
        "gap_direction": "up" if gap_pct > 0 else "down" if gap_pct < 0 else "flat",
    }


def zscore(value: float, series: np.ndarray) -> float:
    s = np.asarray(series, float)
    if s.size < 3:
        return float("nan")
    sd = float(s.std(ddof=1))
    return float((value - s.mean()) / sd) if sd > 0 else 0.0


def relative_volume(current_volume: float, historical_volumes: np.ndarray) -> float:
    """RVOL — participation relative to normal. Below 1 means the move has no
    sponsorship, which is the usual reason a clean-looking setup fails."""
    hv = np.asarray(historical_volumes, float)
    if hv.size == 0:
        return float("nan")
    med = float(np.median(hv))
    return float(current_volume / med) if med > 0 else float("nan")
