"""Return distributions, implied moves, and Monte Carlo.

Intraday equity returns are not Gaussian: they are fat-tailed and their
volatility clusters. Everything here defaults to Student-t rather than normal,
because sizing a stop off a normal quantile systematically under-prices the
move that actually stops you out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252
RNG = np.random.default_rng(20240101)


@dataclass
class ImpliedMove:
    underlying: float
    straddle_price: float
    days_to_expiry: float
    implied_move_abs: float
    implied_move_pct: float
    implied_vol_annual: float
    one_sd_range: tuple[float, float]

    def probability_beyond(self, level: float) -> float:
        """P(price ends beyond `level`) under the implied lognormal."""
        if self.implied_move_abs <= 0:
            return 0.0
        z = (level - self.underlying) / self.implied_move_abs
        return float(0.5 * math.erfc(z / math.sqrt(2))) if level > self.underlying else float(
            0.5 * math.erfc(-z / math.sqrt(2))
        )


def implied_move_from_straddle(
    underlying: float, straddle_price: float, days_to_expiry: float
) -> ImpliedMove:
    """The ATM straddle is the market's own price for the coming move.

    The 0.7979 factor is sqrt(2/pi): for a driftless normal, E|X| = 0.7979*sigma,
    and the ATM straddle is worth roughly E|X| — so sigma ≈ straddle/0.7979.
    This is the cleanest available read on what the option market expects, and
    it is what any intraday target must be measured against.
    """
    dte = max(days_to_expiry, 1 / 24 / 6.5)  # floor at ~10 minutes of a session
    sigma_period = straddle_price / 0.7978845608
    pct = sigma_period / underlying if underlying else 0.0
    annual = pct * math.sqrt(TRADING_DAYS / dte)
    return ImpliedMove(
        underlying=underlying,
        straddle_price=straddle_price,
        days_to_expiry=dte,
        implied_move_abs=sigma_period,
        implied_move_pct=pct,
        implied_vol_annual=annual,
        one_sd_range=(underlying - sigma_period, underlying + sigma_period),
    )


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_price_and_greeks(
    spot: float,
    strike: float,
    days_to_expiry: float,
    iv_annual: float,
    kind: str = "call",
    rate: float = 0.0,
) -> dict:
    """Black-Scholes price and greeks.

    Used to reprice an option at a proposed target or stop instead of
    extrapolating linearly from delta. Over a 1-ATR move the linear estimate is
    badly wrong — it routinely prices a short-dated OTM option to zero or to
    double its true value — and the whole point of quoting premium levels is
    that they are reachable.

    Rate defaults to zero: over a single session the carry term is far smaller
    than the error in the IV input.
    """
    T = max(days_to_expiry, 1e-4) / TRADING_DAYS
    sigma = max(iv_annual, 1e-4)
    sqrtT = math.sqrt(T)

    if spot <= 0 or strike <= 0:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc = math.exp(-rate * T)

    if kind == "call":
        price = spot * norm_cdf(d1) - strike * disc * norm_cdf(d2)
        delta = norm_cdf(d1)
        theta = (-spot * norm_pdf(d1) * sigma / (2 * sqrtT)
                 - rate * strike * disc * norm_cdf(d2)) / TRADING_DAYS
    else:
        price = strike * disc * norm_cdf(-d2) - spot * norm_cdf(-d1)
        delta = norm_cdf(d1) - 1.0
        theta = (-spot * norm_pdf(d1) * sigma / (2 * sqrtT)
                 + rate * strike * disc * norm_cdf(-d2)) / TRADING_DAYS

    return {
        "price": max(price, 0.0),
        "delta": delta,
        "gamma": norm_pdf(d1) / (spot * sigma * sqrtT),
        "theta": theta,                       # per trading day
        "vega": spot * norm_pdf(d1) * sqrtT / 100,   # per 1 vol point
    }


def expected_move(underlying: float, iv_annual: float, days: float) -> float:
    """1-sigma move over `days` from an annualised IV."""
    return underlying * iv_annual * math.sqrt(max(days, 1e-6) / TRADING_DAYS)


def student_t_paths(
    s0: float,
    sigma_daily: float,
    horizon_days: float = 1.0,
    steps: int = 78,          # 5-minute bars in a 6.5h session
    n_paths: int = 20_000,
    df: int = 4,
    drift: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate intraday paths with Student-t innovations (df=4 ≈ observed
    intraday kurtosis for index ETFs). Returns array (n_paths, steps+1)."""
    rng = np.random.default_rng(seed) if seed is not None else RNG
    dt = horizon_days / steps
    scale = sigma_daily * math.sqrt(dt) / math.sqrt(df / (df - 2))  # unit-variance t
    shocks = rng.standard_t(df, size=(n_paths, steps)) * scale
    logret = (drift * dt - 0.5 * (sigma_daily**2) * dt) + shocks
    paths = s0 * np.exp(np.cumsum(logret, axis=1))
    return np.hstack([np.full((n_paths, 1), s0), paths])


@dataclass
class TouchProbabilities:
    p_target_first: float
    p_stop_first: float
    p_neither: float
    expected_r: float
    p_target_touch: float
    p_stop_touch: float


def barrier_probabilities(
    entry: float,
    target: float,
    stop: float,
    sigma_daily: float,
    horizon_days: float = 1.0,
    drift_daily: float = 0.0,
    n_paths: int = 20_000,
    steps: int = 78,
    seed: int | None = None,
) -> TouchProbabilities:
    """Which barrier gets hit first — the only question that matters for a
    bracketed intraday trade.

    Closed-form first-passage formulas exist for GBM, but they assume normal
    increments and no path dependence in volatility. Monte Carlo with t-shocks
    costs milliseconds and is honest about the tails.
    """
    paths = student_t_paths(
        entry, sigma_daily, horizon_days, steps, n_paths, drift=drift_daily, seed=seed
    )
    long_side = target > entry

    if long_side:
        hit_t = paths >= target
        hit_s = paths <= stop
    else:
        hit_t = paths <= target
        hit_s = paths >= stop

    big = steps + 10
    first_t = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1), big)
    first_s = np.where(hit_s.any(axis=1), hit_s.argmax(axis=1), big)

    win = first_t < first_s
    loss = first_s < first_t
    neither = (first_t >= big) & (first_s >= big)

    p_win = float(win.mean())
    p_loss = float(loss.mean())
    p_none = float(neither.mean())

    r_win = abs(target - entry) / max(abs(entry - stop), 1e-9)
    # Unresolved paths are marked to their terminal price, as a real trader
    # flattening at the bell would be.
    terminal = paths[neither, -1] if p_none > 0 else np.array([])
    if terminal.size:
        pnl = (terminal - entry) if long_side else (entry - terminal)
        r_none = float(np.mean(pnl / max(abs(entry - stop), 1e-9)))
    else:
        r_none = 0.0

    exp_r = p_win * r_win - p_loss * 1.0 + p_none * r_none
    return TouchProbabilities(
        p_target_first=p_win,
        p_stop_first=p_loss,
        p_neither=p_none,
        expected_r=exp_r,
        p_target_touch=float(hit_t.any(axis=1).mean()),
        p_stop_touch=float(hit_s.any(axis=1).mean()),
    )


def gap_conditional_stats(
    history: np.ndarray, gap_pct: float, tolerance: float = 0.25
) -> dict:
    """Empirical distribution of the day's return conditional on a similar gap.

    history: (n, 2) array of [overnight_gap_pct, close_minus_open_pct].
    Answers "when this name gapped up ~0.8%, what happened next?" from data
    rather than from a story — and reports the sample size so a 4-observation
    'edge' is visibly worthless.
    """
    if history.size == 0:
        return {"n": 0}
    h = np.asarray(history, float)
    mask = np.abs(h[:, 0] - gap_pct) <= tolerance
    sel = h[mask, 1]
    if sel.size < 3:
        return {"n": int(sel.size), "note": "insufficient analogues"}
    return {
        "n": int(sel.size),
        "mean": float(sel.mean()),
        "median": float(np.median(sel)),
        "p_continuation": float((np.sign(sel) == np.sign(gap_pct)).mean()),
        "p10": float(np.percentile(sel, 10)),
        "p90": float(np.percentile(sel, 90)),
        "std": float(sel.std(ddof=1)),
    }


def bootstrap_ci(
    sample: np.ndarray, stat=np.mean, n_boot: int = 5000, mass: float = 0.9, seed: int | None = None
) -> tuple[float, float]:
    """Percentile bootstrap CI — no distributional assumption, which is the
    point when the sample is 30 skewed observations."""
    x = np.asarray(sample, float)
    if x.size < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed) if seed is not None else RNG
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    stats = np.array([stat(x[i]) for i in idx])
    lo = (1 - mass) / 2 * 100
    return float(np.percentile(stats, lo)), float(np.percentile(stats, 100 - lo))


def realized_volatility(closes: np.ndarray, window: int = 20, annualize: bool = False) -> float:
    """Close-to-close realized vol (daily unless annualized)."""
    c = np.asarray(closes, float)
    if c.size < 3:
        return float("nan")
    r = np.diff(np.log(c))[-window:]
    v = float(np.std(r, ddof=1))
    return v * math.sqrt(TRADING_DAYS) if annualize else v


def parkinson_volatility(highs: np.ndarray, lows: np.ndarray, window: int = 20) -> float:
    """Range-based estimator — roughly 5x more efficient than close-to-close,
    which matters when only 20 bars are available."""
    h, l = np.asarray(highs, float)[-window:], np.asarray(lows, float)[-window:]
    if h.size < 2:
        return float("nan")
    hl = np.log(np.maximum(h, 1e-9) / np.maximum(l, 1e-9)) ** 2
    return float(math.sqrt(hl.mean() / (4 * math.log(2))))


def garman_klass_volatility(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int = 20
) -> float:
    o, h, l, c = (np.asarray(x, float)[-window:] for x in (opens, highs, lows, closes))
    if o.size < 2:
        return float("nan")
    hl = 0.5 * np.log(np.maximum(h, 1e-9) / np.maximum(l, 1e-9)) ** 2
    co = (2 * math.log(2) - 1) * np.log(np.maximum(c, 1e-9) / np.maximum(o, 1e-9)) ** 2
    return float(math.sqrt(np.mean(hl - co)))


def vol_of_vol_ratio(iv_annual: float, realized_annual: float) -> dict:
    """IV vs realized — the variance risk premium in miniature.

    Ratio well above 1 means options are pricing more movement than the stock
    has been delivering: premium selling is favoured, and long premium needs the
    move to arrive fast. Below 1 means long premium is cheap relative to recent
    behaviour.
    """
    if realized_annual <= 0 or math.isnan(realized_annual):
        return {"ratio": float("nan"), "read": "insufficient data"}
    ratio = iv_annual / realized_annual
    if ratio > 1.35:
        read = "options rich vs realized — long premium needs an immediate move"
    elif ratio < 0.85:
        read = "options cheap vs realized — long premium favoured"
    else:
        read = "options fairly priced vs realized"
    return {"ratio": float(ratio), "read": read}
