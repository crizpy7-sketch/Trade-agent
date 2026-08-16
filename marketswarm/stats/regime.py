"""Volatility and market regime classification.

The same setup has a different edge in a low-vol grind than in a VIX-28 tape.
Everything downstream (targets, stops, whether to be long premium at all) is
conditioned on the regime detected here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Regime:
    label: str                 # quiet_trend | choppy | volatile_trend | stress
    vol_percentile: float
    trend_score: float
    breadth_score: float
    persistence: float         # autocorrelation of daily returns
    playbook: str
    notes: list[str] = field(default_factory=list)


def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """H > 0.5 trending, H < 0.5 mean-reverting, ≈0.5 random walk.

    Estimated by the variance of lagged differences. Deliberately noisy on short
    samples, so it is used as a tilt rather than a switch.
    """
    x = np.asarray(series, float)
    if x.size < max_lag * 2:
        return 0.5

    lags, tau = [], []
    for lag in range(2, max_lag):
        sd = float(np.std(x[lag:] - x[:-lag]))
        if sd <= 0:  # degenerate (constant or perfectly linear) — no information
            continue
        lags.append(lag)
        tau.append(math.sqrt(sd))
    if len(lags) < 3:
        return 0.5
    try:
        slope = float(np.polyfit(np.log(lags), np.log(tau), 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.5
    return max(0.0, min(1.0, slope * 2.0))


def garch_forecast(returns: np.ndarray, omega: float | None = None,
                   alpha: float = 0.10, beta: float = 0.85) -> dict:
    """One-step GARCH(1,1) variance forecast with standard equity parameters.

    Fitting alpha/beta on a few hundred daily returns produces estimates whose
    standard errors swamp the difference from the well-known (0.10, 0.85)
    equity-index values, so those are used and omega is pinned to the sample
    variance for the correct long-run level. alpha+beta=0.95 gives realistic
    vol persistence and mean reversion.
    """
    r = np.asarray(returns, float)
    if r.size < 20:
        return {"sigma_daily": float(np.std(r, ddof=1)) if r.size > 2 else float("nan"),
                "long_run": float("nan"), "note": "insufficient history for GARCH"}
    lr_var = float(np.var(r, ddof=1))
    if omega is None:
        omega = lr_var * (1 - alpha - beta)

    var = lr_var
    for x in r:
        var = omega + alpha * x * x + beta * var
    sigma = math.sqrt(max(var, 1e-12))
    return {
        "sigma_daily": sigma,
        "sigma_annual": sigma * math.sqrt(252),
        "long_run": math.sqrt(lr_var),
        "vol_ratio": sigma / math.sqrt(lr_var) if lr_var > 0 else float("nan"),
        "mean_reverting": sigma > math.sqrt(lr_var),
    }


def classify_regime(
    closes: np.ndarray,
    vix: float | None = None,
    breadth_pct_above_ma: float | None = None,
) -> Regime:
    c = np.asarray(closes, float)
    notes: list[str] = []
    if c.size < 30:
        return Regime("unknown", float("nan"), 0.0, 0.0, 0.0,
                      "insufficient history — trade smaller", ["<30 sessions of data"])

    rets = np.diff(np.log(c))
    rv = float(np.std(rets[-20:], ddof=1) * math.sqrt(252))

    window = rets[-250:] if rets.size >= 250 else rets
    rolling = np.array([np.std(window[max(0, i - 20):i], ddof=1) for i in range(21, window.size)])
    vol_pct = float((rolling < np.std(rets[-20:], ddof=1)).mean()) if rolling.size else 0.5

    trend = float(np.sign(c[-1] - c[-21]) * min(1.0, abs(c[-1] / c[-21] - 1) / 0.05))
    persistence = float(np.corrcoef(rets[:-1], rets[1:])[0, 1]) if rets.size > 5 else 0.0
    h = hurst_exponent(c)
    breadth = breadth_pct_above_ma if breadth_pct_above_ma is not None else 0.5

    if vix is not None:
        notes.append(f"VIX {vix:.1f}")
        stressed = vix >= 25
        elevated = vix >= 18
    else:
        notes.append(f"realized vol {rv:.1%} annualised (VIX unavailable)")
        stressed = rv >= 0.30
        elevated = rv >= 0.20

    if stressed:
        label = "stress"
        playbook = ("Wide ranges, gap risk, unreliable levels. Half size, wider stops, "
                    "take profits fast; long premium works but IV crush is severe.")
    elif elevated and abs(trend) > 0.4:
        label = "volatile_trend"
        playbook = ("Directional continuation with real range. Trend-follow pullbacks to VWAP/9EMA; "
                    "targets can be extended, but honour stops — reversals are violent.")
    elif not elevated and abs(trend) > 0.35 and h > 0.52:
        label = "quiet_trend"
        playbook = ("Grinding directional tape. Buy pullbacks / sell rips in trend direction; "
                    "small targets, high hit rate, avoid fading strength.")
    else:
        label = "choppy"
        playbook = ("Mean-reverting range. Fade extremes into levels, avoid breakout entries, "
                    "expect failed follow-through; scalp-sized targets only.")

    notes.append(f"Hurst {h:.2f} ({'trending' if h > 0.52 else 'mean-reverting' if h < 0.48 else 'random walk'})")
    notes.append(f"20d realized vol {rv:.1%}")
    if abs(persistence) > 0.15:
        notes.append(f"daily return autocorrelation {persistence:+.2f}")

    return Regime(label, vol_pct, trend, breadth, persistence, playbook, notes)


def vix_term_structure_signal(vix: float | None, vix3m: float | None) -> dict:
    """Contango vs backwardation.

    VIX above VIX3M (backwardation) means the market is pricing near-term risk
    above longer-dated risk — historically a defensive tell and a poor
    environment for complacent long-delta exposure.
    """
    if vix is None or vix3m is None or vix3m <= 0:
        return {"state": "unknown", "ratio": None, "read": "term structure unavailable"}
    ratio = vix / vix3m
    if ratio > 1.0:
        return {"state": "backwardation", "ratio": round(ratio, 3),
                "read": "near-term risk bid above 3-month — defensive, favours put structures and fast profit-taking"}
    if ratio < 0.90:
        return {"state": "steep contango", "ratio": round(ratio, 3),
                "read": "calm term structure — favours trend continuation and premium selling"}
    return {"state": "contango", "ratio": round(ratio, 3), "read": "normal term structure"}
