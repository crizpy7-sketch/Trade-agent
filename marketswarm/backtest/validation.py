"""Validation that survives contact with reality.

Three specific ways a backtest lies, and the correction for each:

1. **Leakage across the train/test boundary.** Overlapping horizons and
   autocorrelated regimes mean the last training day and the first test day are
   nearly the same observation. Fix: purge and embargo (López de Prado).

2. **Selection bias.** Try forty configurations, report the best, and you have
   measured the maximum of forty noise draws. Fix: the deflated Sharpe ratio,
   which asks whether the winner beats what the best of N random strategies
   would have scored anyway.

3. **Overfitting the whole research process.** Even with clean folds, choosing
   a model *because* it did well in-sample is itself a fit. Fix: combinatorially
   symmetric cross-validation, which estimates the probability that your
   in-sample winner underperforms the median out of sample.

Skipping these is how a system arrives at 68% accuracy in a notebook and 49%
with money on it.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np


# --------------------------------------------------------------------------
# 1. purged, embargoed walk-forward
# --------------------------------------------------------------------------

@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_range: tuple[dt.date, dt.date]
    test_range: tuple[dt.date, dt.date]
    purged: int
    embargoed: int

    def describe(self) -> str:
        return (f"train {self.train_range[0]}→{self.train_range[1]} ({len(self.train_idx)}) | "
                f"test {self.test_range[0]}→{self.test_range[1]} ({len(self.test_idx)}) | "
                f"purged {self.purged}, embargoed {self.embargoed}")


def purged_walk_forward_splits(
    dates: np.ndarray,
    n_splits: int = 5,
    embargo_days: int = 5,
    horizon_days: int = 1,
    min_train: int = 250,
) -> list[Split]:
    """Expanding-window walk-forward with purging and an embargo.

    purge   : drop training samples whose outcome window overlaps the test set
    embargo : additionally drop training samples immediately after the test
              block, because volatility clusters and those days are
              contaminated by the same shocks

    Expanding rather than sliding: it mirrors how the live agent actually
    learns — it never forgets, it only accumulates.
    """
    d = np.asarray(dates)
    order = np.argsort(d)
    d_sorted = d[order]

    unique_days = np.array(sorted(set(d_sorted.tolist())))
    if len(unique_days) < n_splits + 2:
        return []

    fold_edges = np.array_split(unique_days, n_splits + 1)
    splits: list[Split] = []

    for k in range(1, n_splits + 1):
        test_days = set(fold_edges[k].tolist())
        test_start, test_end = min(test_days), max(test_days)

        train_mask = np.zeros(len(d_sorted), dtype=bool)
        purged = embargoed = 0

        for i, day in enumerate(d_sorted):
            if day in test_days:
                continue
            if day > test_end:
                # Strictly walk-forward: never train on the future.
                continue
            # Purge: this sample's outcome resolves at day + horizon. If that
            # lands inside the test block, the sample leaks.
            resolves = day + dt.timedelta(days=horizon_days)
            if resolves >= test_start:
                purged += 1
                continue
            # Embargo before the test block.
            if (test_start - day).days <= embargo_days:
                embargoed += 1
                continue
            train_mask[i] = True

        train_idx = order[train_mask]
        test_idx = order[np.array([day in test_days for day in d_sorted])]

        if len(train_idx) < min_train or len(test_idx) == 0:
            continue

        train_days = d[train_idx]
        splits.append(Split(
            train_idx=train_idx, test_idx=test_idx,
            train_range=(min(train_days), max(train_days)),
            test_range=(test_start, test_end),
            purged=purged, embargoed=embargoed,
        ))

    return splits


# --------------------------------------------------------------------------
# 2. deflated Sharpe ratio
# --------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _norm_ppf(p: float) -> float:
    """Acklam's rational approximation — accurate to ~1e-9, no scipy."""
    if not 0 < p < 1:
        return float("nan")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


@dataclass
class SharpeVerdict:
    sharpe: float
    deflated_sharpe: float
    expected_max_sharpe: float
    p_value: float
    n_trials: int
    n_obs: int
    significant: bool
    verdict: str


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int = 1,
    benchmark_sharpe: float | None = None,
    periods_per_year: int = 252,
) -> SharpeVerdict:
    """Bailey & López de Prado's deflated Sharpe ratio.

    Corrects an observed Sharpe for three things a raw Sharpe ignores: how many
    strategies were tried, how non-normal the returns are (negative skew and
    fat tails make a Sharpe flatter than it looks), and how short the sample is.

    `n_trials` must be honest. If you tried 40 parameter sets, it is 40 — not 1
    because you only wrote one down.
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 20:
        return SharpeVerdict(float("nan"), float("nan"), float("nan"), float("nan"),
                             n_trials, n, False, "too few observations to evaluate")

    mu, sd = float(r.mean()), float(r.std(ddof=1))
    if sd <= 0:
        return SharpeVerdict(0.0, 0.0, 0.0, 1.0, n_trials, n, False, "zero variance")

    sharpe = mu / sd
    skew = float(((r - mu) ** 3).mean() / sd**3)
    kurt = float(((r - mu) ** 4).mean() / sd**4)

    if benchmark_sharpe is None:
        # Expected maximum Sharpe from n_trials independent draws of pure noise.
        gamma = 0.5772156649
        e = math.e
        if n_trials > 1:
            benchmark_sharpe = (
                (1 - gamma) * _norm_ppf(1 - 1 / n_trials)
                + gamma * _norm_ppf(1 - 1 / (n_trials * e))
            ) / math.sqrt(n)
        else:
            benchmark_sharpe = 0.0

    denom = math.sqrt(max(1e-12, 1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2))
    z = (sharpe - benchmark_sharpe) * math.sqrt(n - 1) / denom
    p_dsr = _norm_cdf(z)

    ann = sharpe * math.sqrt(periods_per_year)
    if p_dsr > 0.95:
        verdict = f"significant after deflation (annualised Sharpe {ann:.2f}, DSR {p_dsr:.3f})"
    elif p_dsr > 0.80:
        verdict = f"suggestive but not significant (DSR {p_dsr:.3f}) — needs more out-of-sample data"
    else:
        verdict = (f"not distinguishable from the best of {n_trials} random strategies "
                   f"(DSR {p_dsr:.3f}) — treat the edge as unproven")

    return SharpeVerdict(
        sharpe=sharpe, deflated_sharpe=p_dsr, expected_max_sharpe=benchmark_sharpe,
        p_value=1 - p_dsr, n_trials=n_trials, n_obs=n,
        significant=p_dsr > 0.95, verdict=verdict,
    )


# --------------------------------------------------------------------------
# 3. probability of backtest overfitting
# --------------------------------------------------------------------------

def probability_of_backtest_overfitting(
    performance: np.ndarray, n_partitions: int = 8
) -> dict:
    """CSCV: probability that the in-sample best config is below-median out of sample.

    `performance` is (n_observations, n_configurations) — a return series per
    configuration you considered.

    Split the timeline into S blocks, take every way of choosing S/2 blocks as
    "in sample", pick the winner there, then see where it ranks on the
    complement. If the winner is a coin flip out of sample, PBO ≈ 0.5 and the
    whole selection exercise was noise.
    """
    perf = np.asarray(performance, float)
    if perf.ndim != 2 or perf.shape[1] < 2:
        return {"pbo": float("nan"), "note": "need at least two configurations"}

    n_obs, n_cfg = perf.shape
    S = max(2, n_partitions - (n_partitions % 2))
    if n_obs < S * 5:
        return {"pbo": float("nan"), "note": f"need at least {S * 5} observations"}

    blocks = np.array_split(np.arange(n_obs), S)
    half = S // 2
    logits: list[float] = []

    for combo in combinations(range(S), half):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in combo])

        is_perf = perf[is_idx].mean(axis=0)
        oos_perf = perf[oos_idx].mean(axis=0)

        best = int(np.argmax(is_perf))
        # Rank of the in-sample winner among all configs, out of sample.
        rank = float((oos_perf <= oos_perf[best]).sum()) / n_cfg
        rank = min(max(rank, 1 / (n_cfg + 1)), 1 - 1 / (n_cfg + 1))
        logits.append(math.log(rank / (1 - rank)))

    arr = np.array(logits)
    pbo = float((arr <= 0).mean())

    if pbo < 0.2:
        note = "selection looks robust — the in-sample winner holds up out of sample"
    elif pbo < 0.5:
        note = "moderate overfitting risk — the winner is partly luck"
    else:
        note = "severe overfitting — your configuration choice carries no information"

    return {
        "pbo": pbo,
        "n_combinations": len(logits),
        "n_configs": n_cfg,
        "median_logit": float(np.median(arr)),
        "note": note,
    }


# --------------------------------------------------------------------------
# supporting statistics
# --------------------------------------------------------------------------

def max_drawdown(equity: np.ndarray) -> dict:
    e = np.asarray(equity, float)
    if e.size < 2:
        return {"max_drawdown": 0.0, "peak_idx": 0, "trough_idx": 0}
    running_max = np.maximum.accumulate(e)
    dd = (e - running_max) / np.where(running_max == 0, 1, np.abs(running_max))
    trough = int(np.argmin(dd))
    peak = int(np.argmax(e[: trough + 1])) if trough > 0 else 0
    return {
        "max_drawdown": float(dd.min()),
        "peak_idx": peak,
        "trough_idx": trough,
        "recovery_bars": int(e.size - trough),
    }


def summarize_returns(r: np.ndarray, periods_per_year: int = 252) -> dict:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return {"n": int(r.size), "note": "insufficient data"}

    equity = np.cumsum(r)
    wins = r[r > 0]
    losses = r[r < 0]
    downside = r[r < 0]
    sd = float(r.std(ddof=1))
    dsd = float(downside.std(ddof=1)) if downside.size > 1 else float("nan")

    return {
        "n": int(r.size),
        "mean_r": float(r.mean()),
        "total_r": float(r.sum()),
        "hit_rate": float((r > 0).mean()),
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() else float("inf"),
        "sharpe": float(r.mean() / sd) if sd > 0 else 0.0,
        "sharpe_annual": float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0,
        "sortino": float(r.mean() / dsd * math.sqrt(periods_per_year)) if dsd and dsd > 0 else float("nan"),
        "max_drawdown_r": max_drawdown(equity)["max_drawdown"],
        "worst_r": float(r.min()),
        "best_r": float(r.max()),
    }
