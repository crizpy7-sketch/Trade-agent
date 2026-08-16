"""Scoring rules and recalibration — how the agent learns it was wrong.

Every published probability is scored once the outcome is known. The scores
drive two feedback loops:
  1. per-source and per-agent reliability weights (see bayes.BetaPosterior)
  2. a global recalibration map applied to future raw probabilities
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error of probabilistic forecasts. Lower is better; 0.25 is
    the score of always saying 50%."""
    p, y = np.asarray(probs, float), np.asarray(outcomes, float)
    return float(np.mean((p - y) ** 2))


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(probs, float), eps, 1 - eps)
    y = np.asarray(outcomes, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class BrierDecomposition:
    """Murphy's three-way decomposition: BS = reliability - resolution + uncertainty."""

    brier: float
    reliability: float   # calibration error — lower is better
    resolution: float    # discrimination — higher is better
    uncertainty: float   # irreducible, a property of the base rate
    skill_score: float   # 1 - BS/BS_climatology; >0 beats always-predict-base-rate

    def verdict(self) -> str:
        if self.skill_score <= 0:
            return "no skill vs. the base rate — the model is decoration"
        if self.reliability > 0.02:
            return "discriminates but is miscalibrated — recalibration should help"
        if self.resolution < 0.01:
            return "well calibrated but barely discriminates — probabilities hug the base rate"
        return "calibrated and discriminating"


def brier_decomposition(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> BrierDecomposition:
    p, y = np.asarray(probs, float), np.asarray(outcomes, float)
    n = len(p)
    if n == 0:
        return BrierDecomposition(float("nan"), 0, 0, 0, 0)

    base = float(y.mean())
    unc = base * (1 - base)
    bs = brier_score(p, y)

    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)

    rel = res = 0.0
    for b in range(n_bins):
        mask = idx == b
        nb = int(mask.sum())
        if nb == 0:
            continue
        pb, ob = float(p[mask].mean()), float(y[mask].mean())
        rel += nb * (pb - ob) ** 2
        res += nb * (ob - base) ** 2
    rel /= n
    res /= n

    skill = 1.0 - bs / unc if unc > 0 else 0.0
    return BrierDecomposition(bs, rel, res, unc, skill)


def reliability_curve(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Binned forecast-vs-observed frequency table for the calibration report."""
    p, y = np.asarray(probs, float), np.asarray(outcomes, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & (p < hi if b < n_bins - 1 else p <= hi)
        n = int(mask.sum())
        rows.append(
            {
                "bin": f"{lo:.0%}-{hi:.0%}",
                "n": n,
                "mean_forecast": float(p[mask].mean()) if n else None,
                "observed_rate": float(y[mask].mean()) if n else None,
                "gap": float(p[mask].mean() - y[mask].mean()) if n else None,
            }
        )
    return rows


class PlattCalibrator:
    """Logistic recalibration: p' = sigmoid(a * logit(p) + b).

    Two parameters, so it stays sane on the few hundred observations a daily
    agent accumulates in a year — isotonic regression would overfit that.
    a < 1 means the raw forecasts were overconfident (the usual failure).
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b, self.n_fit = a, b, 0

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def fit(self, probs: np.ndarray, outcomes: np.ndarray, iters: int = 200) -> "PlattCalibrator":
        x = self._logit(np.asarray(probs, float))
        y = np.asarray(outcomes, float)
        n = len(x)
        if n < 20:  # too little data to trust; stay at identity
            self.a, self.b, self.n_fit = 1.0, 0.0, n
            return self

        # Platt's prior-corrected targets guard against separation on small n.
        n_pos, n_neg = float(y.sum()), float(n - y.sum())
        hi = (n_pos + 1) / (n_pos + 2) if n_pos else 0.5
        lo = 1 / (n_neg + 2) if n_neg else 0.5
        t = np.where(y > 0.5, hi, lo)

        a, b = 1.0, 0.0
        for _ in range(iters):  # Newton-Raphson on the log-likelihood
            z = a * x + b
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            g1 = float(np.sum((p - t) * x))
            g0 = float(np.sum(p - t))
            w = p * (1 - p) + 1e-9
            h11 = float(np.sum(w * x * x)) + 1e-6
            h10 = float(np.sum(w * x))
            h00 = float(np.sum(w)) + 1e-6
            det = h11 * h00 - h10 * h10
            if abs(det) < 1e-12:
                break
            da = (h00 * g1 - h10 * g0) / det
            db = (h11 * g0 - h10 * g1) / det
            a, b = a - da, b - db
            if abs(da) < 1e-8 and abs(db) < 1e-8:
                break

        self.a, self.b, self.n_fit = float(a), float(b), n
        return self

    def transform(self, p: float) -> float:
        z = self.a * float(self._logit(np.array([p]))[0]) + self.b
        return float(1 / (1 + math.exp(-max(-30.0, min(30.0, z)))))

    def as_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "n_fit": self.n_fit}

    @classmethod
    def from_dict(cls, d: dict | None) -> "PlattCalibrator":
        if not d:
            return cls()
        c = cls(float(d.get("a", 1.0)), float(d.get("b", 0.0)))
        c.n_fit = int(d.get("n_fit", 0))
        return c

    @property
    def diagnosis(self) -> str:
        if self.n_fit < 20:
            return "identity (insufficient history)"
        if self.a < 0.85:
            return f"shrinking forecasts toward 50% (a={self.a:.2f}) — history says overconfident"
        if self.a > 1.15:
            return f"sharpening forecasts (a={self.a:.2f}) — history says under-confident"
        return f"near-identity (a={self.a:.2f}, b={self.b:+.2f}) — well calibrated"


def sequential_sprt(
    hits: int, n: int, p_null: float = 0.5, p_alt: float = 0.58, alpha: float = 0.05, beta: float = 0.2
) -> tuple[str, float]:
    """Wald's SPRT on whether the strategy's hit rate genuinely beats a coin.

    Returns (decision, log-likelihood ratio). Answers "do I have skill yet?"
    with far fewer observations than a fixed-n test, which matters when each
    observation costs a trading day.
    """
    if n <= 0:
        return "continue", 0.0
    misses = n - hits
    llr = hits * math.log(p_alt / p_null) + misses * math.log((1 - p_alt) / (1 - p_null))
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    if llr >= upper:
        return "skill confirmed", llr
    if llr <= lower:
        return "no skill — stand down", llr
    return "continue", llr
