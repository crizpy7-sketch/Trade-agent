"""Bayesian evidence fusion.

The swarm produces many weak, correlated signals. This module turns them into a
single calibrated probability using log-odds (logit) pooling with per-source
reliability weights, an explicit correlation haircut, and Beta-Binomial
posteriors for the historical hit rates that seed the weights.

Why logit pooling rather than averaging probabilities: independent likelihood
ratios add in log-odds space, so Bayes' rule becomes a weighted sum. Weighted
linear pooling of probabilities has no such interpretation and is systematically
under-confident at the tails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


def clip_prob(p: float, eps: float = 1e-4) -> float:
    return min(1.0 - eps, max(eps, float(p)))


def logit(p: float) -> float:
    p = clip_prob(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class Signal:
    """One piece of directional evidence.

    probability : the source's own P(event) — its raw claim.
    weight      : reliability in [0, 1]; the exponent applied to its likelihood ratio.
    name/source : provenance, carried through to the report.
    """

    name: str
    probability: float
    weight: float = 1.0
    source: str = "unknown"
    note: str = ""

    @property
    def log_likelihood_ratio(self) -> float:
        """LLR relative to the signal's own implied prior of 0.5."""
        return logit(self.probability)


@dataclass
class FusionResult:
    probability: float
    prior: float
    log_odds: float
    effective_n: float
    contributions: dict[str, float] = field(default_factory=dict)
    interval: tuple[float, float] = (0.0, 1.0)

    @property
    def edge_vs_prior(self) -> float:
        return self.probability - self.prior


def fuse(
    signals: list[Signal],
    prior: float = 0.5,
    correlation: float = 0.35,
    max_abs_llr: float = 2.2,
) -> FusionResult:
    """Combine signals into a posterior probability.

    correlation  : assumed average pairwise correlation among signals. The
                   effective sample size is shrunk to n / (1 + rho*(n-1)),
                   which is the standard variance-inflation correction for
                   exchangeably correlated observations. Treating 12 correlated
                   momentum reads as 12 independent votes is the single most
                   common way a signal stack becomes wildly overconfident.
    max_abs_llr  : per-signal cap on |LLR|, so one runaway source cannot
                   dominate the pool (a Huber-style robustness bound).
    """
    if not signals:
        return FusionResult(prior, prior, logit(prior), 0.0, {}, (prior, prior))

    weights = np.array([max(0.0, s.weight) for s in signals], dtype=float)
    llrs = np.array(
        [max(-max_abs_llr, min(max_abs_llr, s.log_likelihood_ratio)) for s in signals],
        dtype=float,
    )

    wsum = float(weights.sum())
    if wsum <= EPS:
        return FusionResult(prior, prior, logit(prior), 0.0, {}, (prior, prior))

    n_eff_raw = wsum**2 / float((weights**2).sum())  # Kish effective sample size
    rho = min(0.95, max(0.0, correlation))
    n_eff = n_eff_raw / (1.0 + rho * (n_eff_raw - 1.0))
    shrink = n_eff / n_eff_raw if n_eff_raw > 0 else 1.0

    contributions = {s.name: float(w * l * shrink) for s, w, l in zip(signals, weights, llrs)}
    log_odds = logit(prior) + float((weights * llrs).sum()) * shrink
    p = sigmoid(log_odds)

    # Posterior spread: SE of the weighted mean LLR, propagated through the link.
    mean_llr = float((weights * llrs).sum() / wsum)
    var = float((weights * (llrs - mean_llr) ** 2).sum() / wsum)
    se = math.sqrt(var / max(1.0, n_eff)) * wsum * shrink
    lo, hi = sigmoid(log_odds - 1.96 * se), sigmoid(log_odds + 1.96 * se)

    return FusionResult(
        probability=clip_prob(p),
        prior=prior,
        log_odds=log_odds,
        effective_n=n_eff,
        contributions=contributions,
        interval=(min(lo, hi), max(lo, hi)),
    )


@dataclass
class BetaPosterior:
    """Beta-Binomial posterior over a source's hit rate.

    Jeffreys prior (0.5, 0.5) by default — proper, invariant, and it does not
    pretend a source with three observations is trustworthy.
    """

    alpha: float = 0.5
    beta: float = 0.5

    def update(self, hits: int, misses: int) -> "BetaPosterior":
        return BetaPosterior(self.alpha + hits, self.beta + misses)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def n(self) -> float:
        return self.alpha + self.beta - 1.0

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1.0))

    def credible_interval(self, mass: float = 0.9) -> tuple[float, float]:
        """Normal approximation on the logit scale — adequate above ~10 obs and
        conservative below it."""
        m = clip_prob(self.mean)
        v = self.variance
        se_logit = math.sqrt(v / (m * (1 - m)) ** 2) if 0 < m < 1 else 1.0
        z = 1.6449 if mass >= 0.9 else 1.0
        c = logit(m)
        return sigmoid(c - z * se_logit), sigmoid(c + z * se_logit)

    def reliability_weight(self, floor: float = 0.15, ceiling: float = 1.6) -> float:
        """Map hit rate to a fusion weight, shrunk toward 0 by sample size.

        A source at 50% carries no weight regardless of sample. A source at 60%
        over 200 calls carries far more than one at 90% over 4 calls, which is
        exactly the behaviour the Beta posterior gives for free.
        """
        lo, _ = self.credible_interval(0.9)
        skill = max(0.0, lo - 0.5) * 2.0  # lower-bound excess over a coin flip
        confidence = self.n / (self.n + 25.0)  # sample-size shrinkage
        return float(min(ceiling, floor + skill * confidence * 2.5))


def hierarchical_shrink(
    rates: dict[str, tuple[int, int]], tau: float = 0.08
) -> dict[str, float]:
    """James-Stein / empirical-Bayes shrinkage of per-source hit rates toward
    the pooled grand mean.

    rates: {source: (hits, total)}. tau is the assumed between-source SD of true
    skill. With few observations per source, individual rates are mostly noise;
    shrinking them toward the pool beats the raw estimates in expected squared
    error (Stein's paradox), which is exactly the regime this agent lives in.
    """
    if not rates:
        return {}
    total_hits = sum(h for h, _ in rates.values())
    total_n = sum(n for _, n in rates.values())
    grand = total_hits / total_n if total_n else 0.5

    out: dict[str, float] = {}
    for src, (hits, n) in rates.items():
        if n <= 0:
            out[src] = grand
            continue
        raw = hits / n
        se2 = max(raw * (1 - raw), 0.01) / n        # sampling variance
        w = tau**2 / (tau**2 + se2)                  # weight on the source's own data
        out[src] = w * raw + (1 - w) * grand
    return out


def bayes_factor_interpretation(bf: float) -> str:
    """Jeffreys' scale, for reporting rather than decision-making."""
    a = abs(math.log10(max(bf, EPS)))
    if a < 0.5:
        return "barely worth mentioning"
    if a < 1.0:
        return "substantial"
    if a < 1.5:
        return "strong"
    if a < 2.0:
        return "very strong"
    return "decisive"


def probability_to_confidence_label(p: float, n_eff: float, dispersion: float) -> tuple[str, int]:
    """Confidence score 0-100 for the report.

    Distinct from the probability itself: a 55% call backed by nine agreeing
    independent sources is high-confidence; an 80% call from one source with a
    thin evidence base is not.
    """
    strength = abs(p - 0.5) * 2.0
    breadth = min(1.0, n_eff / 6.0)
    agreement = max(0.0, 1.0 - dispersion)
    score = int(round(100 * (0.45 * strength + 0.35 * breadth + 0.20 * agreement)))
    score = max(1, min(99, score))
    label = "low" if score < 35 else "moderate" if score < 60 else "high" if score < 80 else "very high"
    return label, score
