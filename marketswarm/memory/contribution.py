"""Contextual agent reliability, measured by contribution rather than agreement.

The 1.x learning loop asked "was this agent pointing the right way?" That
rewards an agent that says +0.5 every day in a bull market — it is right most of
the time and adds nothing. The question worth asking is:

    did the forecast get better because this agent was in it?

which is answered by ablation: recompute the fused probability with the agent's
contribution removed and compare log-loss. An agent whose removal *improves* the
forecast has negative contribution and should lose weight no matter how often it
happens to be directionally correct.

Reliability is also context-dependent. One scalar per agent cannot express that
the options agent is strong into earnings and weak in a macro shock, so scores
are stored per `(agent, context)`.

Small samples are the obvious failure mode here — three lucky calls must not
produce a 2× weight. Three safeguards, all enforced:
  * a minimum observation count before any deviation from baseline
  * shrinkage toward 1.0 proportional to sample size
  * hard bounds on the final weight
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("marketswarm.memory.contribution")

MIN_OBSERVATIONS = 12
SHRINKAGE_N = 30.0          # sample size at which half the raw signal is trusted
WEIGHT_FLOOR = 0.15
WEIGHT_CEILING = 1.75
MAX_WEIGHT_STEP = 0.25      # no single update may move a weight further than this


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, x))))


def _log_loss(p: float, y: float, eps: float = 1e-6) -> float:
    p = min(1 - eps, max(eps, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


@dataclass
class ContextKey:
    """The slice a score applies to. Kept coarse on purpose — fine slicing
    produces many cells with too few observations to mean anything."""

    regime: str = "any"
    event_type: str = "any"
    horizon: str = "intraday"

    def key(self) -> str:
        return f"regime={self.regime}|event={self.event_type}|horizon={self.horizon}"

    @staticmethod
    def parse(key: str) -> "ContextKey":
        parts = dict(p.split("=", 1) for p in key.split("|") if "=" in p)
        return ContextKey(parts.get("regime", "any"), parts.get("event", "any"),
                          parts.get("horizon", "intraday"))

    def generalisations(self) -> list[str]:
        """Progressively broader keys, for backing off when a cell is thin."""
        return [
            self.key(),
            ContextKey(self.regime, "any", self.horizon).key(),
            ContextKey("any", self.event_type, self.horizon).key(),
            ContextKey("any", "any", self.horizon).key(),
            ContextKey("any", "any", "any").key(),
        ]


@dataclass
class ContributionResult:
    agent: str
    context: str
    n: int
    mean_contribution: float        # positive = the agent improved the forecast
    brier_with: float
    brier_without: float
    hit_rate: float
    raw_weight: float
    shrunk_weight: float
    final_weight: float
    ci_low: float | None = None
    ci_high: float | None = None
    verdict: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def helped(self) -> bool:
        return self.mean_contribution > 0


def ablation_contribution(
    contributions: dict[str, float],
    outcome: int,
    prior: float = 0.5,
) -> dict[str, float]:
    """Per-agent log-loss improvement for one resolved prediction.

    `contributions` is the agent → log-odds map the fusion layer already
    records. Removing an agent means subtracting its log-odds and rescoring;
    the difference is what it contributed on this observation.

    Positive means the full forecast beat the forecast without that agent.
    """
    total_lo = _logit(prior) + sum(contributions.values())
    p_full = _sigmoid(total_lo)
    loss_full = _log_loss(p_full, float(outcome))

    out: dict[str, float] = {}
    for agent, lo in contributions.items():
        p_without = _sigmoid(total_lo - lo)
        out[agent] = _log_loss(p_without, float(outcome)) - loss_full
    return out


class ContributionTracker:
    """Persists and reads `agent_context_scores` (migration 003)."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---------- update ----------

    def update_from_resolved(self, rows: list[sqlite3.Row],
                             default_context: ContextKey | None = None) -> list[ContributionResult]:
        """Recompute contextual weights from resolved predictions.

        Each row needs `contributing_agents` (JSON log-odds), `outcome`, and
        ideally `features` carrying the regime and setup that were in force.
        """
        buckets: dict[tuple[str, str], list[tuple[float, int, float]]] = {}

        for r in rows:
            try:
                contribs = json.loads(r["contributing_agents"] or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if not contribs:
                continue
            outcome = int(r["outcome"] or 0)

            try:
                feats = json.loads(r["features"] or "{}")
            except (json.JSONDecodeError, TypeError):
                feats = {}
            ctx = ContextKey(
                regime=str(feats.get("regime", (default_context.regime
                                                if default_context else "any"))),
                event_type=str(feats.get("event_type", feats.get("setup", "any"))),
                horizon=str(r["horizon"] if "horizon" in r.keys() else "intraday"),
            )

            deltas = ablation_contribution(contribs, outcome,
                                           prior=float(r["raw_probability"] or 0.5)
                                           if "raw_probability" in r.keys() else 0.5)
            p_full = float(r["probability"] or 0.5)
            for agent, delta in deltas.items():
                # Store under the specific context and the global one, so a
                # thin slice can still back off to something with a sample.
                for key in {ctx.key(), ContextKey().key()}:
                    buckets.setdefault((agent, key), []).append(
                        (delta, outcome, p_full))

        results: list[ContributionResult] = []
        for (agent, key), obs in buckets.items():
            results.append(self._score(agent, key, obs))
        for res in results:
            self._persist(res)
        return results

    def _score(self, agent: str, context: str,
               obs: list[tuple[float, int, float]]) -> ContributionResult:
        deltas = np.array([o[0] for o in obs], float)
        outcomes = np.array([o[1] for o in obs], float)
        probs = np.array([o[2] for o in obs], float)

        n = len(obs)
        mean_delta = float(deltas.mean())
        hit_rate = float(outcomes.mean())
        brier_with = float(np.mean((probs - outcomes) ** 2))
        brier_without = brier_with + mean_delta * 0.1   # indicative, not a claim

        # Raw weight: 1.0 is neutral. A mean log-loss improvement of 0.05 nat is
        # a strong effect on this kind of data, so it is scaled accordingly.
        raw = 1.0 + mean_delta * 6.0

        # Shrink toward neutral by sample size (empirical Bayes in spirit).
        trust = n / (n + SHRINKAGE_N)
        shrunk = 1.0 + (raw - 1.0) * trust

        notes: list[str] = []
        if n < MIN_OBSERVATIONS:
            shrunk = 1.0
            notes.append(f"only {n} observations — held at neutral until "
                         f"{MIN_OBSERVATIONS}")

        final = float(max(WEIGHT_FLOOR, min(WEIGHT_CEILING, shrunk)))

        # Bound the step so one bad month cannot halve an agent.
        prev = self.get_weight(agent, context, default=None)
        if prev is not None:
            final = float(max(prev - MAX_WEIGHT_STEP,
                              min(prev + MAX_WEIGHT_STEP, final)))
            notes.append(f"step-limited from {prev:.2f}")

        se = float(deltas.std(ddof=1) / math.sqrt(n)) if n > 1 else None
        ci = (mean_delta - 1.96 * se, mean_delta + 1.96 * se) if se else (None, None)

        if n < MIN_OBSERVATIONS:
            verdict = "insufficient data"
        elif ci[0] is not None and ci[0] > 0:
            verdict = "improves the forecast"
        elif ci[1] is not None and ci[1] < 0:
            verdict = "degrades the forecast — consider dropping"
        else:
            verdict = "no measurable contribution"

        return ContributionResult(
            agent=agent, context=context, n=n, mean_contribution=mean_delta,
            brier_with=brier_with, brier_without=brier_without, hit_rate=hit_rate,
            raw_weight=raw, shrunk_weight=shrunk, final_weight=final,
            ci_low=ci[0], ci_high=ci[1], verdict=verdict, notes=notes,
        )

    def _persist(self, r: ContributionResult) -> None:
        import datetime as _dt
        self.conn.execute(
            """INSERT INTO agent_context_scores
               (agent, context_key, n, hits, contribution_sum, brier_with, brier_without,
                weight, weight_lo, weight_hi, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(agent, context_key) DO UPDATE SET
                 n=excluded.n, hits=excluded.hits,
                 contribution_sum=excluded.contribution_sum,
                 brier_with=excluded.brier_with, brier_without=excluded.brier_without,
                 weight=excluded.weight, weight_lo=excluded.weight_lo,
                 weight_hi=excluded.weight_hi, updated_at=excluded.updated_at""",
            (r.agent, r.context, r.n, int(r.hit_rate * r.n),
             r.mean_contribution * r.n, r.brier_with, r.brier_without,
             r.final_weight, r.ci_low, r.ci_high,
             _dt.datetime.now(_dt.timezone.utc).isoformat()),
        )
        self.conn.commit()

    # ---------- read ----------

    def get_weight(self, agent: str, context: str | ContextKey = "",
                   default: float | None = 1.0) -> float | None:
        """Weight for an agent in a context, backing off to broader slices."""
        keys = (context.generalisations() if isinstance(context, ContextKey)
                else [context] if context else [ContextKey().key()])
        for k in keys:
            row = self.conn.execute(
                "SELECT weight, n FROM agent_context_scores WHERE agent=? AND context_key=?",
                (agent, k)).fetchone()
            if row and int(row[1] or 0) >= MIN_OBSERVATIONS:
                return float(row[0])
        return default

    def weights_for_context(self, ctx: ContextKey) -> dict[str, float]:
        agents = [r[0] for r in self.conn.execute(
            "SELECT DISTINCT agent FROM agent_context_scores")]
        return {a: self.get_weight(a, ctx, default=1.0) for a in agents}

    def matrix(self) -> dict[str, dict[str, float]]:
        """agent → context → weight, for the operator report."""
        out: dict[str, dict[str, float]] = {}
        for r in self.conn.execute(
            "SELECT agent, context_key, weight, n FROM agent_context_scores "
            "ORDER BY agent, context_key"
        ):
            out.setdefault(r[0], {})[r[1]] = round(float(r[2]), 3)
        return out

    def underperformers(self, threshold: float = 0.5) -> list[dict]:
        """Agents measurably making things worse — candidates for removal."""
        return [
            {"agent": r[0], "context": r[1], "weight": round(float(r[2]), 3),
             "n": int(r[3] or 0)}
            for r in self.conn.execute(
                "SELECT agent, context_key, weight, n FROM agent_context_scores "
                "WHERE weight < ? AND n >= ? ORDER BY weight",
                (threshold, MIN_OBSERVATIONS))
        ]
