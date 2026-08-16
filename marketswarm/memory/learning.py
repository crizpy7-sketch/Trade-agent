"""The learning loop.

Runs after the close (`marketswarm score`). It:
  1. resolves yesterday's predictions against actual intraday price action
  2. scores them (Brier, log loss, Murphy decomposition, SPRT)
  3. updates per-agent weights and per-source reliability via Beta posteriors
  4. refits the global Platt recalibrator
  5. mines conditional lessons ("gap-fill longs in stress regimes lose")

Everything it learns is persisted and read back by the next morning's run, so
the agent's behaviour changes without anyone editing code.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass

import numpy as np

from ..stats.bayes import BetaPosterior, hierarchical_shrink
from ..stats.calibration import (
    PlattCalibrator,
    brier_decomposition,
    log_loss,
    reliability_curve,
    sequential_sprt,
)
from .store import MemoryStore

log = logging.getLogger("marketswarm.learning")

CALIBRATOR_KEY = "global_platt"
MIN_FIT_SAMPLES = 20


@dataclass
class ScoringResult:
    resolved: int
    unresolvable: int
    brier: float
    log_loss: float
    skill_score: float
    hit_rate: float
    expectancy_r: float
    calibrator: dict
    verdict: str


class LearningEngine:
    def __init__(self, store: MemoryStore):
        self.store = store

    # ---------- 1. resolution ----------

    def resolve_prediction(self, row, bars: dict) -> tuple[int, float, str] | None:
        """Decide whether target or stop came first, from intraday bars.

        bars: {"highs": [...], "lows": [...], "closes": [...]} for the session
        of the prediction, in chronological order. Sequence matters: a day that
        touched both barriers is a win only if the target came first, which is
        the same rule the Monte Carlo used to price it.
        """
        entry, target, stop = row["entry"], row["target"], row["stop"]
        if entry is None or target is None or stop is None:
            return None
        highs, lows = bars.get("highs") or [], bars.get("lows") or []
        closes = bars.get("closes") or []
        if not highs or not lows:
            return None

        long_side = target > entry
        t_idx = s_idx = None
        for i, (h, l) in enumerate(zip(highs, lows)):
            if t_idx is None and ((long_side and h >= target) or (not long_side and l <= target)):
                t_idx = i
            if s_idx is None and ((long_side and l <= stop) or (not long_side and h >= stop)):
                s_idx = i
            if t_idx is not None and s_idx is not None:
                break

        r_unit = abs(entry - stop)
        if t_idx is not None and (s_idx is None or t_idx < s_idx):
            return 1, abs(target - entry) / r_unit, "target reached first"
        if s_idx is not None:
            return 0, -1.0, "stop hit first"

        last = float(closes[-1]) if closes else entry
        pnl = (last - entry) if long_side else (entry - last)
        r = pnl / r_unit
        return (1 if r > 0 else 0), r, f"neither barrier touched; marked to close ({r:+.2f}R)"

    def resolve_batch(self, resolutions: dict[int, tuple[int, float, str]]) -> int:
        for pid, (outcome, r, note) in resolutions.items():
            self.store.resolve(pid, outcome, r, note)
        return len(resolutions)

    # ---------- 2-4. scoring, weights, recalibration ----------

    def score_and_learn(self, lookback_days: int = 180) -> ScoringResult:
        cutoff = (dt.date.today() - dt.timedelta(days=lookback_days)).isoformat()
        rows = [r for r in self.store.scored_history() if r["run_date"] >= cutoff]

        if len(rows) < 5:
            return ScoringResult(
                resolved=len(rows), unresolvable=0, brier=float("nan"), log_loss=float("nan"),
                skill_score=0.0, hit_rate=float("nan"), expectancy_r=float("nan"),
                calibrator=PlattCalibrator().as_dict(),
                verdict=f"only {len(rows)} resolved predictions — keep running, "
                        f"{MIN_FIT_SAMPLES} needed before recalibration engages",
            )

        probs = np.array([float(r["probability"]) for r in rows])
        outcomes = np.array([float(r["outcome"] or 0) for r in rows])
        rs = np.array([float(r["realized_r"] or 0) for r in rows])

        decomp = brier_decomposition(probs, outcomes)
        ll = log_loss(probs, outcomes)

        raw = np.array([float(r["raw_probability"] or r["probability"]) for r in rows])
        calibrator = PlattCalibrator().fit(raw, outcomes)
        self.store.save_state(CALIBRATOR_KEY, calibrator.as_dict())

        self._update_agent_weights(rows)
        self._update_source_reliability(rows)
        self._mine_lessons(rows)

        hits = int(outcomes.sum())
        decision, llr = sequential_sprt(hits, len(rows))

        verdict = (
            f"{len(rows)} scored | Brier {decomp.brier:.3f} (skill {decomp.skill_score:+.3f}) | "
            f"{decomp.verdict} | SPRT: {decision} (LLR {llr:+.2f}) | recalibration: {calibrator.diagnosis}"
        )

        return ScoringResult(
            resolved=len(rows),
            unresolvable=0,
            brier=decomp.brier,
            log_loss=ll,
            skill_score=decomp.skill_score,
            hit_rate=float(outcomes.mean()),
            expectancy_r=float(rs.mean()),
            calibrator=calibrator.as_dict(),
            verdict=verdict,
        )

    def _update_agent_weights(self, rows) -> None:
        """Credit-assign outcomes to the agents that pushed the call.

        An agent is credited when its LLR contribution pointed the same way as
        the realized outcome, weighted by how hard it pushed. Agents that
        habitually push the wrong way lose weight in tomorrow's fusion.
        """
        tallies: dict[str, list[float]] = {}
        for r in rows:
            try:
                contribs = json.loads(r["contributing_agents"] or "{}")
            except json.JSONDecodeError:
                continue
            outcome = int(r["outcome"] or 0)
            for agent, llr in contribs.items():
                if abs(float(llr)) < 0.05:      # no meaningful opinion, no credit
                    continue
                pushed_up = float(llr) > 0
                correct = pushed_up == bool(outcome)
                tallies.setdefault(agent, []).append(1.0 if correct else 0.0)

        raw_rates = {a: (int(sum(v)), len(v)) for a, v in tallies.items()}
        shrunk = hierarchical_shrink(raw_rates)

        for agent, results in tallies.items():
            hits = int(sum(results))
            misses = len(results) - hits
            post = BetaPosterior().update(hits, misses)
            weight = post.reliability_weight()
            # Blend the source's own posterior with the empirical-Bayes shrunk
            # rate so a lucky streak on 6 observations cannot triple a weight.
            shrunk_rate = shrunk.get(agent, 0.5)
            weight *= 0.5 + max(0.0, shrunk_rate - 0.5) * 2.0 + 0.5
            weight = max(0.1, min(1.6, weight))
            self.store.upsert_agent_score(agent, hits, misses, 0.0, len(results), round(weight, 4))
            log.info("agent %s: %d/%d correct → weight %.2f", agent, hits, len(results), weight)

    def _update_source_reliability(self, rows) -> None:
        pred_ids = [r["id"] for r in rows]
        if not pred_ids:
            return
        placeholders = ",".join("?" * len(pred_ids))
        ev = list(
            self.store.conn.execute(
                f"SELECT prediction_id, source_class FROM evidence WHERE prediction_id IN ({placeholders})",
                pred_ids,
            )
        )
        if not ev:
            return
        outcome_by_pred = {r["id"]: int(r["outcome"] or 0) for r in rows}

        tally: dict[str, list[int]] = {}
        for e in ev:
            sc = e["source_class"] or "unknown"
            o = outcome_by_pred.get(e["prediction_id"])
            if o is not None:
                tally.setdefault(sc, []).append(o)

        for sc, outs in tally.items():
            hits, misses = sum(outs), len(outs) - sum(outs)
            post = BetaPosterior().update(hits, misses)
            # Reliability is a trust prior, not a hit rate: keep it in a sane
            # band so one bad month cannot zero out primary-source data.
            rel = max(0.20, min(0.98, 0.5 + (post.mean - 0.5) * 0.8))
            self.store.upsert_source_score(sc, hits, misses, round(rel, 4))

    def _mine_lessons(self, rows) -> None:
        """Find conditions under which the agent is reliably wrong.

        Only conditions with enough observations and a large enough effect are
        promoted to lessons; the rest is noise mining, which would make the
        agent worse rather than better.
        """
        buckets: dict[tuple[str, str], list[tuple[int, float]]] = {}
        for r in rows:
            try:
                feats = json.loads(r["features"] or "{}")
            except json.JSONDecodeError:
                continue
            outcome, realized = int(r["outcome"] or 0), float(r["realized_r"] or 0)
            for key in ("regime", "setup", "gap_bucket", "session_day"):
                val = feats.get(key)
                if val:
                    buckets.setdefault((key, str(val)), []).append((outcome, realized))
            if r["symbol"]:
                buckets.setdefault(("symbol", r["symbol"]), []).append((outcome, realized))

        overall = np.mean([int(r["outcome"] or 0) for r in rows])
        for (key, val), obs in buckets.items():
            n = len(obs)
            if n < 12:
                continue
            hit = float(np.mean([o for o, _ in obs]))
            exp_r = float(np.mean([r for _, r in obs]))
            effect = hit - float(overall)
            if abs(effect) < 0.12:
                continue
            direction = "outperforms" if effect > 0 else "underperforms"
            self.store.add_lesson(
                scope=f"{key}={val}",
                lesson=(
                    f"When {key}={val}: hit rate {hit:.0%} vs {overall:.0%} baseline over {n} calls "
                    f"({direction}, expectancy {exp_r:+.2f}R). "
                    + ("Size up modestly." if effect > 0 else "Require stronger confirmation or skip.")
                ),
                evidence_n=n,
                effect_size=round(effect, 4),
            )

    # ---------- read-side helpers used by the morning run ----------

    def load_calibrator(self) -> PlattCalibrator:
        return PlattCalibrator.from_dict(self.store.load_state(CALIBRATOR_KEY))

    def agent_weights(self, default: float = 1.0) -> dict[str, float]:
        weights = self.store.get_agent_weights()
        return weights or {}

    def calibration_report(self, lookback_days: int = 365) -> dict:
        cutoff = (dt.date.today() - dt.timedelta(days=lookback_days)).isoformat()
        rows = [r for r in self.store.scored_history() if r["run_date"] >= cutoff]
        if len(rows) < 5:
            return {"n": len(rows), "note": "insufficient history for a calibration report"}
        probs = np.array([float(r["probability"]) for r in rows])
        outs = np.array([float(r["outcome"] or 0) for r in rows])
        d = brier_decomposition(probs, outs)
        hits = int(outs.sum())
        decision, llr = sequential_sprt(hits, len(rows))
        return {
            "n": len(rows),
            "hit_rate": float(outs.mean()),
            "mean_forecast": float(probs.mean()),
            "brier": d.brier,
            "reliability": d.reliability,
            "resolution": d.resolution,
            "uncertainty": d.uncertainty,
            "skill_score": d.skill_score,
            "verdict": d.verdict,
            "sprt": {"decision": decision, "llr": llr},
            "curve": reliability_curve(probs, outs),
            "calibrator": self.load_calibrator().as_dict(),
            "lessons": self.store.active_lessons(),
        }
