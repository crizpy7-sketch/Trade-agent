"""The replay engine.

Walks history one session at a time, and on each one:
  1. builds point-in-time features (no lookahead, enforced by the store)
  2. gets a probability from the learned model, trained only on prior folds
  3. builds a bracket with the *production* `_build_bracket` — the same code
     the live agent runs, so this measures the real system
  4. prices the idea with the production barrier Monte Carlo and EV gate
  5. resolves it against the actual session bar, charging realistic friction

Point 3 is the part that matters. A backtest of a reimplementation measures the
reimplementation. This imports the live functions so that a change to the
trading logic changes the backtest too, and a divergence is impossible.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np

from ..agents.synthesis import _build_bracket
from ..stats import distributions as dist
from ..stats import edge as edgemod
from ..stats.calibration import PlattCalibrator, brier_decomposition
from .datastore import PointInTimeStore
from .features import FEATURE_NAMES, build_features, resolve_outcome
from .fills import FillModel, LIQUID_PRESET, STANDARD_PRESET
from .validation import purged_walk_forward_splits, summarize_returns

log = logging.getLogger("marketswarm.backtest.replay")

LIQUID = {"SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD"}


@dataclass
class Trade:
    date: dt.date
    symbol: str
    direction: str
    entry: float
    target: float
    stop: float
    model_prob: float
    calibrated_prob: float
    barrier_prob: float
    expected_r: float
    realized_r: float
    outcome: int
    cost_r: float
    regime: str
    note: str
    taken: bool

    @property
    def net_r(self) -> float:
        return self.realized_r - self.cost_r


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    fold_reports: list[dict] = field(default_factory=list)
    feature_importances: list[tuple[str, float]] = field(default_factory=list)
    n_candidates: int = 0
    n_bracketed: int = 0
    config: dict = field(default_factory=dict)

    @property
    def taken(self) -> list[Trade]:
        return [t for t in self.trades if t.taken]

    def returns(self, net: bool = True) -> np.ndarray:
        return np.array([t.net_r if net else t.realized_r for t in self.taken], dtype=float)

    def probabilities(self) -> np.ndarray:
        return np.array([t.calibrated_prob for t in self.taken], dtype=float)

    def outcomes(self) -> np.ndarray:
        return np.array([float(t.outcome) for t in self.taken], dtype=float)

    def summary(self) -> dict:
        taken = self.taken
        if not taken:
            return {"n": 0, "note": "no trades cleared the expectancy gate"}

        gross = summarize_returns(self.returns(net=False))
        net = summarize_returns(self.returns(net=True))
        if "mean_r" not in gross or "mean_r" not in net:
            return {"n": len(taken), "note": "too few trades to summarise",
                    "mean_net_r": float(np.mean([t.net_r for t in taken]))}
        cal = brier_decomposition(self.probabilities(), self.outcomes())

        by_regime: dict[str, dict] = {}
        for t in taken:
            b = by_regime.setdefault(t.regime, {"n": 0, "r": 0.0, "hits": 0})
            b["n"] += 1
            b["r"] += t.net_r
            b["hits"] += t.outcome
        for b in by_regime.values():
            b["mean_r"] = b["r"] / b["n"]
            b["hit_rate"] = b["hits"] / b["n"]

        return {
            "n_candidates": self.n_candidates,
            "n_bracketed": self.n_bracketed,
            "n_taken": len(taken),
            "selectivity": len(taken) / max(self.n_candidates, 1),
            "gross": gross,
            "net": net,
            "cost_drag_r": gross["mean_r"] - net["mean_r"],
            "calibration": {
                "brier": cal.brier,
                "reliability": cal.reliability,
                "resolution": cal.resolution,
                "skill_score": cal.skill_score,
                "verdict": cal.verdict(),
                "mean_forecast": float(self.probabilities().mean()),
                "actual_hit_rate": float(self.outcomes().mean()),
            },
            "by_regime": by_regime,
        }


class BacktestEngine:
    def __init__(
        self,
        store: PointInTimeStore,
        symbols: list[str],
        target_atr: float = 1.0,
        stop_atr: float = 0.6,
        min_expected_r: float = 0.0,
        mc_paths: int = 2000,
        market_symbol: str = "SPY",
        fill_model: FillModel | None = None,
        account_size: float = 50_000.0,
        risk_pct: float = 0.75,
    ):
        self.store = store
        self.symbols = [s.upper() for s in symbols]
        self.target_atr = target_atr
        self.stop_atr = stop_atr
        self.min_expected_r = min_expected_r
        self.mc_paths = mc_paths
        self.market_symbol = market_symbol
        self.fills = fill_model or STANDARD_PRESET
        self.account_size = account_size
        self.risk_pct = risk_pct

    # ----------------------------------------------------------------

    def run_walk_forward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: list[dict],
        n_splits: int = 5,
        embargo_days: int = 5,
        l2: float = 1.0,
        calibrate: bool = True,
    ) -> BacktestResult:
        """Train and evaluate across purged walk-forward folds.

        The model in each fold has seen only data that predates its test block,
        after purging and embargo. The Platt calibrator is fit on the *training*
        fold too — calibrating on the test set would be the subtlest and most
        flattering leak available.
        """
        from ..models.logistic import LogisticModel

        result = BacktestResult(config={
            "symbols": len(self.symbols), "target_atr": self.target_atr,
            "stop_atr": self.stop_atr, "min_expected_r": self.min_expected_r,
            "n_splits": n_splits, "embargo_days": embargo_days, "l2": l2,
            "mc_paths": self.mc_paths,
        })

        dates = np.array([m["date"] for m in meta])
        splits = purged_walk_forward_splits(dates, n_splits=n_splits, embargo_days=embargo_days)
        if not splits:
            log.error("not enough history for %d purged folds", n_splits)
            return result

        for k, sp in enumerate(splits, 1):
            log.info("fold %d: %s", k, sp.describe())

            model = LogisticModel(feature_names=list(FEATURE_NAMES), l2=l2)
            model.fit(X[sp.train_idx], y[sp.train_idx])

            calibrator = PlattCalibrator()
            if calibrate:
                # Fit calibration on a held-out tail of the training block, so
                # neither the model nor the calibrator has seen the test data.
                n_tr = len(sp.train_idx)
                cut = int(n_tr * 0.8)
                if n_tr - cut >= 50:
                    inner_train, inner_cal = sp.train_idx[:cut], sp.train_idx[cut:]
                    inner_model = LogisticModel(feature_names=list(FEATURE_NAMES), l2=l2)
                    inner_model.fit(X[inner_train], y[inner_train])
                    calibrator.fit(inner_model.predict_proba(X[inner_cal]), y[inner_cal])

            probs = model.predict_proba(X[sp.test_idx])
            fold_trades: list[Trade] = []

            for local_i, global_i in enumerate(sp.test_idx):
                trade = self._evaluate_candidate(meta[global_i], float(probs[local_i]), calibrator)
                if trade is not None:
                    fold_trades.append(trade)

            result.trades.extend(fold_trades)
            taken = [t for t in fold_trades if t.taken]
            fold_summary = {
                "fold": k,
                "train": [str(sp.train_range[0]), str(sp.train_range[1])],
                "test": [str(sp.test_range[0]), str(sp.test_range[1])],
                "n_train": len(sp.train_idx),
                "n_test": len(sp.test_idx),
                "purged": sp.purged,
                "embargoed": sp.embargoed,
                "n_taken": len(taken),
                "mean_net_r": float(np.mean([t.net_r for t in taken])) if taken else 0.0,
                "hit_rate": float(np.mean([t.outcome for t in taken])) if taken else 0.0,
                "calibrator": calibrator.as_dict(),
            }
            result.fold_reports.append(fold_summary)
            log.info("  fold %d: %d taken, mean net %+.3fR, hit %.1f%%", k, len(taken),
                     fold_summary["mean_net_r"], fold_summary["hit_rate"] * 100)

            if k == len(splits):
                result.feature_importances = model.importances()

        result.n_candidates = len(result.trades)
        result.n_bracketed = len(result.trades)
        return result

    # ----------------------------------------------------------------

    def _evaluate_candidate(self, m: dict, model_prob: float,
                            calibrator: PlattCalibrator) -> Trade | None:
        """Turn one scored feature row into a simulated trade decision."""
        entry, atr, sigma = m["entry"], m["atr"], m["sigma"]
        if entry <= 0 or atr <= 0:
            return None

        p = calibrator.transform(model_prob)
        long_side = p >= 0.5

        # Production bracket logic — same function the live agent calls.
        max_reach = atr * self.target_atr
        bracket = _build_bracket(
            price=entry, atr=atr, max_reach=max_reach, long_side=long_side,
            support=m.get("support"), resistance=m.get("resistance"),
            min_risk_atr=self.stop_atr, max_risk_atr=self.stop_atr * 1.8,
        )
        if bracket is None:
            return None
        _, target, stop = bracket

        drift = (p - 0.5) * sigma * 2 * (1 if long_side else -1)
        barrier = dist.barrier_probabilities(
            entry=entry, target=target, stop=stop, sigma_daily=sigma,
            horizon_days=1.0, drift_daily=drift, n_paths=self.mc_paths,
            seed=0,
        )

        preset = LIQUID_PRESET if m["symbol"] in LIQUID else self.fills
        shares = max(1.0, (self.account_size * self.risk_pct / 100) / max(abs(entry - stop), 1e-6))
        adv = 2_000_000.0
        cost_r = min(0.5, preset.round_trip_cost_r(entry, stop, shares, adv, daily_vol=sigma))

        trade_math = edgemod.evaluate_bracket(barrier, entry, target, stop, cost_r=cost_r)
        taken = trade_math.edge_after_costs > self.min_expected_r

        outcome = resolve_outcome(self.store, m["symbol"], m["date"], entry, target, stop)
        if outcome is None:
            return None

        return Trade(
            date=m["date"], symbol=m["symbol"],
            direction="long" if long_side else "short",
            entry=entry, target=target, stop=stop,
            model_prob=model_prob, calibrated_prob=p,
            barrier_prob=barrier.p_target_first,
            expected_r=trade_math.edge_after_costs,
            realized_r=outcome[1], outcome=outcome[0],
            cost_r=cost_r, regime=m.get("regime", "unknown"),
            note=outcome[2], taken=taken,
        )

    # ----------------------------------------------------------------

    def run_baselines(self, X: np.ndarray, y: np.ndarray, meta: list[dict]) -> dict:
        """Null models the strategy must beat to be worth anything.

        Without these, any positive number looks like skill. Most of the time
        the honest finding is that always-long matched it.
        """
        outcomes = np.array([m["realized_r"] for m in meta], float)
        rng = np.random.default_rng(42)

        coin = rng.random(len(outcomes)) < 0.5
        return {
            "always_long": {
                "n": len(outcomes),
                "mean_r": float(outcomes.mean()),
                "hit_rate": float((outcomes > 0).mean()),
            },
            "random_entry": {
                "n": int(coin.sum()),
                "mean_r": float(outcomes[coin].mean()) if coin.sum() else 0.0,
                "hit_rate": float((outcomes[coin] > 0).mean()) if coin.sum() else 0.0,
            },
            "base_rate": float(y.mean()),
        }
