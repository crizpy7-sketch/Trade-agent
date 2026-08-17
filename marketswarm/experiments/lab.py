"""Champion / challenger experiment laboratory.

The rule this module exists to enforce:

    SELF-IMPROVEMENT IS NOT SELF-DEPLOYMENT.

A challenger may propose any change to weights, routing, thresholds or scoring.
It runs against history in isolation. It becomes the champion only by passing
every promotion gate, and promotion is an explicit call that a human can require
approval for.

The champion configuration is deliberately immutable from the challenger side:
`Challenger.apply_to()` returns a *copy*, and there is no code path from a
challenger to the live config. That is verified by test, not by convention —
"the agent probably won't overwrite production" is not a safety property.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum

log = logging.getLogger("marketswarm.experiments.lab")


class ExperimentStatus(str, Enum):
    PROPOSED = "proposed"
    RUNNING = "running"
    EVALUATED = "evaluated"
    PROMOTED = "promoted"
    REJECTED = "rejected"


@dataclass
class Challenger:
    """A candidate change. Expressed as a patch, never as mutated state."""

    name: str
    hypothesis: str
    config_patch: dict = field(default_factory=dict)
    kind: str = "weights"
    proposed_by: str = "human"
    parent_id: str | None = None
    id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:12]}")
    created_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    status: ExperimentStatus = ExperimentStatus.PROPOSED

    def apply_to(self, champion_config: dict) -> dict:
        """Return a patched COPY. The champion object is never touched."""
        merged = copy.deepcopy(champion_config)
        for key, value in self.config_patch.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class ExperimentResult:
    experiment_id: str
    dataset: str
    n_samples: int
    n_trades: int
    mean_net_r: float
    hit_rate: float
    brier: float
    skill_score: float
    sharpe_annual: float
    max_drawdown_r: float
    deflated_sharpe: float | None = None
    pbo: float | None = None
    created_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PromotionGates:
    """Every gate must pass. One improved metric is not evidence.

    These thresholds are intentionally hard to clear. The base rate for
    "promising backtest result that survives contact with reality" is low, and
    a permissive gate simply automates self-deception.
    """

    min_sample_size: int = 200
    min_trades: int = 100
    min_mean_r_improvement: float = 0.01     # absolute R per trade
    max_drawdown_worsening: float = 0.20     # 20% relative
    calibration_may_not_worsen: bool = True
    max_brier_worsening: float = 0.005
    require_deflated_sharpe: float | None = 0.90
    max_pbo: float | None = 0.35
    require_positive_absolute_r: bool = True


@dataclass
class PromotionVerdict:
    promoted: bool
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentLab:
    def __init__(self, conn: sqlite3.Connection, gates: PromotionGates | None = None,
                 require_human_approval: bool = True):
        self.conn = conn
        self.gates = gates or PromotionGates()
        # Default ON. An agent that can promote its own changes without a human
        # is a deployment pipeline with no reviewer.
        self.require_human_approval = require_human_approval

    # ---------- champion ----------

    def champion(self) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE role='champion' AND status='promoted' "
            "ORDER BY promoted_at DESC LIMIT 1").fetchone()
        if not row:
            return None
        return {"id": row["id"], "name": row["name"],
                "config_patch": json.loads(row["config_patch"] or "{}"),
                "promoted_at": row["promoted_at"]}

    def set_initial_champion(self, name: str = "baseline_1x",
                             config_patch: dict | None = None) -> str:
        existing = self.champion()
        if existing:
            return existing["id"]
        cid = f"exp_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            """INSERT INTO experiments
               (id, created_at, name, hypothesis, kind, role, status, config_patch,
                proposed_by, promoted_at)
               VALUES (?,?,?,?,?,'champion','promoted',?,?,?)""",
            (cid, dt.datetime.now(dt.timezone.utc).isoformat(), name,
             "the production configuration", "baseline",
             json.dumps(config_patch or {}), "system",
             dt.datetime.now(dt.timezone.utc).isoformat()))
        self.conn.commit()
        return cid

    # ---------- challengers ----------

    def register(self, challenger: Challenger) -> str:
        self.conn.execute(
            """INSERT INTO experiments
               (id, created_at, name, hypothesis, kind, role, status, config_patch,
                proposed_by, parent_id)
               VALUES (?,?,?,?,?,'challenger',?,?,?,?)""",
            (challenger.id, challenger.created_at, challenger.name,
             challenger.hypothesis, challenger.kind, challenger.status.value,
             json.dumps(challenger.config_patch), challenger.proposed_by,
             challenger.parent_id))
        self.conn.commit()
        log.info("registered challenger %s (%s)", challenger.name, challenger.id)
        return challenger.id

    def record_result(self, result: ExperimentResult) -> None:
        self.conn.execute(
            """INSERT INTO experiment_results
               (experiment_id, created_at, dataset, n_samples, n_trades, mean_net_r,
                hit_rate, brier, skill_score, sharpe_annual, max_drawdown_r,
                deflated_sharpe, pbo, raw)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (result.experiment_id, result.created_at, result.dataset,
             result.n_samples, result.n_trades, result.mean_net_r, result.hit_rate,
             result.brier, result.skill_score, result.sharpe_annual,
             result.max_drawdown_r, result.deflated_sharpe, result.pbo,
             json.dumps(result.to_dict())))
        self.conn.execute("UPDATE experiments SET status='evaluated' WHERE id=?",
                          (result.experiment_id,))
        self.conn.commit()

    # ---------- gates ----------

    def evaluate(self, challenger_result: ExperimentResult,
                 champion_result: ExperimentResult) -> PromotionVerdict:
        """Run every gate. All must pass."""
        g = self.gates
        passed: list[str] = []
        failed: list[str] = []

        def check(ok: bool, label: str) -> None:
            (passed if ok else failed).append(label)

        check(challenger_result.n_samples >= g.min_sample_size,
              f"sample size {challenger_result.n_samples} >= {g.min_sample_size}")
        check(challenger_result.n_trades >= g.min_trades,
              f"trades {challenger_result.n_trades} >= {g.min_trades}")

        improvement = challenger_result.mean_net_r - champion_result.mean_net_r
        check(improvement >= g.min_mean_r_improvement,
              f"mean R improvement {improvement:+.4f} >= {g.min_mean_r_improvement}")

        if g.require_positive_absolute_r:
            check(challenger_result.mean_net_r > 0,
                  f"absolute mean R {challenger_result.mean_net_r:+.4f} > 0")

        champ_dd = abs(champion_result.max_drawdown_r) or 1e-9
        dd_ratio = abs(challenger_result.max_drawdown_r) / champ_dd
        check(dd_ratio <= 1 + g.max_drawdown_worsening,
              f"drawdown ratio {dd_ratio:.2f} <= {1 + g.max_drawdown_worsening:.2f}")

        if g.calibration_may_not_worsen:
            brier_delta = challenger_result.brier - champion_result.brier
            check(brier_delta <= g.max_brier_worsening,
                  f"Brier change {brier_delta:+.4f} <= {g.max_brier_worsening}")

        if g.require_deflated_sharpe is not None:
            dsr = challenger_result.deflated_sharpe
            check(dsr is not None and dsr >= g.require_deflated_sharpe,
                  f"deflated Sharpe {dsr if dsr is not None else 'missing'} "
                  f">= {g.require_deflated_sharpe}")

        if g.max_pbo is not None:
            pbo = challenger_result.pbo
            check(pbo is not None and pbo <= g.max_pbo,
                  f"PBO {pbo if pbo is not None else 'missing'} <= {g.max_pbo}")

        promoted = not failed
        summary = (f"{len(passed)}/{len(passed) + len(failed)} gates passed — "
                   + ("eligible for promotion" if promoted
                      else f"blocked by: {'; '.join(failed[:3])}"))
        return PromotionVerdict(promoted, passed, failed, summary)

    def promote(self, challenger_id: str, verdict: PromotionVerdict,
                human_approved: bool = False) -> bool:
        """Promote a challenger to champion.

        Refuses unless every gate passed AND, when human approval is required,
        it has been explicitly given. There is no override parameter, because an
        override would be the first thing an over-eager automation reached for.
        """
        if not verdict.promoted:
            self.conn.execute(
                "UPDATE experiments SET status='rejected', rejected_reason=? WHERE id=?",
                (verdict.summary[:500], challenger_id))
            self.conn.commit()
            log.info("challenger %s rejected: %s", challenger_id, verdict.summary)
            return False

        if self.require_human_approval and not human_approved:
            log.warning("challenger %s passed all gates but awaits human approval",
                        challenger_id)
            self.conn.execute(
                "UPDATE experiments SET status='evaluated', rejected_reason=? WHERE id=?",
                ("passed gates; awaiting human approval", challenger_id))
            self.conn.commit()
            return False

        now = dt.datetime.now(dt.timezone.utc).isoformat()
        self.conn.execute("UPDATE experiments SET role='challenger' WHERE role='champion'")
        self.conn.execute(
            "UPDATE experiments SET role='champion', status='promoted', promoted_at=? "
            "WHERE id=?", (now, challenger_id))
        self.conn.commit()
        log.info("challenger %s promoted to champion", challenger_id)
        return True

    # ---------- queries ----------

    def list_experiments(self, status: ExperimentStatus | None = None,
                         limit: int = 50) -> list[dict]:
        q = "SELECT * FROM experiments"
        params: list = []
        if status:
            q += " WHERE status=?"
            params.append(status.value)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(q, params)]

    def results_for(self, experiment_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM experiment_results WHERE experiment_id=? ORDER BY created_at",
            (experiment_id,))]

    def stats(self) -> dict:
        by_status = {r[0]: r[1] for r in self.conn.execute(
            "SELECT status, COUNT(*) FROM experiments GROUP BY status")}
        champ = self.champion()
        return {
            "champion": champ["name"] if champ else None,
            "by_status": by_status,
            "require_human_approval": self.require_human_approval,
            "total_results": self.conn.execute(
                "SELECT COUNT(*) FROM experiment_results").fetchone()[0],
        }
