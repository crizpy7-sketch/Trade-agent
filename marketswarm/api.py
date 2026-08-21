"""Read-only query surface for a future command centre.

Everything here reads. There are no mutating methods, no order placement, no
configuration writes — a dashboard should not be able to change the system it
is displaying, and making that structural is cheaper than reviewing it later.

Plain dicts are returned rather than a web framework's models, so this can be
served by FastAPI, Flask, a CLI, or a template without changing anything.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from dataclasses import dataclass

from . import clock
from .memory.institutional import InstitutionalMemory
from .resilience import BREAKERS

log = logging.getLogger("marketswarm.api")


@dataclass
class ReadOnlyAPI:
    """Query interface over the MarketSwarm database."""

    conn: sqlite3.Connection

    def __post_init__(self):
        self.conn.row_factory = sqlite3.Row
        self.memory = InstitutionalMemory(self.conn)

    # ---------- system ----------

    def system_status(self) -> dict:
        now = clock.now_et()
        status = clock.day_status(now.date())
        last_run = self.conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        return {
            "now_et": now.isoformat(),
            "session": clock.session_at().value,
            "market_open_today": status.is_trading_day,
            "market_status": status.summary,
            "next_trading_day": clock.next_trading_day(now.date()).isoformat(),
            "minutes_to_open": round(clock.minutes_to_open(), 1),
            "last_run": dict(last_run) if last_run else None,
            "schema_version": self._schema_version(),
            "open_circuits": BREAKERS.open_circuits(),
        }

    def _schema_version(self) -> int:
        try:
            row = self.conn.execute(
                "SELECT MAX(version) FROM schema_migrations").fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        except sqlite3.Error:
            return 0

    # ---------- investigations ----------

    def current_investigations(self, limit: int = 20) -> list[dict]:
        return self._rows(
            "SELECT * FROM investigations WHERE status NOT IN ('COMPLETE','FAILED') "
            "ORDER BY created_at DESC LIMIT ?", (limit,))

    def investigation(self, investigation_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM investigations WHERE id=?", (investigation_id,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["evidence"] = self.evidence_for(investigation_id)
        out["recommendations"] = self._rows(
            "SELECT * FROM recommendations WHERE investigation_id=?", (investigation_id,))
        return out

    # ---------- agents ----------

    def agent_status(self, run_id: int | None = None) -> list[dict]:
        if run_id is None:
            row = self.conn.execute("SELECT MAX(run_id) FROM agent_runs").fetchone()
            run_id = row[0] if row else None
        if run_id is None:
            return []
        return self._rows(
            "SELECT agent, status, latency_ms, error, evidence_count, signal_count, "
            "cost_usd FROM agent_runs WHERE run_id=? ORDER BY latency_ms DESC",
            (run_id,))

    def agent_performance(self) -> dict:
        matrix: dict[str, dict] = {}
        for r in self.conn.execute(
            "SELECT agent, context_key, weight, n, contribution_sum "
            "FROM agent_context_scores ORDER BY agent"
        ):
            matrix.setdefault(r["agent"], {})[r["context_key"]] = {
                "weight": round(float(r["weight"]), 3),
                "n": int(r["n"] or 0),
                "mean_contribution": (round(float(r["contribution_sum"]) / r["n"], 5)
                                      if r["n"] else None),
            }
        return matrix

    def agent_latency(self, days: int = 30) -> list[dict]:
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
        return self._rows(
            "SELECT agent, COUNT(*) runs, AVG(latency_ms) avg_ms, MAX(latency_ms) max_ms, "
            "SUM(CASE WHEN status='FAILED' THEN 1 ELSE 0 END) failures "
            "FROM agent_runs WHERE started_at >= ? GROUP BY agent ORDER BY avg_ms DESC",
            (cutoff,))

    # ---------- evidence ----------

    def evidence_for(self, investigation_id: str) -> list[dict]:
        return self._rows(
            "SELECT * FROM evidence_nodes WHERE investigation_id=? "
            "ORDER BY confidence DESC", (investigation_id,))

    def evidence_edges(self, investigation_id: str) -> list[dict]:
        return self._rows(
            "SELECT * FROM evidence_edges WHERE investigation_id=?", (investigation_id,))

    # ---------- recommendations ----------

    # Only these statuses are live recommendations. Everything else — most
    # importantly REJECTED — is history, and serving it here would reintroduce
    # the exact bypass the review gate exists to close.
    ACTIVE_STATUSES = ("APPROVED", "MODIFIED")

    def recommendations(self, run_date: str | None = None, limit: int = 50,
                        include_inactive: bool = False) -> list[dict]:
        """Active recommendations. Rejected candidates are excluded by default
        and are reachable only through `rejected()` or the audit endpoint."""
        status_clause = "" if include_inactive else (
            " AND status IN ({})".format(",".join("?" * len(self.ACTIVE_STATUSES))))
        status_params = () if include_inactive else self.ACTIVE_STATUSES

        if run_date:
            return self._rows(
                f"SELECT * FROM recommendations WHERE run_date=?{status_clause} "
                f"ORDER BY confidence DESC",
                (run_date, *status_params))
        return self._rows(
            f"SELECT * FROM recommendations WHERE 1=1{status_clause} "
            f"ORDER BY created_at DESC LIMIT ?",
            (*status_params, limit))

    def rejected(self, run_date: str | None = None, limit: int = 50) -> list[dict]:
        """Review history: candidates that did not survive the gate."""
        if run_date:
            return self._rows(
                "SELECT * FROM recommendations WHERE run_date=? AND status='REJECTED' "
                "ORDER BY created_at DESC", (run_date,))
        return self._rows(
            "SELECT * FROM recommendations WHERE status='REJECTED' "
            "ORDER BY created_at DESC LIMIT ?", (limit,))

    def recommendation(self, rec_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM recommendations WHERE id=?", (rec_id,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["revisions"] = self._rows(
            "SELECT * FROM recommendation_revisions WHERE recommendation_id=? "
            "ORDER BY iteration", (rec_id,))
        out["after_action"] = self._rows(
            "SELECT * FROM after_action_reviews WHERE recommendation_id=?", (rec_id,))
        return out

    def recommendation_audit(self, rec_id: str) -> dict | None:
        """Full reconstruction: why did MarketSwarm say this?"""
        rec = self.recommendation(rec_id)
        if not rec:
            return None
        inv_id = rec.get("investigation_id")
        return {
            "recommendation": rec,
            "investigation": self.investigation(inv_id) if inv_id else None,
            "evidence": self.evidence_for(inv_id) if inv_id else [],
            "revisions": rec.get("revisions", []),
            "outcome": {"outcome_r": rec.get("outcome_r"),
                        "note": rec.get("outcome_note"),
                        "resolved": bool(rec.get("resolved"))},
        }

    def screened_options(self, run_date: str | None = None,
                         slots: int = 3) -> dict[str, list[dict]]:
        """Fixed call/put screening board for the Discord ``!plays`` command.

        Rejected rows are returned only with an explicit status and reason.
        This read surface does not change the active recommendation query and
        cannot make a rejected candidate trackable.
        """
        slots = max(1, min(int(slots), 10))
        if run_date is None:
            try:
                latest = self.conn.execute(
                    "SELECT run_date FROM recommendations "
                    "WHERE source_kind IN ('call','put') "
                    "ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                run_date = str(latest[0]) if latest else None
            except sqlite3.Error:
                run_date = None

        out: dict[str, list[dict]] = {"calls": [], "puts": []}
        if run_date:
            rows = self._rows(
                "SELECT subject, source_kind, status, rec_type, conviction, "
                "direction, forecast_probability probability, confidence, "
                "expected_r, entry, target, stop, rejection_reason, created_at, "
                "presentation_payload "
                "FROM recommendations WHERE run_date=? "
                "AND source_kind IN ('call','put') "
                "ORDER BY CASE status WHEN 'APPROVED' THEN 0 WHEN 'MODIFIED' THEN 0 "
                "WHEN 'REJECTED' THEN 1 ELSE 2 END, expected_r DESC, confidence DESC",
                (run_date,),
            )
            for row in rows:
                payload = _json_object(row.pop("presentation_payload", None))
                for field in ("strike", "expiration", "option_entry",
                              "option_target", "option_stop"):
                    if payload.get(field) is not None:
                        row[field] = payload[field]
                key = "calls" if row.get("source_kind") == "call" else "puts"
                status = row.get("status")
                actionable = (
                    status in self.ACTIVE_STATUSES
                    and row.get("rec_type") in ("FAVORABLE", "UNFAVORABLE")
                    and row.get("conviction") in ("HIGH_CONVICTION", "MODERATE_CONVICTION")
                )
                if actionable:
                    screen_status, reason = "QUALIFIED", "cleared both gates"
                elif status in self.ACTIVE_STATUSES:
                    screen_status, reason = "WATCH ONLY", str(row.get("rec_type") or "no edge")
                elif status == "REJECTED":
                    screen_status, reason = "REJECTED", str(row.get("rejection_reason") or "review gate")
                else:
                    screen_status, reason = "WITHHELD", str(row.get("rejection_reason") or status)
                row.update({"screen_status": screen_status,
                            "screen_reason": reason, "run_date": run_date})
                if len(out[key]) < slots:
                    out[key].append(row)

        for key, kind in (("calls", "call"), ("puts", "put")):
            while len(out[key]) < slots:
                out[key].append({
                    "subject": None,
                    "source_kind": kind,
                    "screen_status": "DATA UNAVAILABLE",
                    "screen_reason": "no reviewed contract available for this slot",
                    "run_date": run_date,
                })
        return out

    # ---------- performance & learning ----------

    def performance(self, days: int = 90) -> dict:
        cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        rows = self._rows(
            "SELECT kind, outcome, probability, realized_r FROM predictions "
            "WHERE resolved=1 AND run_date >= ?", (cutoff,))
        if not rows:
            return {"n": 0, "note": "no resolved predictions in the window"}
        n = len(rows)
        hits = sum(int(r["outcome"] or 0) for r in rows)
        mean_r = sum(float(r["realized_r"] or 0) for r in rows) / n
        mean_p = sum(float(r["probability"] or 0) for r in rows) / n
        return {
            "n": n, "hit_rate": hits / n, "mean_forecast": mean_p,
            "calibration_gap": mean_p - hits / n, "expectancy_r": mean_r,
            "window_days": days,
        }

    def calibration_history(self, limit: int = 30) -> list[dict]:
        return self._rows(
            "SELECT * FROM calibration_results ORDER BY created_at DESC LIMIT ?", (limit,))

    def learning_metrics(self) -> dict:
        return {
            "agent_weights": self.agent_performance(),
            "performance": self.performance(),
            "memory": self.memory.stats(),
        }

    # ---------- memory & failures ----------

    def memories(self, category: str | None = None, limit: int = 30) -> list[dict]:
        q = "SELECT * FROM memories WHERE active=1"
        params: list = []
        if category:
            q += " AND category=?"
            params.append(category)
        q += " ORDER BY confidence DESC LIMIT ?"
        params.append(limit)
        return self._rows(q, tuple(params))

    def failures(self, limit: int = 30) -> list[dict]:
        return self._rows(
            "SELECT * FROM mistakes ORDER BY created_at DESC LIMIT ?", (limit,))

    def failure_taxonomy(self) -> dict[str, int]:
        return self.memory.mistake_frequency()

    # ---------- experiments ----------

    def experiments(self, limit: int = 30) -> list[dict]:
        return self._rows(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,))

    def experiment_results(self, experiment_id: str) -> list[dict]:
        return self._rows(
            "SELECT * FROM experiment_results WHERE experiment_id=? ORDER BY created_at",
            (experiment_id,))

    # ---------- costs ----------

    def costs(self, days: int = 30) -> dict:
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
        rows = self._rows(
            "SELECT agent, SUM(cost_usd) cost, SUM(llm_input_tokens+llm_output_tokens) tok, "
            "COUNT(*) calls FROM agent_runs WHERE started_at >= ? "
            "AND cost_usd > 0 GROUP BY agent ORDER BY cost DESC", (cutoff,))
        return {
            "window_days": days,
            "total_usd": round(sum(float(r["cost"] or 0) for r in rows), 4),
            "by_agent": rows,
        }

    # ---------- dashboard composite ----------

    def dashboard(self) -> dict:
        """One call for a command-centre landing page."""
        return {
            "system": self.system_status(),
            "investigations": self.current_investigations(limit=10),
            "agents": self.agent_status(),
            "recommendations": self.recommendations(limit=10),
            "performance": self.performance(),
            "failures": self.failure_taxonomy(),
            "experiments": self.experiments(limit=5),
            "costs": self.costs(7),
            "memory": self.memory.stats(),
        }

    # ---------- helpers ----------

    def _rows(self, q: str, params: tuple = ()) -> list[dict]:
        try:
            return [dict(r) for r in self.conn.execute(q, params)]
        except sqlite3.Error as exc:
            # A missing table means migrations have not run yet. Return empty
            # rather than failing a dashboard render.
            log.debug("query failed (%s): %s", q.split()[3] if len(q.split()) > 3 else "?", exc)
            return []


def _json_object(raw) -> dict:
    """Decode optional presentation JSON without making reads fragile."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def open_api(db_path) -> ReadOnlyAPI:
    """Open the database read-only. Enforced at the connection level, so a
    dashboard bug cannot write to production state."""
    from pathlib import Path
    p = Path(db_path).expanduser()
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
    return ReadOnlyAPI(conn)
