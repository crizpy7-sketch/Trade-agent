"""Durable memory: every prediction, its evidence, and what actually happened.

SQLite because it needs zero operational care on a VPS, survives restarts, and
is trivially backed up by copying one file. The schema is designed so the agent
can answer "what kind of call do I get wrong, and under what conditions?" —
not merely "what is my win rate".
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("marketswarm.memory")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    session_type TEXT,
    regime TEXT,
    agents_ok INTEGER DEFAULT 0,
    agents_failed INTEGER DEFAULT 0,
    report_path TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    run_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,              -- direction | call_idea | put_idea | stock_setup
    symbol TEXT NOT NULL,
    thesis TEXT,
    direction TEXT,                  -- long | short
    probability REAL NOT NULL,       -- calibrated P(target before stop)
    raw_probability REAL,            -- pre-calibration
    confidence INTEGER,
    entry REAL, target REAL, stop REAL,
    expected_r REAL,
    horizon TEXT DEFAULT 'intraday',
    features TEXT,                   -- JSON snapshot of the deciding inputs
    contributing_agents TEXT,        -- JSON {agent: llr_contribution}
    invalidation TEXT,
    resolved INTEGER DEFAULT 0,
    outcome INTEGER,                 -- 1 target first, 0 stop first
    realized_r REAL,
    resolution_note TEXT,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    prediction_id INTEGER REFERENCES predictions(id),
    claim TEXT NOT NULL,
    source TEXT NOT NULL,
    source_class TEXT,
    url TEXT,
    reliability REAL,
    observed_at TEXT,
    tags TEXT
);

CREATE TABLE IF NOT EXISTS agent_scores (
    agent TEXT PRIMARY KEY,
    hits INTEGER DEFAULT 0,
    misses INTEGER DEFAULT 0,
    brier_sum REAL DEFAULT 0,
    n_scored INTEGER DEFAULT 0,
    weight REAL DEFAULT 1.0,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS source_scores (
    source_class TEXT PRIMARY KEY,
    hits INTEGER DEFAULT 0,
    misses INTEGER DEFAULT 0,
    reliability REAL DEFAULT 0.5,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS calibration_state (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    scope TEXT,                      -- regime / setup / symbol
    lesson TEXT NOT NULL,
    evidence_n INTEGER,
    effect_size REAL,
    active INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(run_date);
CREATE INDEX IF NOT EXISTS idx_pred_unresolved ON predictions(resolved, run_date);
CREATE INDEX IF NOT EXISTS idx_pred_symbol ON predictions(symbol);
"""


@dataclass
class Prediction:
    run_date: str
    kind: str
    symbol: str
    direction: str
    probability: float
    entry: float | None = None
    target: float | None = None
    stop: float | None = None
    expected_r: float | None = None
    raw_probability: float | None = None
    confidence: int | None = None
    thesis: str = ""
    invalidation: str = ""
    horizon: str = "intraday"
    features: dict[str, Any] = field(default_factory=dict)
    contributing_agents: dict[str, float] = field(default_factory=dict)
    id: int | None = None

    def to_row(self, run_id: int | None) -> tuple:
        return (
            run_id,
            self.run_date,
            dt.datetime.now(dt.timezone.utc).isoformat(),
            self.kind,
            self.symbol,
            self.thesis,
            self.direction,
            float(self.probability),
            self.raw_probability,
            self.confidence,
            self.entry,
            self.target,
            self.stop,
            self.expected_r,
            self.horizon,
            json.dumps(self.features, default=str),
            json.dumps(self.contributing_agents, default=str),
            self.invalidation,
        )


class MemoryStore:
    def __init__(self, path: Path | str = "~/.marketswarm/memory.db"):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------- runs ----------

    def start_run(self, run_date: str, session_type: str = "premarket") -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (run_date, started_at, session_type) VALUES (?, ?, ?)",
            (run_date, dt.datetime.now(dt.timezone.utc).isoformat(), session_type),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, *, regime: str = "", agents_ok: int = 0,
                   agents_failed: int = 0, report_path: str = "", notes: str = "") -> None:
        self.conn.execute(
            """UPDATE runs SET finished_at=?, regime=?, agents_ok=?, agents_failed=?,
               report_path=?, notes=? WHERE id=?""",
            (dt.datetime.now(dt.timezone.utc).isoformat(), regime, agents_ok, agents_failed,
             report_path, notes, run_id),
        )
        self.conn.commit()

    # ---------- predictions ----------

    def record_prediction(self, pred: Prediction, run_id: int | None = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO predictions
               (run_id, run_date, created_at, kind, symbol, thesis, direction, probability,
                raw_probability, confidence, entry, target, stop, expected_r, horizon,
                features, contributing_agents, invalidation)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            pred.to_row(run_id),
        )
        self.conn.commit()
        pred.id = int(cur.lastrowid)
        return pred.id

    def record_evidence(self, items: list[dict], run_id: int | None = None,
                        prediction_id: int | None = None) -> None:
        self.conn.executemany(
            """INSERT INTO evidence (run_id, prediction_id, claim, source, source_class, url,
                                     reliability, observed_at, tags)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [
                (run_id, prediction_id, e.get("claim", ""), e.get("source", ""),
                 e.get("source_class", ""), e.get("url"), e.get("reliability"),
                 e.get("observed_at"), json.dumps(e.get("tags", [])))
                for e in items
            ],
        )
        self.conn.commit()

    def unresolved_predictions(self, before_date: str | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM predictions WHERE resolved=0"
        params: list[Any] = []
        if before_date:
            q += " AND run_date <= ?"
            params.append(before_date)
        q += " ORDER BY run_date"
        return list(self.conn.execute(q, params))

    def resolve(self, pred_id: int, outcome: int, realized_r: float, note: str = "") -> None:
        self.conn.execute(
            """UPDATE predictions SET resolved=1, outcome=?, realized_r=?, resolution_note=?,
               resolved_at=? WHERE id=?""",
            (int(outcome), float(realized_r), note, dt.datetime.now(dt.timezone.utc).isoformat(), pred_id),
        )
        self.conn.commit()

    def scored_history(self, kind: str | None = None, limit: int = 2000) -> list[sqlite3.Row]:
        q = "SELECT * FROM predictions WHERE resolved=1"
        params: list[Any] = []
        if kind:
            q += " AND kind=?"
            params.append(kind)
        q += " ORDER BY run_date DESC LIMIT ?"
        params.append(limit)
        return list(self.conn.execute(q, params))

    def performance_summary(self, days: int = 90) -> dict:
        cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        rows = list(
            self.conn.execute(
                "SELECT kind, outcome, probability, realized_r FROM predictions "
                "WHERE resolved=1 AND run_date >= ?",
                (cutoff,),
            )
        )
        if not rows:
            return {"n": 0, "note": "no resolved predictions yet"}
        by_kind: dict[str, dict] = {}
        for r in rows:
            b = by_kind.setdefault(r["kind"], {"n": 0, "hits": 0, "r_sum": 0.0, "p_sum": 0.0})
            b["n"] += 1
            b["hits"] += int(r["outcome"] or 0)
            b["r_sum"] += float(r["realized_r"] or 0)
            b["p_sum"] += float(r["probability"] or 0)
        for b in by_kind.values():
            b["hit_rate"] = b["hits"] / b["n"]
            b["mean_forecast"] = b["p_sum"] / b["n"]
            b["expectancy_r"] = b["r_sum"] / b["n"]
            b["calibration_gap"] = b["mean_forecast"] - b["hit_rate"]
        return {"n": len(rows), "window_days": days, "by_kind": by_kind}

    # ---------- learned weights ----------

    def get_agent_weights(self) -> dict[str, float]:
        return {r["agent"]: float(r["weight"]) for r in self.conn.execute("SELECT agent, weight FROM agent_scores")}

    def get_agent_record(self, agent: str) -> tuple[int, int]:
        row = self.conn.execute("SELECT hits, misses FROM agent_scores WHERE agent=?", (agent,)).fetchone()
        return (int(row["hits"]), int(row["misses"])) if row else (0, 0)

    def upsert_agent_score(self, agent: str, hits: int, misses: int, brier_sum: float,
                           n_scored: int, weight: float) -> None:
        self.conn.execute(
            """INSERT INTO agent_scores (agent, hits, misses, brier_sum, n_scored, weight, updated_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(agent) DO UPDATE SET hits=excluded.hits, misses=excluded.misses,
                 brier_sum=excluded.brier_sum, n_scored=excluded.n_scored,
                 weight=excluded.weight, updated_at=excluded.updated_at""",
            (agent, hits, misses, brier_sum, n_scored, weight, dt.datetime.now(dt.timezone.utc).isoformat()),
        )
        self.conn.commit()

    def get_source_reliability(self) -> dict[str, float]:
        return {
            r["source_class"]: float(r["reliability"])
            for r in self.conn.execute("SELECT source_class, reliability FROM source_scores")
        }

    def upsert_source_score(self, source_class: str, hits: int, misses: int, reliability: float) -> None:
        self.conn.execute(
            """INSERT INTO source_scores (source_class, hits, misses, reliability, updated_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(source_class) DO UPDATE SET hits=excluded.hits, misses=excluded.misses,
                 reliability=excluded.reliability, updated_at=excluded.updated_at""",
            (source_class, hits, misses, reliability, dt.datetime.now(dt.timezone.utc).isoformat()),
        )
        self.conn.commit()

    def save_state(self, key: str, payload: dict) -> None:
        self.conn.execute(
            """INSERT INTO calibration_state (key, payload, updated_at) VALUES (?,?,?)
               ON CONFLICT(key) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at""",
            (key, json.dumps(payload), dt.datetime.now(dt.timezone.utc).isoformat()),
        )
        self.conn.commit()

    def load_state(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT payload FROM calibration_state WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["payload"])
        except json.JSONDecodeError:
            return None

    # ---------- lessons ----------

    def add_lesson(self, scope: str, lesson: str, evidence_n: int, effect_size: float) -> None:
        exists = self.conn.execute(
            "SELECT id FROM lessons WHERE scope=? AND lesson=? AND active=1", (scope, lesson)
        ).fetchone()
        if exists:
            self.conn.execute(
                "UPDATE lessons SET evidence_n=?, effect_size=? WHERE id=?",
                (evidence_n, effect_size, exists["id"]),
            )
        else:
            self.conn.execute(
                "INSERT INTO lessons (created_at, scope, lesson, evidence_n, effect_size) VALUES (?,?,?,?,?)",
                (dt.datetime.now(dt.timezone.utc).isoformat(), scope, lesson, evidence_n, effect_size),
            )
        self.conn.commit()

    def active_lessons(self, limit: int = 12) -> list[dict]:
        rows = self.conn.execute(
            "SELECT scope, lesson, evidence_n, effect_size FROM lessons WHERE active=1 "
            "ORDER BY ABS(effect_size) DESC, evidence_n DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]
