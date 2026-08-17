"""Versioned schema migrations.

1.x created tables with `CREATE TABLE IF NOT EXISTS` and no version tracking,
which silently does nothing when a column is added to an existing install. Every
2.0 schema change goes through this module instead.

Rules:
  - migrations are append-only and numbered; never edit a shipped migration
  - each runs inside a transaction and is recorded in `schema_migrations`
  - existing 1.x data is preserved; nothing is dropped or rewritten
  - a database ahead of this code is left alone rather than downgraded
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from dataclasses import dataclass

log = logging.getLogger("marketswarm.migrations")

SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


# --------------------------------------------------------------------------
# 001 — baseline marker.
# The 1.x tables are created by store.SCHEMA before migrations run, so this
# records that an existing database has been adopted rather than recreated.
# --------------------------------------------------------------------------

M001 = Migration(1, "baseline_1x", "SELECT 1;")


# --------------------------------------------------------------------------
# 002 — investigations, evidence graph, recommendations
# --------------------------------------------------------------------------

M002 = Migration(2, "investigations_and_evidence_graph", """
CREATE TABLE IF NOT EXISTS investigations (
    id TEXT PRIMARY KEY,
    run_id INTEGER,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    subject TEXT NOT NULL,
    trigger TEXT,
    priority TEXT,
    status TEXT NOT NULL DEFAULT 'PLANNING',
    regime TEXT,
    regime_confidence REAL,
    hypotheses TEXT,
    required_agents TEXT,
    optional_agents TEXT,
    agents_run TEXT,
    agents_skipped TEXT,
    evidence_needed TEXT,
    stop_reason TEXT,
    iterations INTEGER DEFAULT 0,
    max_iterations INTEGER DEFAULT 3,
    cost_estimate REAL DEFAULT 0,
    latency_ms INTEGER,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_inv_subject ON investigations(subject, created_at);
CREATE INDEX IF NOT EXISTS idx_inv_status ON investigations(status);

-- Evidence nodes. Deliberately separate from the 1.x `evidence` table, which
-- stays intact for historical continuity.
CREATE TABLE IF NOT EXISTS evidence_nodes (
    id TEXT PRIMARY KEY,
    investigation_id TEXT REFERENCES investigations(id),
    created_at TEXT NOT NULL,
    node_type TEXT NOT NULL,          -- observation|claim|hypothesis|recommendation|event
    subject TEXT,
    claim TEXT NOT NULL,
    raw_observation TEXT,
    source TEXT,
    source_type TEXT,
    url TEXT,
    observed_at TEXT,
    retrieved_at TEXT,
    cluster TEXT,                     -- correlation family
    factual_reliability REAL,
    predictive_utility REAL,
    timeliness REAL,
    novelty REAL,
    market_impact REAL,
    independence REAL,
    confidence REAL,
    is_inferred INTEGER DEFAULT 0,    -- inferred vs directly observed
    provenance TEXT,
    tags TEXT
);
CREATE INDEX IF NOT EXISTS idx_ev_inv ON evidence_nodes(investigation_id);
CREATE INDEX IF NOT EXISTS idx_ev_subject ON evidence_nodes(subject);
CREATE INDEX IF NOT EXISTS idx_ev_cluster ON evidence_nodes(cluster);

CREATE TABLE IF NOT EXISTS evidence_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id TEXT REFERENCES investigations(id),
    src_id TEXT NOT NULL REFERENCES evidence_nodes(id),
    dst_id TEXT NOT NULL REFERENCES evidence_nodes(id),
    relation TEXT NOT NULL,           -- supports|contradicts|derives_from|duplicates|explains
    weight REAL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_edge_src ON evidence_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edge_dst ON evidence_edges(dst_id);

CREATE TABLE IF NOT EXISTS recommendations (
    id TEXT PRIMARY KEY,
    investigation_id TEXT REFERENCES investigations(id),
    run_id INTEGER,
    prediction_id INTEGER REFERENCES predictions(id),
    created_at TEXT NOT NULL,
    run_date TEXT NOT NULL,
    subject TEXT NOT NULL,
    rec_type TEXT NOT NULL,           -- WATCH|FAVORABLE|NO_EDGE|...
    conviction TEXT NOT NULL,         -- HIGH_CONVICTION|...|INSUFFICIENT_EVIDENCE
    direction TEXT,
    forecast_probability REAL,
    confidence INTEGER,
    original_confidence INTEGER,
    expected_r REAL,
    entry REAL, target REAL, stop REAL,
    horizon TEXT,
    regime TEXT,
    data_quality TEXT,
    observation TEXT,
    interpretation TEXT,
    rationale TEXT,
    key_risks TEXT,
    invalidation TEXT,
    change_our_mind TEXT,
    supporting_evidence TEXT,
    contradicting_evidence TEXT,
    historical_analogues TEXT,
    agent_contributors TEXT,
    review_status TEXT,
    revision_count INTEGER DEFAULT 0,
    model_version TEXT,
    system_version TEXT,
    experiment_id TEXT,
    outcome_r REAL,
    outcome_note TEXT,
    resolved INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_rec_date ON recommendations(run_date);
CREATE INDEX IF NOT EXISTS idx_rec_subject ON recommendations(subject);
CREATE INDEX IF NOT EXISTS idx_rec_unresolved ON recommendations(resolved, run_date);

CREATE TABLE IF NOT EXISTS recommendation_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id TEXT NOT NULL REFERENCES recommendations(id),
    iteration INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    stage TEXT,                       -- original|red_team|revision|final
    status TEXT,
    severity TEXT,
    confidence_before INTEGER,
    confidence_after INTEGER,
    findings TEXT,
    rejected_claims TEXT,
    accepted_claims TEXT,
    required_followups TEXT,
    rationale TEXT,
    snapshot TEXT                     -- JSON of the idea at this point
);
CREATE INDEX IF NOT EXISTS idx_rev_rec ON recommendation_revisions(recommendation_id, iteration);
""")


# --------------------------------------------------------------------------
# 003 — contextual learning, regimes, memory, mistakes, experiments
# --------------------------------------------------------------------------

M003 = Migration(3, "learning_memory_experiments", """
-- Per-agent performance sliced by context, replacing the single scalar weight.
CREATE TABLE IF NOT EXISTS agent_context_scores (
    agent TEXT NOT NULL,
    context_key TEXT NOT NULL,        -- e.g. regime=stress, event=earnings
    n INTEGER DEFAULT 0,
    hits INTEGER DEFAULT 0,
    contribution_sum REAL DEFAULT 0,  -- summed ablation deltas (log-loss improvement)
    brier_with REAL DEFAULT 0,
    brier_without REAL DEFAULT 0,
    weight REAL DEFAULT 1.0,
    weight_lo REAL,
    weight_hi REAL,
    updated_at TEXT,
    PRIMARY KEY (agent, context_key)
);

CREATE TABLE IF NOT EXISTS market_regimes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_on TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL,
    vix REAL,
    realized_vol REAL,
    trend_score REAL,
    breadth REAL,
    evidence TEXT,
    UNIQUE(observed_on)
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    category TEXT NOT NULL,           -- episodic|semantic|company|agent|strategy|failure|experiment
    subject TEXT,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    evidence_n INTEGER DEFAULT 0,
    effect_size REAL,
    source TEXT,
    provenance TEXT,
    related_entities TEXT,
    version INTEGER DEFAULT 1,
    supersedes TEXT,
    contradicted_by TEXT,
    valid_until TEXT,                 -- staleness horizon
    last_accessed TEXT,
    access_count INTEGER DEFAULT 0,
    active INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mem_cat ON memories(category, active);
CREATE INDEX IF NOT EXISTS idx_mem_subject ON memories(subject, active);

CREATE TABLE IF NOT EXISTS mistakes (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    recommendation_id TEXT REFERENCES recommendations(id),
    run_date TEXT,
    subject TEXT,
    predicted TEXT,
    actual TEXT,
    error_r REAL,
    overweighted TEXT,
    ignored TEXT,
    correlation_double_counted INTEGER DEFAULT 0,
    stale_information INTEGER DEFAULT 0,
    regime_misread INTEGER DEFAULT 0,
    red_team_caught INTEGER DEFAULT 0,
    red_team_ignored INTEGER DEFAULT 0,
    genuinely_unpredictable INTEGER DEFAULT 0,
    taxonomy TEXT,
    lesson TEXT,
    memory_id TEXT REFERENCES memories(id)
);
CREATE INDEX IF NOT EXISTS idx_mistake_subject ON mistakes(subject);
CREATE INDEX IF NOT EXISTS idx_mistake_tax ON mistakes(taxonomy);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    kind TEXT,                        -- weights|routing|scoring|threshold|features
    role TEXT NOT NULL DEFAULT 'challenger',   -- champion|challenger
    status TEXT NOT NULL DEFAULT 'proposed',   -- proposed|running|evaluated|promoted|rejected
    config_patch TEXT,
    proposed_by TEXT,
    parent_id TEXT,
    promoted_at TEXT,
    rejected_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status, role);

CREATE TABLE IF NOT EXISTS experiment_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    created_at TEXT NOT NULL,
    dataset TEXT,
    n_samples INTEGER,
    n_trades INTEGER,
    mean_net_r REAL,
    hit_rate REAL,
    brier REAL,
    skill_score REAL,
    sharpe_annual REAL,
    max_drawdown_r REAL,
    deflated_sharpe REAL,
    pbo REAL,
    gates_passed TEXT,
    gates_failed TEXT,
    verdict TEXT,
    raw TEXT
);
CREATE INDEX IF NOT EXISTS idx_expres_exp ON experiment_results(experiment_id);

CREATE TABLE IF NOT EXISTS calibration_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    segment TEXT NOT NULL,            -- all|regime=..|event=..|horizon=..
    n INTEGER,
    mean_forecast REAL,
    observed_rate REAL,
    brier REAL,
    reliability REAL,
    resolution REAL,
    skill_score REAL,
    ece REAL,
    verdict TEXT,
    UNIQUE(created_at, segment)
);

CREATE TABLE IF NOT EXISTS after_action_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    recommendation_id TEXT REFERENCES recommendations(id),
    expected TEXT,
    actual TEXT,
    got_right TEXT,
    got_wrong TEXT,
    agents_helped TEXT,
    agents_hurt TEXT,
    evidence_mattered TEXT,
    evidence_noise TEXT,
    red_team_performance TEXT,
    calibration_error REAL,
    lesson TEXT,
    became_memory INTEGER DEFAULT 0,
    triggered_experiment INTEGER DEFAULT 0
);
""")


# --------------------------------------------------------------------------
# 004 — observability
# --------------------------------------------------------------------------

M004 = Migration(4, "observability", """
CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    investigation_id TEXT,
    agent TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    latency_ms INTEGER,
    retries INTEGER DEFAULT 0,
    error TEXT,
    tool_calls INTEGER DEFAULT 0,
    providers TEXT,
    llm_model TEXT,
    llm_input_tokens INTEGER DEFAULT 0,
    llm_output_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    evidence_count INTEGER DEFAULT 0,
    signal_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_arun_run ON agent_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_arun_agent ON agent_runs(agent, started_at);

CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    level TEXT NOT NULL,
    component TEXT NOT NULL,
    event TEXT NOT NULL,
    run_id INTEGER,
    investigation_id TEXT,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_sysev_time ON system_events(created_at);
CREATE INDEX IF NOT EXISTS idx_sysev_comp ON system_events(component, level);
""")


# 005 — publication lifecycle and lineage.
#
# The `recommendations` table from 002 stored a recommendation but had no way
# to say whether it was ever allowed out. Without a status column, "rejected"
# and "published" are the same row, and the audit trail cannot answer the one
# question it exists to answer. Lineage columns let a published call be traced
# back to the candidate, evidence graph and review decision that produced it.
M005 = Migration(5, "publication_lifecycle", """
ALTER TABLE recommendations ADD COLUMN status TEXT NOT NULL DEFAULT 'APPROVED';
ALTER TABLE recommendations ADD COLUMN candidate_id TEXT;
ALTER TABLE recommendations ADD COLUMN evidence_graph_id TEXT;
ALTER TABLE recommendations ADD COLUMN review_decision_id TEXT;
ALTER TABLE recommendations ADD COLUMN revision_parent_id TEXT;
ALTER TABLE recommendations ADD COLUMN rejection_reason TEXT;
ALTER TABLE recommendations ADD COLUMN review_audit TEXT;
ALTER TABLE recommendations ADD COLUMN orchestration_mode TEXT;
ALTER TABLE recommendations ADD COLUMN source_kind TEXT;
CREATE INDEX IF NOT EXISTS idx_rec_status ON recommendations(status, run_date);

-- One row per run recording which architecture actually executed. This is the
-- table that makes a silent fallback to legacy behaviour impossible to hide.
CREATE TABLE IF NOT EXISTS run_control_path (
    run_id INTEGER PRIMARY KEY,
    run_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    orchestration_mode TEXT NOT NULL,
    event_count INTEGER DEFAULT 0,
    investigation_count INTEGER DEFAULT 0,
    agents_available INTEGER DEFAULT 0,
    agents_selected INTEGER DEFAULT 0,
    agents_executed INTEGER DEFAULT 0,
    agents_skipped INTEGER DEFAULT 0,
    agent_execution_reason TEXT,
    review_iterations INTEGER DEFAULT 0,
    recommendations_candidate INTEGER DEFAULT 0,
    recommendations_approved INTEGER DEFAULT 0,
    recommendations_modified INTEGER DEFAULT 0,
    recommendations_rejected INTEGER DEFAULT 0,
    recommendations_published INTEGER DEFAULT 0,
    legacy_fallback_used INTEGER DEFAULT 0,
    degradation_level TEXT,
    learning_context_loaded INTEGER DEFAULT 0,
    memories_loaded INTEGER DEFAULT 0,
    estimated_cost_units REAL DEFAULT 0,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_rcp_date ON run_control_path(run_date);
""")


# 006 — review rounds, evidence versions and the event context.
#
# 2.0.1 could tell you a candidate was reviewed. It could not tell you how many
# times, against which evidence state, or whether the reviewer actually ran —
# and it recorded no event type, so contextual learning had to guess one from
# the setup string. All three are now stored explicitly.
M006 = Migration(6, "review_rounds_and_event_context", """
CREATE TABLE IF NOT EXISTS review_rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    run_id INTEGER,
    candidate_id TEXT NOT NULL,
    recommendation_id TEXT,
    subject TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    graph_version INTEGER NOT NULL DEFAULT 1,
    evidence_nodes INTEGER DEFAULT 0,
    effective_independent REAL,
    red_team_execution_status TEXT NOT NULL,
    n_findings INTEGER DEFAULT 0,
    findings TEXT,
    requested_research TEXT,
    followup_agents TEXT,
    decision TEXT,
    confidence_before INTEGER,
    confidence_after INTEGER
);
CREATE INDEX IF NOT EXISTS idx_rr_candidate ON review_rounds(candidate_id, round_number);
CREATE INDEX IF NOT EXISTS idx_rr_run ON review_rounds(run_id);

ALTER TABLE recommendations ADD COLUMN graph_version INTEGER DEFAULT 1;
ALTER TABLE recommendations ADD COLUMN review_execution_status TEXT DEFAULT 'completed';
ALTER TABLE recommendations ADD COLUMN review_round_count INTEGER DEFAULT 1;
ALTER TABLE recommendations ADD COLUMN event_type TEXT DEFAULT 'any';
ALTER TABLE recommendations ADD COLUMN sector TEXT DEFAULT 'any';
ALTER TABLE recommendations ADD COLUMN horizon_context TEXT DEFAULT 'intraday';
CREATE INDEX IF NOT EXISTS idx_rec_event ON recommendations(event_type, run_date);

ALTER TABLE run_control_path ADD COLUMN review_rounds INTEGER DEFAULT 0;
ALTER TABLE run_control_path ADD COLUMN red_team_attempts INTEGER DEFAULT 0;
ALTER TABLE run_control_path ADD COLUMN red_team_successes INTEGER DEFAULT 0;
ALTER TABLE run_control_path ADD COLUMN red_team_failures INTEGER DEFAULT 0;
ALTER TABLE run_control_path ADD COLUMN review_incomplete INTEGER DEFAULT 0;
ALTER TABLE run_control_path ADD COLUMN followup_requests INTEGER DEFAULT 0;
ALTER TABLE run_control_path ADD COLUMN followup_agents_selected INTEGER DEFAULT 0;
ALTER TABLE run_control_path ADD COLUMN followup_agents_executed INTEGER DEFAULT 0;
ALTER TABLE run_control_path ADD COLUMN followup_agents_failed INTEGER DEFAULT 0;
ALTER TABLE run_control_path ADD COLUMN evidence_nodes_before_followup INTEGER DEFAULT 0;
ALTER TABLE run_control_path ADD COLUMN evidence_nodes_after_followup INTEGER DEFAULT 0;
ALTER TABLE run_control_path ADD COLUMN graph_versions INTEGER DEFAULT 1;
ALTER TABLE run_control_path ADD COLUMN event_context_resolved TEXT DEFAULT 'any';
ALTER TABLE run_control_path ADD COLUMN contextual_weights_used INTEGER DEFAULT 0;
""")


MIGRATIONS: list[Migration] = [M001, M002, M003, M004, M005, M006]
LATEST_VERSION = max(m.version for m in MIGRATIONS)


def current_version(conn: sqlite3.Connection) -> int:
    conn.executescript(SCHEMA_VERSION_TABLE)
    row = conn.execute("SELECT MAX(version) v FROM schema_migrations").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def applied(conn: sqlite3.Connection) -> list[dict]:
    conn.executescript(SCHEMA_VERSION_TABLE)
    return [
        {"version": r[0], "name": r[1], "applied_at": r[2]}
        for r in conn.execute(
            "SELECT version, name, applied_at FROM schema_migrations ORDER BY version"
        )
    ]


def migrate(conn: sqlite3.Connection, target: int | None = None) -> list[str]:
    """Apply pending migrations in order. Returns the names applied.

    Idempotent: running twice applies nothing the second time. A database whose
    version exceeds this code's `LATEST_VERSION` is left untouched — silently
    downgrading a schema loses data.
    """
    target = LATEST_VERSION if target is None else target
    have = current_version(conn)

    if have > LATEST_VERSION:
        log.warning(
            "database schema v%d is newer than this code (v%d); leaving it alone",
            have, LATEST_VERSION,
        )
        return []

    done: list[str] = []
    for m in sorted(MIGRATIONS, key=lambda x: x.version):
        if m.version <= have or m.version > target:
            continue
        log.info("applying migration %03d %s", m.version, m.name)
        try:
            conn.executescript(m.sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?,?,?)",
                (m.version, m.name, dt.datetime.now(dt.timezone.utc).isoformat()),
            )
            conn.commit()
            done.append(m.name)
        except sqlite3.Error as exc:
            conn.rollback()
            raise RuntimeError(f"migration {m.version} ({m.name}) failed: {exc}") from exc
    return done


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    if not table_exists(conn, table):
        return False
    return any(r[1] == column for r in conn.execute(f"PRAGMA table_info({table})"))
