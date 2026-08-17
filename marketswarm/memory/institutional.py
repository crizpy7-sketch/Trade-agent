"""Institutional memory.

1.x had one flat `lessons` table. That is a notebook, not a memory: nothing
expires, nothing contradicts anything, and there is no way to ask "what do we
know about NVDA specifically?" or "what did we get wrong last time conditions
looked like this?".

Seven categories, each with a different retention policy, because a lesson
about a company ages differently from a lesson about an agent:

    episodic     what happened on a specific day
    semantic     recurring patterns held across many observations
    company      what we have learned about one issuer
    agent        which specialists perform when
    strategy     which hypotheses have already been tested
    failure      mistakes, with a taxonomy (see MistakeMemory)
    experiment   what was tried and what came of it

Memory hygiene is enforced rather than hoped for: entries carry a staleness
horizon, superseded entries are versioned rather than overwritten, and
contradictions are recorded instead of silently resolved.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger("marketswarm.memory.institutional")


class MemoryCategory(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    COMPANY = "company"
    AGENT = "agent"
    STRATEGY = "strategy"
    FAILURE = "failure"
    EXPERIMENT = "experiment"


# How long each kind of memory stays trustworthy without refresh.
DEFAULT_TTL_DAYS: dict[MemoryCategory, int | None] = {
    MemoryCategory.EPISODIC: 365,
    MemoryCategory.SEMANTIC: None,      # patterns persist until contradicted
    MemoryCategory.COMPANY: 180,        # companies change
    MemoryCategory.AGENT: 120,          # performance drifts with regime
    MemoryCategory.STRATEGY: None,
    MemoryCategory.FAILURE: None,       # never forget a mistake
    MemoryCategory.EXPERIMENT: None,
}


class MistakeTaxonomy(str, Enum):
    """Why a call went wrong. The point is to make failure *countable* — a
    taxonomy lets you notice you have made the same error thirty times."""

    OVERWEIGHTED_CORRELATED = "overweighted_correlated_evidence"
    IGNORED_CONTRADICTION = "ignored_contradicting_evidence"
    STALE_INFORMATION = "acted_on_stale_information"
    REGIME_MISREAD = "misread_the_regime"
    RED_TEAM_IGNORED = "red_team_objection_ignored"
    OVERCONFIDENT = "confidence_exceeded_evidence"
    MISSING_CATALYST = "failed_to_find_the_catalyst"
    BAD_LEVELS = "poor_entry_or_stop_placement"
    COST_UNDERESTIMATED = "underestimated_trading_costs"
    UNPREDICTABLE = "genuinely_unpredictable_event"
    DATA_ERROR = "bad_or_missing_data"


@dataclass
class Memory:
    category: MemoryCategory
    title: str
    body: str
    subject: str | None = None
    confidence: float = 0.5
    evidence_n: int = 0
    effect_size: float | None = None
    source: str = ""
    provenance: str = ""
    related_entities: list[str] = field(default_factory=list)
    valid_until: str | None = None
    version: int = 1
    supersedes: str | None = None
    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    created_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )

    def is_stale(self, now: dt.datetime | None = None) -> bool:
        if not self.valid_until:
            return False
        now = now or dt.datetime.now(dt.timezone.utc)
        try:
            until = dt.datetime.fromisoformat(self.valid_until.replace("Z", "+00:00"))
        except ValueError:
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=dt.timezone.utc)
        return now > until

    def to_dict(self) -> dict:
        return {
            "id": self.id, "category": self.category.value, "title": self.title,
            "body": self.body, "subject": self.subject, "confidence": self.confidence,
            "evidence_n": self.evidence_n, "effect_size": self.effect_size,
            "created_at": self.created_at, "valid_until": self.valid_until,
            "version": self.version, "related_entities": self.related_entities,
        }


@dataclass
class Mistake:
    subject: str
    predicted: str
    actual: str
    taxonomy: list[MistakeTaxonomy]
    lesson: str
    error_r: float | None = None
    recommendation_id: str | None = None
    run_date: str | None = None
    overweighted: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    correlation_double_counted: bool = False
    stale_information: bool = False
    regime_misread: bool = False
    red_team_caught: bool = False
    red_team_ignored: bool = False
    genuinely_unpredictable: bool = False
    id: str = field(default_factory=lambda: f"mis_{uuid.uuid4().hex[:12]}")
    created_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )

    @property
    def was_avoidable(self) -> bool:
        """Was this a process failure rather than bad luck?

        The distinction matters: only avoidable mistakes should change future
        behaviour. Treating an unforecastable shock as a lesson teaches the
        system to fear the wrong things.
        """
        return not self.genuinely_unpredictable

    def to_dict(self) -> dict:
        return {
            "id": self.id, "subject": self.subject, "predicted": self.predicted,
            "actual": self.actual, "taxonomy": [t.value for t in self.taxonomy],
            "lesson": self.lesson, "error_r": self.error_r,
            "avoidable": self.was_avoidable, "created_at": self.created_at,
            "red_team_caught": self.red_team_caught,
            "red_team_ignored": self.red_team_ignored,
        }


class InstitutionalMemory:
    """Reads and writes the `memories` and `mistakes` tables (migration 003)."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---------- write ----------

    def remember(self, memory: Memory, ttl_days: int | None = -1) -> str:
        """Store a memory. `ttl_days=-1` uses the category default; None means
        no expiry; a number sets an explicit horizon."""
        if ttl_days == -1:
            ttl_days = DEFAULT_TTL_DAYS.get(memory.category)
        if ttl_days is not None and not memory.valid_until:
            memory.valid_until = (
                dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=ttl_days)
            ).isoformat()

        self.conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, created_at, category, subject, title, body, confidence, evidence_n,
                effect_size, source, provenance, related_entities, version, supersedes,
                valid_until, active)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
            (memory.id, memory.created_at, memory.category.value, memory.subject,
             memory.title, memory.body, memory.confidence, memory.evidence_n,
             memory.effect_size, memory.source, memory.provenance,
             json.dumps(memory.related_entities), memory.version, memory.supersedes,
             memory.valid_until),
        )
        if memory.supersedes:
            self.conn.execute("UPDATE memories SET active=0 WHERE id=?",
                              (memory.supersedes,))
        self.conn.commit()
        return memory.id

    def supersede(self, old_id: str, new: Memory) -> str:
        """Replace a memory with a newer version, keeping the old one readable.

        Overwriting would destroy the record of what we used to believe, which
        is exactly what an after-action review needs.
        """
        row = self.conn.execute("SELECT version FROM memories WHERE id=?",
                                (old_id,)).fetchone()
        new.version = (int(row[0]) + 1) if row else 1
        new.supersedes = old_id
        return self.remember(new)

    def record_contradiction(self, memory_id: str, contradicted_by: str) -> None:
        """Flag rather than delete. A contradicted memory is information."""
        self.conn.execute(
            "UPDATE memories SET contradicted_by=?, confidence=MAX(0.05, confidence*0.5) "
            "WHERE id=?", (contradicted_by, memory_id))
        self.conn.commit()

    def record_mistake(self, mistake: Mistake, create_memory: bool = True) -> str:
        mem_id = None
        if create_memory and mistake.was_avoidable:
            mem = Memory(
                category=MemoryCategory.FAILURE,
                title=f"{mistake.subject}: {mistake.taxonomy[0].value if mistake.taxonomy else 'error'}",
                body=mistake.lesson,
                subject=mistake.subject,
                confidence=0.7,
                evidence_n=1,
                effect_size=mistake.error_r,
                source="after_action_review",
                related_entities=[mistake.subject],
            )
            mem_id = self.remember(mem, ttl_days=None)

        self.conn.execute(
            """INSERT INTO mistakes
               (id, created_at, recommendation_id, run_date, subject, predicted, actual,
                error_r, overweighted, ignored, correlation_double_counted,
                stale_information, regime_misread, red_team_caught, red_team_ignored,
                genuinely_unpredictable, taxonomy, lesson, memory_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (mistake.id, mistake.created_at, mistake.recommendation_id, mistake.run_date,
             mistake.subject, mistake.predicted, mistake.actual, mistake.error_r,
             json.dumps(mistake.overweighted), json.dumps(mistake.ignored),
             int(mistake.correlation_double_counted), int(mistake.stale_information),
             int(mistake.regime_misread), int(mistake.red_team_caught),
             int(mistake.red_team_ignored), int(mistake.genuinely_unpredictable),
             json.dumps([t.value for t in mistake.taxonomy]), mistake.lesson, mem_id),
        )
        self.conn.commit()
        return mistake.id

    # ---------- read ----------

    def recall(self, category: MemoryCategory | None = None, subject: str | None = None,
               include_stale: bool = False, limit: int = 20) -> list[Memory]:
        q = "SELECT * FROM memories WHERE active=1"
        params: list = []
        if category:
            q += " AND category=?"
            params.append(category.value)
        if subject:
            q += " AND (subject=? OR related_entities LIKE ?)"
            params.extend([subject.upper(), f'%"{subject.upper()}"%'])
        q += " ORDER BY confidence DESC, evidence_n DESC LIMIT ?"
        params.append(limit * 3 if not include_stale else limit)

        out: list[Memory] = []
        for r in self.conn.execute(q, params):
            m = self._row_to_memory(r)
            if not include_stale and m.is_stale():
                continue
            out.append(m)
            if len(out) >= limit:
                break

        if out:
            ids = [m.id for m in out]
            self.conn.executemany(
                "UPDATE memories SET access_count=access_count+1, last_accessed=? WHERE id=?",
                [(dt.datetime.now(dt.timezone.utc).isoformat(), i) for i in ids])
            self.conn.commit()
        return out

    def relevant_to(self, subject: str, regime: str | None = None,
                    event_types: list[str] | None = None, limit: int = 8) -> list[Memory]:
        """What should the investigator know before starting?

        Company memories first, then regime-conditional lessons, then general
        failures — narrowest and most specific first, because that is the order
        in which they are useful.
        """
        found: list[Memory] = []
        seen: set[str] = set()

        for m in self.recall(MemoryCategory.COMPANY, subject=subject, limit=limit):
            if m.id not in seen:
                seen.add(m.id)
                found.append(m)

        for m in self.recall(MemoryCategory.FAILURE, subject=subject, limit=limit):
            if m.id not in seen:
                seen.add(m.id)
                found.append(m)

        if regime:
            for m in self.recall(MemoryCategory.SEMANTIC, limit=limit * 2):
                if m.id in seen:
                    continue
                hay = f"{m.title} {m.body}".lower()
                if regime.lower() in hay or any((e or "").lower() in hay
                                                for e in (event_types or [])):
                    seen.add(m.id)
                    found.append(m)

        return found[:limit]

    def mistakes_like(self, subject: str | None = None,
                      taxonomy: MistakeTaxonomy | None = None,
                      limit: int = 10) -> list[dict]:
        q = "SELECT * FROM mistakes WHERE 1=1"
        params: list = []
        if subject:
            q += " AND subject=?"
            params.append(subject.upper())
        if taxonomy:
            q += " AND taxonomy LIKE ?"
            params.append(f'%"{taxonomy.value}"%')
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(q, params)]

    def mistake_frequency(self) -> dict[str, int]:
        """Which errors recur? The most valuable single query in the system —
        a taxonomy that keeps appearing is a process defect, not bad luck."""
        counts: dict[str, int] = {}
        for r in self.conn.execute("SELECT taxonomy FROM mistakes"):
            try:
                for t in json.loads(r[0] or "[]"):
                    counts[t] = counts.get(t, 0) + 1
            except json.JSONDecodeError:
                continue
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    # ---------- hygiene ----------

    def prune(self, now: dt.datetime | None = None) -> dict:
        """Deactivate stale and never-used memories.

        Storing everything forever is not memory, it is a landfill — recall
        quality degrades as the store fills with things that were never useful.
        """
        now = now or dt.datetime.now(dt.timezone.utc)
        stale = self.conn.execute(
            "UPDATE memories SET active=0 WHERE active=1 AND valid_until IS NOT NULL "
            "AND valid_until < ?", (now.isoformat(),)).rowcount

        cutoff = (now - dt.timedelta(days=90)).isoformat()
        unused = self.conn.execute(
            "UPDATE memories SET active=0 WHERE active=1 AND access_count=0 "
            "AND evidence_n < 5 AND created_at < ? AND category NOT IN ('failure','experiment')",
            (cutoff,)).rowcount

        low_conf = self.conn.execute(
            "UPDATE memories SET active=0 WHERE active=1 AND confidence < 0.15 "
            "AND contradicted_by IS NOT NULL").rowcount

        self.conn.commit()
        return {"expired": stale, "unused": unused, "contradicted": low_conf}

    def stats(self) -> dict:
        by_cat = {
            r[0]: r[1] for r in self.conn.execute(
                "SELECT category, COUNT(*) FROM memories WHERE active=1 GROUP BY category")
        }
        total = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE active=1").fetchone()[0]
        mistakes = self.conn.execute("SELECT COUNT(*) FROM mistakes").fetchone()[0]
        return {
            "active_memories": total,
            "by_category": by_cat,
            "mistakes": mistakes,
            "mistake_frequency": self.mistake_frequency(),
        }

    @staticmethod
    def _row_to_memory(r: sqlite3.Row) -> Memory:
        try:
            related = json.loads(r["related_entities"] or "[]")
        except (json.JSONDecodeError, TypeError):
            related = []
        return Memory(
            category=MemoryCategory(r["category"]),
            title=r["title"], body=r["body"], subject=r["subject"],
            confidence=float(r["confidence"] or 0.5),
            evidence_n=int(r["evidence_n"] or 0),
            effect_size=r["effect_size"],
            source=r["source"] or "", provenance=r["provenance"] or "",
            related_entities=related,
            valid_until=r["valid_until"],
            version=int(r["version"] or 1),
            supersedes=r["supersedes"],
            id=r["id"], created_at=r["created_at"],
        )
