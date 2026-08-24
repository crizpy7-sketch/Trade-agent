"""The swarm orchestrator.

Responsibilities:
  1. refuse to run when the market is closed
  2. run a cheap broad scan, let the Chief Investigator plan from what it sees,
     and execute only the specialists that plan selects
  3. contain failures so a partial outage degrades the report instead of
     killing the run
  4. route every candidate through the review gate and publish nothing else

The execution order is the architecture, so it is worth stating plainly:

    SCAN_AGENTS        cheap, foundational, always run — this is what the
                       Event Brain looks at
        ↓
    EVENT BRAIN        deterministic anomaly detection
        ↓
    CHIEF INVESTIGATOR hypotheses, priority, budget
        ↓
    CAPABILITY REGISTRY selects specialists for the detected event mix
        ↓
    SELECTED SPECIALISTS  — the ones not selected are never instantiated
        ↓
    DECISION_AGENTS    fusion, risk, playbook, red team
        ↓
    PIPELINE 2         evidence graph → recommendation → review gate
        ↓
    PublicationSet     the only thing anything downstream may read

Before this was wired, the full 16-agent DAG ran first and the Chief planned
afterwards, which made its plan a description of what had already happened.
A planner that runs after execution is a commentator.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import clock
from .__init__ import __version__
from .agents import ALL_AGENTS, AgentReport, SwarmContext
from .agents.base import DEFAULT_AGENT_TIMEOUT
from .config import Config
from .observability import Observatory
from .memory import LearningEngine, MemoryStore, Prediction
from .publication import PublicationSet
from .resilience import BREAKERS
from .providers.base import DataClient
from .providers.earnings import EarningsData
from .providers.econ import EconData
from .providers.edgar import EdgarData
from .providers.market import MarketData
from .providers.news import NewsData
from .providers.options import OptionsData

log = logging.getLogger("marketswarm.orchestrator")


# The cheap broad scan. Foundational, dependency-free (or nearly), and enough
# for the Event Brain to detect gaps, volatility shifts, index moves and
# relative volume. Everything here is `always_run` in the capability registry.
SCAN_AGENTS = ("overnight_scan", "futures", "volatility_regime", "technicals")

# Synthesis. These consume whatever the scan and the selected specialists
# produced, and they always run — without them there is nothing to review.
# Sentiment owns the permissioned community intake. It runs before fusion on
# every session so configured Discord/TradingView/X evidence cannot disappear
# merely because today's deterministic event label chose a different route.
DECISION_AGENTS = ("sentiment", "cross_verify", "risk", "playbook", "red_team")

# Everything else is routed: run only when the Chief Investigator asks for it.
ORCHESTRATION_MODES = ("dynamic", "full", "legacy")


def scaled_timeout(declared: float, budget: float) -> float:
    """Resolve one agent's timeout against the operator's configured budget.

    `timeout_seconds` is the per-agent budget, and an agent's own `timeout` is a
    statement of relative need: a scraper that declares twice the standard
    should still get twice it when the budget moves. So the declared value is
    scaled by budget/standard rather than capped by it.

    The previous rule was `min(declared, budget * 2)`, which — with the agent
    default and the config default both at 45s — could only ever *lower* a
    timeout. Raising `timeout_seconds` to 120 did nothing at all, which is
    precisely the advice that was given and had to be retracted.
    """
    if budget <= 0:
        return declared
    return max(1.0, declared * (budget / DEFAULT_AGENT_TIMEOUT))


@dataclass
class ControlPathTrace:
    """What actually executed. Persisted every run so a silent fallback to
    legacy behaviour is impossible to hide."""

    orchestration_mode: str = "dynamic"
    event_count: int = 0
    investigation_count: int = 0
    agents_available: int = 0
    agents_selected: int = 0
    agents_executed: int = 0
    agents_skipped: int = 0
    agent_execution_reason: dict = field(default_factory=dict)
    review_iterations: int = 0
    recommendations_candidate: int = 0
    recommendations_approved: int = 0
    recommendations_modified: int = 0
    recommendations_rejected: int = 0
    recommendations_published: int = 0
    legacy_fallback_used: bool = False
    degradation_level: str = "none"
    learning_context_loaded: int = 0
    memories_loaded: int = 0
    estimated_cost_units: float = 0.0
    notes: str = ""

    # --- review and follow-up. These exist so a stale review loop is visible
    # --- in the trace rather than having to be inferred from behaviour.
    review_rounds: int = 0
    red_team_attempts: int = 0
    red_team_successes: int = 0
    red_team_failures: int = 0
    review_incomplete: bool = False
    followup_requests: int = 0
    followup_agents_selected: int = 0
    followup_agents_executed: int = 0
    followup_agents_failed: int = 0
    evidence_nodes_before_followup: int = 0
    evidence_nodes_after_followup: int = 0
    graph_versions: int = 1

    # --- event-aware routing
    event_context_resolved: str = "any"
    contextual_weights_used: int = 0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["legacy_fallback_used"] = int(self.legacy_fallback_used)
        return d

    def render(self) -> str:
        ran = sorted(k for k, v in self.agent_execution_reason.items()
                     if not v.startswith("skipped"))
        skipped = sorted(k for k, v in self.agent_execution_reason.items()
                         if v.startswith("skipped"))
        return (
            f"mode={self.orchestration_mode} events={self.event_count} "
            f"investigations={self.investigation_count}\n"
            f"  executed ({len(ran)}): {', '.join(ran) or 'none'}\n"
            f"  skipped  ({len(skipped)}): {', '.join(skipped) or 'none'}\n"
            f"  candidates={self.recommendations_candidate} "
            f"approved={self.recommendations_approved} "
            f"rejected={self.recommendations_rejected} "
            f"published={self.recommendations_published}\n"
            f"  legacy_fallback={self.legacy_fallback_used} "
            f"degradation={self.degradation_level}"
        )


@dataclass
class SwarmResult:
    run_date: dt.date
    market_open: bool
    closed_reason: str | None = None
    reports: dict[str, AgentReport] = field(default_factory=dict)
    probability: float = 0.5
    confidence: int = 0
    run_id: int | None = None
    report_path: Path | None = None
    started_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    finished_at: dt.datetime | None = None
    narrative: str | None = None
    v2: object | None = None          # Pipeline2Result when the 2.0 layers ran
    publication: PublicationSet = field(default_factory=PublicationSet)
    trace: ControlPathTrace = field(default_factory=ControlPathTrace)

    @property
    def ideas(self) -> dict:
        """The legacy `{calls, puts, stocks}` view.

        Read-only and derived: it is computed from the approved recommendations
        every time it is asked for, so there is no way to assign a pre-review
        idea into it. That assignment is exactly the bug this property exists
        to make unrepresentable.
        """
        return self.publication.ideas_view()

    @property
    def recommendations(self) -> list:
        """Every active recommendation, including the non-directional ones a
        `{calls, puts, stocks}` view cannot express."""
        return self.publication.active()

    @property
    def screened_ideas(self) -> dict:
        """Three call and three put screening slots, safety-labelled.

        Unlike :attr:`ideas`, this may include rejected audit candidates and
        explicit empty slots. It is display-only and is never persisted as a
        prediction, monitored, scored, or treated as an active recommendation.
        """
        playbook = self.reports.get("playbook")
        data = playbook.data if playbook and playbook.usable else {}
        return self.publication.screening_view(data, slots=3)

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or dt.datetime.now(dt.timezone.utc)
        return (end - self.started_at).total_seconds()

    @property
    def agents_ok(self) -> int:
        return sum(1 for r in self.reports.values() if r.usable)

    @property
    def agents_failed(self) -> int:
        return sum(1 for r in self.reports.values() if not r.usable)


def stage_agents(agent_classes) -> list[list]:
    """Topologically stage agents so each stage can run fully concurrently."""
    remaining = {a.name: a for a in agent_classes}
    done: set[str] = set()
    stages: list[list] = []

    while remaining:
        ready = [
            a for a in remaining.values()
            if all(dep in done or dep not in {x.name for x in agent_classes} for dep in a.depends_on)
        ]
        if not ready:
            # Dependency cycle or a missing agent: run whatever is left rather
            # than deadlocking. Agents already tolerate absent inputs.
            log.warning("dependency cycle among %s — running as one stage", list(remaining))
            ready = list(remaining.values())
        stages.append(ready)
        for a in ready:
            done.add(a.name)
            remaining.pop(a.name, None)
    return stages


class Swarm:
    def __init__(self, config: Config, store: MemoryStore | None = None):
        self.config = config
        self.store = store or MemoryStore(config.db_path)
        self._contextual_count = 0
        self.learning = LearningEngine(self.store)

        # 2.0 schema. Additive and idempotent — 1.x data is preserved.
        try:
            from .memory.migrations import migrate
            applied = migrate(self.store.conn)
            if applied:
                log.info("applied %d schema migration(s): %s", len(applied),
                         ", ".join(applied))
        except Exception as exc:  # noqa: BLE001 — never block a run on migrations
            log.error("schema migration failed, continuing on the existing schema: %s", exc)

    async def run(self, run_date: dt.date | None = None, force: bool = False,
                  mode: str | None = None) -> SwarmResult:
        run_date = run_date or clock.now_et().date()
        status = clock.day_status(run_date)

        if not status.is_trading_day and not force:
            log.info("market closed on %s (%s)", run_date, status.reason)
            return SwarmResult(run_date=run_date, market_open=False, closed_reason=status.reason,
                               finished_at=dt.datetime.now(dt.timezone.utc))

        mode = (mode or self.config.orchestration_mode or "dynamic").lower()
        if mode not in ORCHESTRATION_MODES:
            log.warning("unknown orchestration mode %r; using dynamic", mode)
            mode = "dynamic"

        result = SwarmResult(run_date=run_date, market_open=True)
        result.trace.orchestration_mode = mode
        result.run_id = self.store.start_run(run_date.isoformat(), "premarket")

        # Load everything the agent learned from previous sessions. The
        # resolver is the single API for that — see memory/resolver.py.
        self.config.calibrator = self.learning.load_calibrator()
        self.config.agent_weights = self._resolve_agent_weights()
        self.config.lessons = self.store.active_lessons()
        result.trace.learning_context_loaded = len(self.config.agent_weights)
        result.trace.memories_loaded = len(self.config.lessons)
        log.info("mode=%s · calibrator (%s) · %d agent weights · %d lessons",
                 mode, self.config.calibrator.diagnosis,
                 len(self.config.agent_weights), len(self.config.lessons))

        async with DataClient(
            cache_dir=self.config.cache_dir,
            cache_ttl=self.config.cache_ttl_seconds,
            user_agent=self.config.user_agent,
            max_retries=self.config.max_retries,
            rate_per_second=self.config.rate_per_second,
        ) as client:
            ctx = SwarmContext(
                run_date=run_date,
                prev_session=clock.previous_trading_day(run_date),
                universe=self.config.universe,
                index_symbols=self.config.index_symbols,
                market=MarketData(client),
                options=OptionsData(client),
                news=NewsData(client, universe=set(self.config.universe)),
                econ=EconData(client, self.config.fred_api_key),
                edgar=EdgarData(client, self.config.user_agent),
                earnings=EarningsData(client),
                config=self.config,
                agent_weights=self.config.agent_weights,
                lessons=self.config.lessons,
            )

            plans = await self._execute_swarm(ctx, result, mode)
            result.reports = ctx.reports

            # ---- MarketSwarm 2.0 decision layer ----
            # This is the only thing allowed to authorise publication. If it
            # fails, nothing is published — the pre-review playbook does not
            # stand in for a review that did not happen.
            await self._decide(ctx, result, run_date, plans, mode)

        cv = ctx.reports.get("cross_verify")
        if cv and cv.usable:
            result.probability = cv.data.get("probability", 0.5)
            result.confidence = cv.data.get("confidence_score", 0)

        self._persist_recommendations(result)
        self._persist_predictions(result, ctx)
        self._persist_evidence(result, ctx)
        self._persist_trace(result)
        result.finished_at = dt.datetime.now(dt.timezone.utc)
        return result

    def _resolve_agent_weights(self, regime: str | None = None,
                               event_type: str = "any") -> dict[str, float]:
        """Weights for this context, from contextual learning where it exists.

        Called twice per run, deliberately:

          stage 1  before the scan, with the previous session's regime and no
                   event type — enough to weight the foundational agents
          stage 2  after the Event Brain has classified today, with the real
                   (regime, event_type) — this is the one that routes

        Resolving only once, before detection, was the bug: the store can hold
        "options_flow is worth 1.6 around earnings" and the run could never ask
        for it, because at resolution time it did not yet know it was an
        earnings day.

        Backoff and shrinkage live in `ContextualWeightResolver` and
        `ContributionTracker`; a thin context simply falls through to a broader
        one rather than inventing a number.
        """
        baseline = self.learning.agent_weights()
        self._contextual_count = 0
        try:
            from .memory.resolver import ContextualWeightResolver

            resolver = ContextualWeightResolver(self.store.conn, baseline)
            names = [a.name for a in ALL_AGENTS]
            weights = resolver.routing_weights(
                names,
                regime=regime if regime is not None else self._last_regime(),
                event_type=event_type)
            summary = resolver.summary()
            self._contextual_count = summary.get("from_contextual_learning", 0)
            log.info("weight provenance: %s", summary)
            return weights
        except Exception as exc:  # noqa: BLE001 — never block a run on learning
            log.warning("contextual weight resolution failed, using baseline: %s", exc)
            return baseline

    def _last_regime(self) -> str:
        try:
            row = self.store.conn.execute(
                "SELECT regime FROM runs WHERE regime != '' "
                "ORDER BY id DESC LIMIT 1").fetchone()
            return str(row[0]) if row else ""
        except sqlite3.Error:
            return ""

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------

    async def _run_agents(self, ctx: SwarmContext, names: list[str],
                          result: SwarmResult, reason: str) -> None:
        """Instantiate and execute exactly the named agents, in dependency order.

        Agents not named here are never constructed. That is what makes the
        routing real rather than advisory — a test can assert on which classes
        were instantiated, not merely on which ones a plan mentioned.
        """
        wanted = [a for a in ALL_AGENTS if a.name in set(names)]
        if not wanted:
            return

        # An agent whose provider has failed repeatedly is short-circuited
        # rather than retried every stage. Without this the breaker existed
        # but nothing consulted it, and a dead provider cost the full timeout
        # on every run.
        open_now = set(BREAKERS.open_circuits())
        blocked = [a for a in wanted if a.name in open_now]
        for a in blocked:
            result.trace.agent_execution_reason[a.name] = (
                "skipped: circuit breaker open after repeated failures")
            log.warning("  %s skipped — circuit breaker open", a.name)
        wanted = [a for a in wanted if a.name not in open_now]
        if not wanted:
            return

        for stage_no, stage in enumerate(stage_agents(wanted), 1):
            instances = [a() for a in stage]
            for inst in instances:
                inst.timeout = scaled_timeout(inst.timeout, self.config.timeout_seconds)
            log.info("  stage %d (%s): %s", stage_no, reason,
                     ", ".join(i.name for i in instances))
            reports = await asyncio.gather(*(i.execute(ctx) for i in instances))
            for r in reports:
                ctx.reports[r.agent] = r
                result.trace.agents_executed += 1
                # An agent recruited later by a follow-up was marked skipped
                # when the main pass finished. Overwrite that, and give the
                # skip counter back, or the trace would report an agent as
                # skipped in the same run it demonstrably executed.
                previous = result.trace.agent_execution_reason.get(r.agent, "")
                if previous.startswith("skipped"):
                    result.trace.agents_skipped = max(
                        0, result.trace.agents_skipped - 1)
                result.trace.agent_execution_reason[r.agent] = reason

                # Feed the breaker so a persistently failing agent stops being
                # attempted. `usable` is the agent's own verdict on whether it
                # produced anything worth having.
                breaker = BREAKERS.get(r.agent)
                if r.usable:
                    breaker.record_success()
                else:
                    breaker.record_failure()

                log.info("    %s: %s (%s, %dms)", r.agent, r.headline, r.status,
                         r.duration_ms)

    async def _rerun_agents(self, ctx: SwarmContext, names: list[str],
                            result: SwarmResult, reason: str) -> None:
        """Re-execute agents that have already run this session.

        `_run_agents` is happy to re-run anything it is given, but the callers
        that build agent lists filter out names already in `ctx.reports`. This
        exists so the adversarial refresh after follow-up research is explicit:
        re-running the red team is deliberate, not an accident of bookkeeping.
        """
        if not names:
            return
        await self._run_agents(ctx, names, result, reason)

    async def _execute_swarm(self, ctx: SwarmContext, result: SwarmResult,
                             mode: str) -> list:
        """Run the swarm for this mode. Returns the investigation plans."""
        from .investigation.chief import ChiefInvestigator
        from .investigation.event_brain import EventBrain

        all_names = [a.name for a in ALL_AGENTS]
        result.trace.agents_available = len(all_names)

        if mode in ("full", "legacy"):
            await self._run_agents(ctx, all_names, result, f"{mode} mode: all agents")
            result.trace.agents_selected = len(all_names)
            result.trace.agents_skipped = 0
            if mode == "legacy":
                return []
            snapshot = _snapshot_from(ctx.reports)
            plans = ChiefInvestigator(brain=EventBrain()).plan_session(
                snapshot, agent_weights=self.config.agent_weights)
            result.trace.investigation_count = len(plans)
            result.trace.event_count = sum(len(p.events) for p in plans)
            return plans

        # ---------------- dynamic: the 2.0 default ----------------

        # 1. cheap broad scan
        await self._run_agents(ctx, list(SCAN_AGENTS), result, "scan: foundational")

        # 2. detect — before planning, and before routing weights are resolved
        snapshot = _snapshot_from(ctx.reports)
        brain = EventBrain()
        events = brain.scan(snapshot)
        ctx.detected_events = events

        # 3. resolve routing weights for *this* session's event context.
        #
        # Stage 1 ran before the scan and could only know the previous run's
        # regime, because today's event had not been detected yet — which made
        # every event-specific weight in the store unreachable at the moment it
        # was needed. This is stage 2: now that the Event Brain has spoken,
        # re-resolve against the real (regime, event_type) and route on that.
        regime = ctx.data_of("volatility_regime", "regime", "") or self._last_regime()
        event_type = _dominant_event_type(events)
        ctx.event_type = event_type
        ctx.regime = regime
        result.trace.event_context_resolved = f"regime={regime}|event={event_type}"

        routing_weights = self._resolve_agent_weights(regime=regime,
                                                      event_type=event_type)
        self.config.agent_weights = routing_weights
        ctx.agent_weights = routing_weights
        result.trace.contextual_weights_used = self._contextual_count
        log.info("event-aware routing context: regime=%s event=%s "
                 "(%d contextual weight(s))", regime or "unknown", event_type,
                 self._contextual_count)

        # 4. plan against that context
        chief = ChiefInvestigator(brain=brain)
        plans = chief.plan_session(snapshot, agent_weights=routing_weights)
        result.trace.investigation_count = len(plans)
        result.trace.event_count = sum(len(p.events) for p in plans)
        log.info("event brain + chief: %d event(s) → %d investigation(s)",
                 result.trace.event_count, len(plans))

        # 4. the union of what the plans selected, minus what already ran
        selected: set[str] = set()
        for plan in plans:
            selected.update(plan.all_agents)
            result.trace.estimated_cost_units += plan.budget.cost_used
        selected.update(DECISION_AGENTS)
        selected.update(SCAN_AGENTS)

        result.trace.agents_selected = len(selected)
        specialists = [n for n in all_names
                       if n in selected and n not in SCAN_AGENTS
                       and n not in DECISION_AGENTS]

        # 5. execute only those specialists
        if specialists:
            await self._run_agents(ctx, specialists, result,
                                   "selected by chief investigator")

        # 6. synthesis always runs — there must be something to review
        await self._run_agents(ctx, list(DECISION_AGENTS), result,
                               "decision: always runs")

        for name in all_names:
            if name not in ctx.reports:
                result.trace.agent_execution_reason[name] = (
                    "skipped: not selected for the detected event mix")
                result.trace.agents_skipped += 1
        log.info("dynamic routing: ran %d of %d agents, skipped %d",
                 result.trace.agents_executed, len(all_names),
                 result.trace.agents_skipped)
        return plans

    # ------------------------------------------------------------------
    # the decision layer
    # ------------------------------------------------------------------

    async def _decide(self, ctx: SwarmContext, result: SwarmResult,
                      run_date: dt.date, plans: list, mode: str) -> None:
        """Route candidates through review and set the authoritative output."""
        if mode == "legacy":
            result.trace.legacy_fallback_used = True
            if not self.config.allow_unreviewed_publication:
                result.publication = PublicationSet.suppressed_set(
                    "legacy mode runs the 1.x swarm with no review gate; "
                    "publication is disabled unless allow_unreviewed_publication "
                    "is explicitly set", mode="legacy")
                log.warning("legacy mode: analysis only, recommendations suppressed")
            else:
                result.publication = _unreviewed_publication(ctx.reports)
                log.warning("legacy mode with allow_unreviewed_publication: "
                            "publishing UNREVIEWED 1.x ideas")
            return

        try:
            from .pipeline2 import Pipeline2

            gap = None
            perf = self.store.performance_summary(90)
            for b in (perf.get("by_kind") or {}).values():
                gap = b.get("calibration_gap")
                break

            # With no run_id every agent_runs row is written NULL, and
            # agent_status() keys off MAX(run_id) — so the whole table was
            # invisible to the read API and `!agents` answered "no runs"
            # forever, on a database that had them.
            pipeline = Pipeline2(
                self.config, store=self.store,
                observatory=Observatory(conn=getattr(self.store, "conn", None),
                                        run_id=result.run_id),
            )
            # Pipeline2 is a synchronous state machine by design. Running it on
            # a worker thread lets its follow-up investigator schedule real
            # agent work back onto this event loop without either side having
            # to pretend to be the other.
            loop = asyncio.get_running_loop()
            result.v2 = await asyncio.to_thread(
                pipeline.run,
                ctx.reports, run_date,
                calibration_gap=gap,
                prior_failures=self._prior_failures(ctx),
                plans=plans,
                investigator=self._make_investigator(ctx, result, loop),
                mode=mode,
            )
            result.publication = result.v2.publication
            log.info("2.0 pipeline: %s", result.publication.summary())
        except Exception as exc:  # noqa: BLE001 — must not publish on failure
            log.exception("2.0 decision layer failed")
            result.v2 = None
            result.publication = PublicationSet.suppressed_set(
                f"the 2.0 review layer failed ({type(exc).__name__}); no "
                f"recommendation was reviewed, so none is published",
                mode=mode)

        pub = result.publication
        result.trace.recommendations_candidate = len(pub.approved) + len(pub.rejected)
        result.trace.recommendations_approved = len(pub.approved)
        result.trace.recommendations_rejected = len(pub.rejected)
        result.trace.recommendations_modified = sum(
            1 for r in pub.active() if r.revision_count > 0)
        result.trace.recommendations_published = len(pub.active())
        result.trace.review_iterations = pub.review_iterations
        result.trace.review_incomplete = bool(
            pub.suppressed and "review did not complete"
            in (pub.suppression_reason or ""))
        if result.v2 is not None:
            result.trace.degradation_level = str(
                (result.v2.degradation or {}).get("level", "none"))

            sessions = getattr(result.v2, "review_sessions", []) or []
            if sessions:
                result.trace.review_rounds = sum(len(s.rounds) for s in sessions)
                result.trace.graph_versions = max(s.graph_version for s in sessions)
                result.trace.evidence_nodes_before_followup = max(
                    s.nodes_before_followup for s in sessions)
                result.trace.evidence_nodes_after_followup = max(
                    s.nodes_after_followup for s in sessions)

    def _make_investigator(self, ctx: SwarmContext, result: SwarmResult,
                           loop: asyncio.AbstractEventLoop):
        """Build the bounded follow-up researcher the review gate can call.

        A REQUEST_MORE_RESEARCH verdict has to buy evidence to mean anything.
        This asks the Chief which agents answer the open questions, runs the
        ones that have not run yet, and hands the refreshed reports back.

        Bounded three ways: the follow-up plan's own budget, the review loop's
        iteration cap, and the rule that an agent already run this session is
        never re-run for the same session.
        """
        from .investigation.chief import ChiefInvestigator

        chief = ChiefInvestigator()
        known = {a.name for a in ALL_AGENTS}

        def investigate(symbol: str, current: dict, questions: list[str],
                        parent_plan) -> dict:
            if not questions:
                return current

            if parent_plan is not None:
                plan = chief.followup_plan(parent_plan, questions)
                agents = list(plan.all_agents) if plan is not None else []
            else:
                agents = chief.agents_for_questions(questions)

            fresh = [a for a in agents if a in known and a not in ctx.reports]
            current["open_questions"] = list(questions)
            if not fresh:
                reusable = [a for a in agents
                            if getattr(ctx.reports.get(a), "usable", False)]
                if reusable:
                    # Another candidate may already have requested the same
                    # evidence. Reuse it and let this candidate rebuild its
                    # graph; "not new" is not the same as "not available".
                    current["followup_agents"] = reusable
                    current["followup_note"] = (
                        "follow-up requested; reused evidence already obtained "
                        "this session from " + ", ".join(reusable))
                    current.pop("followup_failed", None)
                    current.pop("followup_error", None)
                    return current
                current["followup_note"] = (
                    "follow-up requested; every agent that could answer it had "
                    "already run this session")
                return current

            log.info("follow-up research for %s: running %s", symbol, ", ".join(fresh))
            result.trace.followup_requests += 1
            result.trace.followup_agents_selected += len(fresh)

            future = asyncio.run_coroutine_threadsafe(
                self._run_agents(ctx, fresh, result,
                                 "follow-up: requested by review gate"),
                loop)
            try:
                future.result(timeout=self.config.timeout_seconds * 3)
            except Exception as exc:  # noqa: BLE001 — follow-up is best-effort
                log.warning("follow-up research failed for %s: %s", symbol, exc)
                result.trace.followup_agents_failed += len(fresh)
                current["followup_note"] = f"follow-up research failed: {exc}"
                current["followup_failed"] = True
                current["followup_error"] = str(exc)[:200]
                return current

            produced = [a for a in fresh
                        if getattr(ctx.reports.get(a), "usable", False)]
            result.trace.followup_agents_executed += len(produced)
            result.trace.followup_agents_failed += len(fresh) - len(produced)

            if not produced:
                # Agents ran and every one of them came back empty. The
                # requested evidence does not exist, and saying otherwise would
                # turn an unanswered question into an answered one.
                current["followup_failed"] = True
                current["followup_error"] = (
                    f"{len(fresh)} agent(s) ran and none produced usable evidence")
                current["followup_agents"] = fresh
                return current

            # New evidence exists, so the adversary must be re-run against it.
            # Re-running fusion first means the independence count the red team
            # complains about actually reflects what arrived; re-running the
            # red team means round 2 argues with the new picture instead of
            # replaying round 1's objections.
            refresh = [a for a in ("cross_verify", "red_team") if a in known]
            try:
                asyncio.run_coroutine_threadsafe(
                    self._rerun_agents(ctx, refresh, result,
                                       "refresh: re-attacked after follow-up"),
                    loop).result(timeout=self.config.timeout_seconds * 3)
                result.trace.red_team_attempts += 1
                if getattr(ctx.reports.get("red_team"), "usable", False):
                    result.trace.red_team_successes += 1
                else:
                    result.trace.red_team_failures += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("adversarial refresh failed for %s: %s", symbol, exc)
                result.trace.red_team_failures += 1

            current["followup_agents"] = produced
            current.pop("followup_failed", None)
            return current

        return investigate

    def _prior_failures(self, ctx: SwarmContext) -> dict[str, list[str]]:
        """Retrieve what the system already learned about these symbols.

        This is where institutional memory enters the decision, rather than
        sitting in a table nothing reads.
        """
        out: dict[str, list[str]] = {}
        try:
            from .memory.institutional import InstitutionalMemory, MemoryCategory
            mem = InstitutionalMemory(self.store.conn)
            for sym in ctx.universe:
                relevant = mem.recall(category=MemoryCategory.FAILURE,
                                      subject=sym, limit=3)
                if relevant:
                    out[sym] = [m.title for m in relevant]
        except Exception as exc:  # noqa: BLE001 — memory must not block a run
            log.warning("institutional memory recall failed: %s", exc)
        return out

    def _persist_recommendations(self, result: SwarmResult) -> None:
        """Store approved and rejected candidates, distinguished by status.

        Both are kept — the rejected ones are the audit trail and the raw
        material the learning loop needs. They are stored as REJECTED, which
        is what stops the monitor, the scorer and the API from ever treating
        them as live ideas.
        """
        pub = result.publication
        conn = self.store.conn
        try:
            for rec in pub.active():
                status = ("MODIFIED" if rec.revision_count > 0
                          else "SUPPRESSED" if pub.suppressed else "APPROVED")
                conn.execute(
                    """INSERT OR REPLACE INTO recommendations
                       (id, investigation_id, run_id, created_at, run_date, subject,
                        rec_type, conviction, direction, forecast_probability,
                        confidence, original_confidence, expected_r, entry, target,
                        stop, horizon, regime, data_quality, observation,
                        interpretation, rationale, key_risks, invalidation,
                        change_our_mind, supporting_evidence, contradicting_evidence,
                        historical_analogues, agent_contributors, review_status,
                        revision_count, model_version, system_version, experiment_id,
                        resolved, status, candidate_id, evidence_graph_id,
                        review_decision_id, revision_parent_id, orchestration_mode,
                        source_kind, presentation_payload)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                               ?,?,?,?,?,?,?,0,?,?,?,?,?,?,?,?)""",
                    (rec.id, rec.investigation_id, result.run_id, rec.created_at,
                     result.run_date.isoformat(), rec.subject, rec.rec_type.value,
                     rec.conviction.value, rec.direction, rec.forecast_probability,
                     rec.confidence, rec.original_confidence, rec.expected_r,
                     rec.entry, rec.target, rec.stop, rec.horizon, rec.regime,
                     rec.data_quality, rec.observation, rec.interpretation,
                     rec.rationale, json.dumps(rec.key_risks),
                     json.dumps(rec.invalidation), json.dumps(rec.change_our_mind),
                     json.dumps(rec.supporting_evidence),
                     json.dumps(rec.contradicting_evidence),
                     json.dumps(rec.historical_analogues),
                     json.dumps(rec.agent_contributors), rec.review_status,
                     rec.revision_count, rec.model_version, rec.system_version,
                     rec.experiment_id, status, rec.candidate_id,
                     rec.evidence_graph_id, rec.review_decision_id,
                     rec.revision_parent_id, result.trace.orchestration_mode,
                     rec.source_kind,
                     json.dumps(rec.source_payload, default=str)[:200000]))

            self._persist_review_rounds(result)

            for rej in pub.rejected:
                conn.execute(
                    """INSERT OR REPLACE INTO recommendations
                       (id, run_id, created_at, run_date, subject, rec_type,
                        conviction, direction, forecast_probability, confidence,
                        original_confidence, expected_r, entry, target, stop,
                        resolved, status, candidate_id, rejection_reason,
                        review_audit, orchestration_mode, system_version,
                        source_kind, presentation_payload)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?,?,?,?)""",
                    (rej.candidate_id, result.run_id,
                     dt.datetime.now(dt.timezone.utc).isoformat(),
                     result.run_date.isoformat(), rej.subject, "REJECTED",
                     "REJECTED_BY_REVIEW",
                     rej.source_payload.get("direction"),
                     rej.source_payload.get("probability"),
                     0, rej.original_confidence,
                     rej.source_payload.get("expected_r"),
                     rej.source_payload.get("entry"),
                     rej.source_payload.get("target"),
                     rej.source_payload.get("stop"),
                     rej.status.value, rej.candidate_id, rej.reason,
                     json.dumps(rej.audit)[:200000],
                     result.trace.orchestration_mode,
                     __version__, rej.source_kind,
                     json.dumps(rej.source_payload, default=str)[:200000]))
            conn.commit()
        except sqlite3.Error as exc:
            log.error("failed to persist recommendations: %s", exc)

    def _persist_review_rounds(self, result: SwarmResult) -> None:
        """One row per adversarial round, oldest first.

        Round 1 is never overwritten by round 2. After-action review needs to
        see that an objection was raised, research was demanded, evidence
        arrived, and the objection was then withdrawn — a single final-state
        row cannot express any of that.
        """
        if result.v2 is None:
            return
        sessions = getattr(result.v2, "review_sessions", []) or []
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            for session in sessions:
                for rnd in session.rounds:
                    self.store.conn.execute(
                        """INSERT INTO review_rounds
                           (created_at, run_id, candidate_id, subject, round_number,
                            graph_version, evidence_nodes, effective_independent,
                            red_team_execution_status, n_findings, findings,
                            followup_agents)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (now, result.run_id, getattr(session, "candidate_id", ""),
                         session.symbol, rnd.get("round", 0),
                         rnd.get("graph_version", 1), rnd.get("evidence_nodes", 0),
                         rnd.get("effective_independent"),
                         rnd.get("execution_status", "completed"),
                         rnd.get("n_findings", 0),
                         json.dumps(rnd.get("objections", [])),
                         json.dumps(rnd.get("followup_agents", []))))
        except sqlite3.Error as exc:
            log.error("failed to persist review rounds: %s", exc)

    def _persist_trace(self, result: SwarmResult) -> None:
        """Record which architecture actually executed."""
        if result.run_id is None:
            return
        t = result.trace
        try:
            self.store.conn.execute(
                """INSERT OR REPLACE INTO run_control_path
                   (run_id, run_date, created_at, orchestration_mode, event_count,
                    investigation_count, agents_available, agents_selected,
                    agents_executed, agents_skipped, agent_execution_reason,
                    review_iterations, recommendations_candidate,
                    recommendations_approved, recommendations_modified,
                    recommendations_rejected, recommendations_published,
                    legacy_fallback_used, degradation_level,
                    learning_context_loaded, memories_loaded,
                    estimated_cost_units, notes,
                    review_rounds, red_team_attempts, red_team_successes,
                    red_team_failures, review_incomplete, followup_requests,
                    followup_agents_selected, followup_agents_executed,
                    followup_agents_failed, evidence_nodes_before_followup,
                    evidence_nodes_after_followup, graph_versions,
                    event_context_resolved, contextual_weights_used)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                           ?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (result.run_id, result.run_date.isoformat(),
                 dt.datetime.now(dt.timezone.utc).isoformat(),
                 t.orchestration_mode, t.event_count, t.investigation_count,
                 t.agents_available, t.agents_selected, t.agents_executed,
                 t.agents_skipped, json.dumps(t.agent_execution_reason),
                 t.review_iterations, t.recommendations_candidate,
                 t.recommendations_approved, t.recommendations_modified,
                 t.recommendations_rejected, t.recommendations_published,
                 int(t.legacy_fallback_used), t.degradation_level,
                 t.learning_context_loaded, t.memories_loaded,
                 t.estimated_cost_units,
                 result.publication.suppression_reason or "",
                 t.review_rounds, t.red_team_attempts, t.red_team_successes,
                 t.red_team_failures, int(t.review_incomplete),
                 t.followup_requests, t.followup_agents_selected,
                 t.followup_agents_executed, t.followup_agents_failed,
                 t.evidence_nodes_before_followup, t.evidence_nodes_after_followup,
                 t.graph_versions, t.event_context_resolved,
                 t.contextual_weights_used))
            self.store.conn.commit()
        except sqlite3.Error as exc:
            log.error("failed to persist the control-path trace: %s", exc)

    def _persist_predictions(self, result: SwarmResult, ctx: SwarmContext) -> None:
        """Write approved ideas to memory so they can be scored tonight.

        Reads `result.ideas`, which is derived from the approved
        recommendations. A rejected candidate has no route into this table,
        and therefore none into the monitor or the scorer.
        """
        regime = ctx.data_of("volatility_regime", "regime", "unknown")
        contributions = ctx.data_of("cross_verify", "contributions", {}) or {}
        ideas_view = result.ideas

        for kind, ideas in (("call_idea", ideas_view.get("calls", [])),
                            ("put_idea", ideas_view.get("puts", [])),
                            ("stock_setup", ideas_view.get("stocks", []))):
            for i in ideas:
                gap = (ctx.data_of("overnight_scan", "profiles", {}) or {}).get(i["symbol"], {}).get("gap_pct")
                pred = Prediction(
                    run_date=result.run_date.isoformat(),
                    kind=kind,
                    symbol=i["symbol"],
                    direction=i["direction"],
                    probability=i["probability"],
                    raw_probability=i.get("raw_probability"),
                    confidence=i.get("confidence"),
                    entry=i.get("entry"),
                    target=i.get("target"),
                    stop=i.get("stop"),
                    expected_r=i.get("expected_r"),
                    thesis=i.get("rationale", "")[:1000],
                    invalidation=i.get("invalidation", "")[:1000],
                    features={
                        "regime": regime,
                        "setup": kind,
                        # The event type the Event Brain actually classified,
                        # stored explicitly. Inferring it later from `setup`
                        # would learn about "stock_setup", which is a shape,
                        # not a situation — and the two produce different
                        # contextual buckets.
                        "event_type": ctx.event_type,
                        "horizon": "intraday",
                        "investigation_id": i.get("investigation_id"),
                        "gap_bucket": _gap_bucket(gap),
                        "session_day": result.run_date.strftime("%A"),
                        "strike": i.get("strike"),
                        "expiration": i.get("expiration"),
                        "liquidity": i.get("liquidity"),
                        "agents_executed": sorted(
                            k for k, v in result.trace.agent_execution_reason.items()
                            if not v.startswith("skipped")),
                        "agents_skipped": sorted(
                            k for k, v in result.trace.agent_execution_reason.items()
                            if v.startswith("skipped")),
                    },
                    contributing_agents=contributions,
                )
                self.store.record_prediction(pred, result.run_id)

        # A single index-level directional call, scored the same way — this is
        # the series that drives the global calibrator. It is a directional
        # prediction like any other, so a suppressed publication suppresses it
        # too; otherwise a failed review layer would still be quietly filing
        # a market call every morning.
        spy = (ctx.data_of("technicals", "setups", {}) or {}).get("SPY")
        if spy and result.publication.may_publish_index_call():
            self.store.record_prediction(
                Prediction(
                    run_date=result.run_date.isoformat(),
                    kind="direction",
                    symbol="SPY",
                    direction="long" if result.probability >= 0.5 else "short",
                    probability=max(result.probability, 1 - result.probability),
                    raw_probability=result.probability,
                    confidence=result.confidence,
                    entry=spy["price"],
                    target=spy["price"] * (1.004 if result.probability >= 0.5 else 0.996),
                    stop=spy["price"] * (0.996 if result.probability >= 0.5 else 1.004),
                    thesis=f"Swarm index read: P(up)={result.probability:.1%}",
                    invalidation="Scored on close vs open, not on a bracket",
                    features={"regime": regime, "setup": "index_direction",
                              "event_type": ctx.event_type, "horizon": "intraday",
                              "session_day": result.run_date.strftime("%A"),
                              "agents_executed": sorted(
                                  k for k, v in
                                  result.trace.agent_execution_reason.items()
                                  if not v.startswith("skipped")),
                              "agents_skipped": sorted(
                                  k for k, v in
                                  result.trace.agent_execution_reason.items()
                                  if v.startswith("skipped"))},
                    contributing_agents=contributions,
                ),
                result.run_id,
            )

    def _persist_evidence(self, result: SwarmResult, ctx: SwarmContext) -> None:
        items = []
        for r in ctx.reports.values():
            for e in r.evidence:
                d = e.to_dict()
                d["source_class"] = e.source
                items.append(d)
        if items:
            self.store.record_evidence(items, result.run_id)

    def finalize(self, result: SwarmResult, report_path: Path | None = None) -> None:
        if result.run_id is None:
            return
        self.store.finish_run(
            result.run_id,
            regime=str(result.reports.get("volatility_regime").data.get("regime", ""))
            if result.reports.get("volatility_regime") else "",
            agents_ok=result.agents_ok,
            agents_failed=result.agents_failed,
            report_path=str(report_path) if report_path else "",
            notes=f"P(up)={result.probability:.3f} conf={result.confidence}",
        )


def _dominant_event_type(events: list) -> str:
    """The event type this session is really about.

    Highest priority wins, ties broken by count. Returns "any" when the Event
    Brain classified nothing — an honest absence, not a guessed label, so the
    resolver falls back through its normal hierarchy instead of routing on a
    context that was never observed.
    """
    if not events:
        return "any"
    ranked: dict[str, tuple[int, int]] = {}
    for e in events:
        key = getattr(getattr(e, "event_type", None), "value", None)
        if not key:
            continue
        priority = getattr(getattr(e, "priority", None), "rank", None)
        if priority is None:
            priority = {"critical": 4, "high": 3, "normal": 2, "low": 1}.get(
                str(getattr(getattr(e, "priority", ""), "value", "")).lower(), 2)
        best, count = ranked.get(key, (0, 0))
        ranked[key] = (max(best, int(priority)), count + 1)
    if not ranked:
        return "any"
    return max(ranked.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]


def _snapshot_from(reports: dict) -> dict:
    """Build the Event Brain's input from whatever has run so far.

    Shared with Pipeline2 rather than duplicated, because two slightly
    different snapshot builders would eventually disagree about what the
    market looked like this morning.
    """
    from .pipeline2 import Pipeline2

    return Pipeline2.build_snapshot(reports)


def _unreviewed_publication(reports: dict) -> PublicationSet:
    """Legacy-mode escape hatch: 1.x ideas, explicitly marked unreviewed.

    Reachable only when `allow_unreviewed_publication` is set *and* the mode is
    `legacy`. It exists so the pre-2.0 behaviour stays available for
    benchmarking and debugging, and it is loud about what it is. It is never
    reached by a failure — a failure suppresses.
    """
    from .recommend.engine import Conviction, Recommendation, RecommendationType

    pb = reports.get("playbook")
    if not pb or not getattr(pb, "usable", False):
        return PublicationSet(mode="legacy")

    approved: list[Recommendation] = []
    for kind, key in (("call", "calls"), ("put", "puts"), ("stock", "stocks")):
        for idea in pb.data.get(key, []) or []:
            direction = idea.get("direction")
            approved.append(Recommendation(
                id=f"legacy_{uuid.uuid4().hex[:12]}",
                subject=idea.get("symbol", "?"),
                rec_type=(RecommendationType.FAVORABLE if direction == "long"
                          else RecommendationType.UNFAVORABLE),
                # Moderate, not high: an unreviewed idea is publishable in this
                # mode but never carries the system's top conviction label.
                conviction=Conviction.MODERATE_CONVICTION,
                direction=direction,
                forecast_probability=idea.get("probability"),
                confidence=int(idea.get("confidence", 0)),
                original_confidence=int(idea.get("confidence", 0)),
                expected_r=idea.get("expected_r"),
                entry=idea.get("entry"), target=idea.get("target"),
                stop=idea.get("stop"),
                rationale=idea.get("rationale", ""),
                invalidation=[idea.get("invalidation", "")],
                review_status="NOT_REVIEWED",
                uncertainty_notes=["legacy mode: this idea did not pass through "
                                   "the 2.0 review gate"],
                system_version="1.x-compat",
                source_kind=kind,
                source_payload=dict(idea),
            ))
    return PublicationSet(approved=approved, mode="legacy")


def _gap_bucket(gap: float | None) -> str:
    if gap is None:
        return "none"
    a = abs(gap)
    side = "up" if gap > 0 else "down"
    if a < 0.5:
        return "flat"
    if a < 1.5:
        return f"small_{side}"
    if a < 3.0:
        return f"medium_{side}"
    return f"large_{side}"
