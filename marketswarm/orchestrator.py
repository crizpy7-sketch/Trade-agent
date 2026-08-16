"""The swarm orchestrator.

Responsibilities:
  1. refuse to run when the market is closed
  2. stage agents by dependency and run each stage concurrently
  3. contain failures so a partial outage degrades the report instead of
     killing the run
  4. fuse, persist, and hand off to the renderer
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import clock
from .agents import ALL_AGENTS, AgentReport, SwarmContext
from .config import Config
from .memory import LearningEngine, MemoryStore, Prediction
from .providers.base import DataClient
from .providers.earnings import EarningsData
from .providers.econ import EconData
from .providers.edgar import EdgarData
from .providers.market import MarketData
from .providers.news import NewsData
from .providers.options import OptionsData

log = logging.getLogger("marketswarm.orchestrator")


@dataclass
class SwarmResult:
    run_date: dt.date
    market_open: bool
    closed_reason: str | None = None
    reports: dict[str, AgentReport] = field(default_factory=dict)
    probability: float = 0.5
    confidence: int = 0
    ideas: dict = field(default_factory=dict)
    run_id: int | None = None
    report_path: Path | None = None
    started_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    finished_at: dt.datetime | None = None
    narrative: str | None = None

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
        self.learning = LearningEngine(self.store)

    async def run(self, run_date: dt.date | None = None, force: bool = False) -> SwarmResult:
        run_date = run_date or clock.now_et().date()
        status = clock.day_status(run_date)

        if not status.is_trading_day and not force:
            log.info("market closed on %s (%s)", run_date, status.reason)
            return SwarmResult(run_date=run_date, market_open=False, closed_reason=status.reason,
                               finished_at=dt.datetime.now(dt.timezone.utc))

        result = SwarmResult(run_date=run_date, market_open=True)
        result.run_id = self.store.start_run(run_date.isoformat(), "premarket")

        # Load everything the agent learned from previous sessions.
        self.config.calibrator = self.learning.load_calibrator()
        self.config.agent_weights = self.learning.agent_weights()
        self.config.lessons = self.store.active_lessons()
        log.info("loaded calibrator (%s) and %d agent weights",
                 self.config.calibrator.diagnosis, len(self.config.agent_weights))

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

            for stage_no, stage in enumerate(stage_agents(ALL_AGENTS), 1):
                names = [a.name for a in stage]
                log.info("stage %d: %s", stage_no, ", ".join(names))
                instances = [a() for a in stage]
                for inst in instances:
                    inst.timeout = min(inst.timeout, self.config.timeout_seconds * 2)
                reports = await asyncio.gather(*(i.execute(ctx) for i in instances))
                for r in reports:
                    ctx.reports[r.agent] = r
                    log.info("  %s: %s (%s, %dms)", r.agent, r.headline, r.status, r.duration_ms)

            result.reports = ctx.reports

        cv = ctx.reports.get("cross_verify")
        if cv and cv.usable:
            result.probability = cv.data.get("probability", 0.5)
            result.confidence = cv.data.get("confidence_score", 0)

        pb = ctx.reports.get("playbook")
        if pb and pb.usable:
            result.ideas = {"calls": pb.data.get("calls", []),
                            "puts": pb.data.get("puts", []),
                            "stocks": pb.data.get("stocks", [])}
            self._persist_predictions(result, ctx)

        self._persist_evidence(result, ctx)
        result.finished_at = dt.datetime.now(dt.timezone.utc)
        return result

    def _persist_predictions(self, result: SwarmResult, ctx: SwarmContext) -> None:
        """Write every published idea to memory so it can be scored tonight."""
        regime = ctx.data_of("volatility_regime", "regime", "unknown")
        contributions = ctx.data_of("cross_verify", "contributions", {}) or {}

        for kind, ideas in (("call_idea", result.ideas.get("calls", [])),
                            ("put_idea", result.ideas.get("puts", [])),
                            ("stock_setup", result.ideas.get("stocks", []))):
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
                        "gap_bucket": _gap_bucket(gap),
                        "session_day": result.run_date.strftime("%A"),
                        "strike": i.get("strike"),
                        "expiration": i.get("expiration"),
                        "liquidity": i.get("liquidity"),
                    },
                    contributing_agents=contributions,
                )
                self.store.record_prediction(pred, result.run_id)

        # A single index-level directional call, scored the same way — this is
        # the series that drives the global calibrator.
        spy = (ctx.data_of("technicals", "setups", {}) or {}).get("SPY")
        if spy:
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
                              "session_day": result.run_date.strftime("%A")},
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


def _gap_bucket(gap: float | None) -> str:
    if gap is None:
        return "none"
    a = abs(gap)
    side = "up" if gap > 0 else "down"
    if a < 0.5:
        return f"flat"
    if a < 1.5:
        return f"small_{side}"
    if a < 3.0:
        return f"medium_{side}"
    return f"large_{side}"
