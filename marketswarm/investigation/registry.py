"""Agent capability registry.

In 1.x the 16 agents were a fixed sequence: everything ran, every session. Here
they become *capabilities* the planner can select from, described by what they
are good for, what they cost, and how they have actually performed.

The registry is data, not behaviour. It does not run agents — it tells the
Chief Investigator which ones are worth running for a given event, and the
orchestrator still owns execution, timeouts and failure containment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .event_brain import EventType


@dataclass
class AgentCapability:
    name: str
    specialties: list[str]
    supported_event_types: list[EventType]
    expected_cost: float          # relative units; 1.0 ≈ one cheap provider call
    expected_latency_ms: int
    required_inputs: list[str] = field(default_factory=list)
    output_schema: list[str] = field(default_factory=list)
    produces_index_signal: bool = False
    produces_symbol_signal: bool = False
    always_run: bool = False      # foundational; skipping breaks other agents
    evidence_clusters: list[str] = field(default_factory=list)
    historical_reliability: float = 1.0   # overwritten from learned scores

    def handles(self, event_type: EventType) -> bool:
        return event_type in self.supported_event_types

    def value_score(self, event_types: set[EventType]) -> float:
        """Crude expected-value-per-cost for this event mix.

        Deliberately crude: the point is to avoid running nine agents when three
        will do, not to optimise a schedule to three decimal places.
        """
        if self.always_run:
            return 1e6
        matches = sum(1 for e in event_types if self.handles(e))
        if matches == 0:
            return 0.0
        return (matches * max(self.historical_reliability, 0.05)) / max(self.expected_cost, 0.1)


# An optional agent whose measured contribution has fallen to (or near) the
# learning layer's weight floor has been shown to subtract value in this
# context, over at least `MIN_OBSERVATIONS` resolved outcomes. Running it anyway
# spends budget to make the answer worse.
#
# Without this, a learned weight could only ever change the *order* in which
# agents were considered — and since the budget rarely binds on a busy session,
# ordering changed nothing at all. Learning that cannot change behaviour is not
# learning.
#
# `always_run` agents are never excluded on reliability: losing fusion or
# levels breaks the report outright, which is worse than a weak signal. The
# floor sits above `WEIGHT_FLOOR` (0.15) so only agents pinned at the bottom by
# real evidence are dropped, never merely below-average ones.
UNRELIABLE_THRESHOLD = 0.25

_E = EventType

DEFAULT_CAPABILITIES: list[AgentCapability] = [
    AgentCapability(
        name="overnight_scan",
        specialties=["pre-market gaps", "overnight participation", "watchlist breadth"],
        supported_event_types=[_E.PRICE_GAP, _E.RELATIVE_VOLUME, _E.QUIET,
                               _E.EARNINGS_REACTION, _E.COMPANY_NEWS],
        expected_cost=2.0, expected_latency_ms=4000,
        required_inputs=["universe"],
        output_schema=["profiles", "gappers", "breadth_pct_up"],
        produces_index_signal=True, always_run=True,
        evidence_clusters=["market_beta", "company_news"],
    ),
    AgentCapability(
        name="global_markets",
        specialties=["Asia session", "Europe session", "overnight risk tone"],
        supported_event_types=[_E.INDEX_MOVE, _E.MACRO_RELEASE, _E.GEOPOLITICAL,
                               _E.VOLATILITY_SPIKE],
        expected_cost=1.5, expected_latency_ms=3000,
        output_schema=["asia_avg_pct", "europe_avg_pct", "aligned"],
        produces_index_signal=True, evidence_clusters=["market_beta", "macro"],
    ),
    AgentCapability(
        name="futures",
        specialties=["index futures", "rates", "dollar", "commodities"],
        supported_event_types=[_E.INDEX_MOVE, _E.MACRO_RELEASE, _E.VOLATILITY_SPIKE,
                               _E.CORRELATION_BREAKDOWN],
        expected_cost=1.5, expected_latency_ms=3000,
        output_schema=["es_pct", "nq_pct", "rotation", "tnx"],
        produces_index_signal=True, always_run=True,
        evidence_clusters=["market_beta", "macro"],
    ),
    AgentCapability(
        name="volatility_regime",
        specialties=["VIX complex", "term structure", "GARCH", "regime"],
        supported_event_types=[_E.VOLATILITY_SPIKE, _E.INDEX_MOVE, _E.MACRO_RELEASE,
                               _E.CORRELATION_BREAKDOWN],
        expected_cost=2.5, expected_latency_ms=5000,
        output_schema=["vix", "regime", "term_structure", "garch"],
        produces_index_signal=True, always_run=True,
        evidence_clusters=["volatility"],
    ),
    AgentCapability(
        name="breaking_news",
        specialties=["overnight headlines", "corroboration", "materiality"],
        supported_event_types=[_E.COMPANY_NEWS, _E.GEOPOLITICAL, _E.PRICE_GAP,
                               _E.ANALYST_ACTION, _E.GUIDANCE],
        expected_cost=3.0, expected_latency_ms=8000,
        output_schema=["high_materiality", "ticker_mentions"],
        evidence_clusters=["company_news", "macro"],
    ),
    AgentCapability(
        name="econ_calendar",
        specialties=["scheduled releases", "FOMC", "treasury curve"],
        supported_event_types=[_E.MACRO_RELEASE, _E.VOLATILITY_SPIKE],
        expected_cost=1.5, expected_latency_ms=3000,
        output_schema=["events", "event_risk", "very_high_impact"],
        evidence_clusters=["macro"],
    ),
    AgentCapability(
        name="earnings",
        specialties=["reactions", "surprise vs gap", "reporting tonight"],
        supported_event_types=[_E.EARNINGS_REACTION, _E.EARNINGS_UPCOMING,
                               _E.GUIDANCE, _E.PRICE_GAP],
        expected_cost=2.5, expected_latency_ms=5000,
        output_schema=["reacting", "reporting_tonight"],
        produces_symbol_signal=True,
        evidence_clusters=["company_fundamental", "company_news"],
    ),
    AgentCapability(
        name="sec_filings",
        specialties=["8-K", "primary source", "items"],
        supported_event_types=[_E.SEC_FILING, _E.PRICE_GAP, _E.GUIDANCE,
                               _E.EARNINGS_REACTION],
        expected_cost=3.5, expected_latency_ms=9000,
        output_schema=["filings", "high_impact"],
        produces_symbol_signal=True,
        evidence_clusters=["company_fundamental"],
    ),
    AgentCapability(
        name="options_flow",
        specialties=["implied move", "skew", "OI walls", "inferred activity"],
        supported_event_types=[_E.OPTIONS_ANOMALY, _E.EARNINGS_UPCOMING,
                               _E.VOLATILITY_SPIKE, _E.PRICE_GAP],
        expected_cost=4.0, expected_latency_ms=10000,
        output_schema=["flows"],
        produces_index_signal=True, produces_symbol_signal=True,
        evidence_clusters=["options_positioning", "volatility"],
    ),
    AgentCapability(
        name="institutional",
        specialties=["Form 4 clusters", "13D/G"],
        supported_event_types=[_E.SEC_FILING, _E.COMPANY_NEWS],
        expected_cost=3.5, expected_latency_ms=9000,
        output_schema=["insider_clusters", "ownership_moves"],
        evidence_clusters=["insider"],
    ),
    AgentCapability(
        name="technicals",
        specialties=["trend", "levels", "ATR", "squeeze"],
        supported_event_types=[_E.PRICE_GAP, _E.SECTOR_DIVERGENCE, _E.QUIET,
                               _E.INDEX_MOVE, _E.RELATIVE_VOLUME],
        expected_cost=3.0, expected_latency_ms=7000,
        required_inputs=["overnight_scan"],
        output_schema=["setups"],
        produces_index_signal=True, produces_symbol_signal=True,
        always_run=True,           # brackets cannot be built without levels
        evidence_clusters=["technical"],
    ),
    AgentCapability(
        name="sentiment",
        specialties=["measurable positioning proxies"],
        supported_event_types=[_E.VOLATILITY_SPIKE, _E.INDEX_MOVE, _E.QUIET],
        expected_cost=0.5, expected_latency_ms=500,
        required_inputs=["volatility_regime", "options_flow"],
        output_schema=["score", "label"],
        produces_index_signal=True,
        evidence_clusters=["sentiment"],
    ),
    AgentCapability(
        name="cross_verify",
        specialties=["fusion", "contradiction", "single-source audit"],
        supported_event_types=list(EventType),
        expected_cost=0.2, expected_latency_ms=200,
        output_schema=["probability", "contributions", "confidence_score"],
        always_run=True, evidence_clusters=[],
    ),
    AgentCapability(
        name="risk",
        specialties=["risk budget", "event risk", "portfolio heat"],
        supported_event_types=list(EventType),
        expected_cost=0.2, expected_latency_ms=200,
        required_inputs=["cross_verify"],
        output_schema=["risks", "suggested_risk_pct"],
        always_run=True, evidence_clusters=[],
    ),
    AgentCapability(
        name="playbook",
        specialties=["bracket construction", "option legs", "ranking"],
        supported_event_types=list(EventType),
        expected_cost=1.0, expected_latency_ms=2000,
        required_inputs=["cross_verify", "technicals"],
        output_schema=["calls", "puts", "stocks"],
        always_run=True, evidence_clusters=[],
    ),
    AgentCapability(
        name="red_team",
        specialties=["adversarial review"],
        supported_event_types=list(EventType),
        expected_cost=0.3, expected_latency_ms=300,
        required_inputs=["playbook"],
        output_schema=["objections"],
        always_run=True, evidence_clusters=[],
    ),
]


class CapabilityRegistry:
    def __init__(self, capabilities: list[AgentCapability] | None = None):
        self._caps: dict[str, AgentCapability] = {
            c.name: c for c in (capabilities or DEFAULT_CAPABILITIES)
        }

    def __contains__(self, name: str) -> bool:
        return name in self._caps

    def __len__(self) -> int:
        return len(self._caps)

    def get(self, name: str) -> AgentCapability | None:
        return self._caps.get(name)

    def all(self) -> list[AgentCapability]:
        return list(self._caps.values())

    def names(self) -> list[str]:
        return list(self._caps)

    def always_run(self) -> list[str]:
        return [c.name for c in self._caps.values() if c.always_run]

    def for_event(self, event_type: EventType) -> list[AgentCapability]:
        return [c for c in self._caps.values() if c.handles(event_type)]

    def select(
        self,
        event_types: set[EventType],
        budget: float | None = None,
        max_agents: int | None = None,
        exclude: set[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Pick the smallest effective team. Returns (selected, skipped).

        Mandatory agents are always included and are not charged against the
        budget — dropping them produces a report that cannot be assembled at
        all, which is worse than an expensive one.
        """
        exclude = exclude or set()
        chosen: list[str] = []
        spend = 0.0

        for c in self._caps.values():
            if c.always_run and c.name not in exclude:
                chosen.append(c.name)

        optional = [
            c for c in self._caps.values()
            if not c.always_run and c.name not in exclude
            and c.value_score(event_types) > 0
            and c.historical_reliability > UNRELIABLE_THRESHOLD
        ]
        optional.sort(key=lambda c: -c.value_score(event_types))

        for c in optional:
            if max_agents is not None and len(chosen) >= max_agents:
                break
            if budget is not None and spend + c.expected_cost > budget:
                continue
            chosen.append(c.name)
            spend += c.expected_cost

        # Honour declared dependencies: a selected agent whose input is missing
        # would silently degrade, which is exactly what 2.0 is meant to stop.
        changed = True
        while changed:
            changed = False
            for name in list(chosen):
                cap = self._caps[name]
                for dep in cap.required_inputs:
                    if dep in self._caps and dep not in chosen:
                        chosen.append(dep)
                        spend += self._caps[dep].expected_cost
                        changed = True

        skipped = [n for n in self._caps if n not in chosen]
        return chosen, skipped

    def estimated_cost(self, names: list[str]) -> float:
        return sum(self._caps[n].expected_cost for n in names if n in self._caps)

    def update_reliability(self, scores: dict[str, float]) -> None:
        """Feed learned weights back in so routing improves with experience."""
        for name, w in scores.items():
            cap = self._caps.get(name)
            if cap:
                cap.historical_reliability = max(0.05, min(2.0, float(w)))


REGISTRY = CapabilityRegistry()
