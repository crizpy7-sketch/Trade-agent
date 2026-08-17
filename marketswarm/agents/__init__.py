from .base import AgentReport, BaseAgent, SwarmContext
from .flow_agents import InstitutionalAgent, OptionsFlowAgent, SentimentAgent
from .market_agents import (
    FuturesAgent,
    GlobalMarketsAgent,
    OvernightScanAgent,
    TechnicalAgent,
    VolatilityRegimeAgent,
)
from .news_agents import BreakingNewsAgent, EarningsAgent, EconCalendarAgent, SECFilingsAgent
from .redteam import RedTeamAgent
from .synthesis import CrossVerificationAgent, PlaybookAgent, RiskAgent

# Execution order is derived from `depends_on`; this is just the roster.
ALL_AGENTS: list[type[BaseAgent]] = [
    OvernightScanAgent,
    GlobalMarketsAgent,
    FuturesAgent,
    VolatilityRegimeAgent,
    BreakingNewsAgent,
    EconCalendarAgent,
    EarningsAgent,
    SECFilingsAgent,
    OptionsFlowAgent,
    InstitutionalAgent,
    TechnicalAgent,
    SentimentAgent,
    CrossVerificationAgent,
    RiskAgent,
    PlaybookAgent,
    RedTeamAgent,
]

__all__ = ["ALL_AGENTS", "AgentReport", "BaseAgent", "SwarmContext"]
