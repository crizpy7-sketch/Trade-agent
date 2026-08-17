"""Cost-aware model routing.

Three tiers, and the first one is not a model:

    DETERMINISTIC  no LLM. Thresholds, arithmetic, lookups, schema validation.
                   Faster, free, reproducible and testable. This is the default
                   and most work should stay here.
    CHEAP          classification, extraction, routine summarisation.
    STRONG         contradiction resolution, red-team reasoning, investigation
                   planning, hypothesis generation.

The routing rule is that reasoning is spent where reasoning changes the answer.
Sending "is 3.2 greater than 2.0" to a frontier model is not sophistication, it
is waste with extra latency and a chance of being wrong.

Budgets are enforced, not advisory: once the daily cap is reached the router
degrades to DETERMINISTIC and the caller must cope, which every caller here is
written to do.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger("marketswarm.llm_router")


class Tier(str, Enum):
    DETERMINISTIC = "deterministic"
    CHEAP = "cheap"
    STRONG = "strong"


@dataclass
class ModelSpec:
    tier: Tier
    model_id: str | None
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    max_tokens: int = 1024

    def estimate_cost(self, in_tok: int, out_tok: int) -> float:
        return (in_tok / 1e6 * self.input_cost_per_mtok
                + out_tok / 1e6 * self.output_cost_per_mtok)


# Model IDs are configuration, not hard-coded intelligence. Swapping a model
# must never require touching logic — the durable value of this system lives in
# its data, evaluation and tests, not in one vendor's checkpoint.
DEFAULT_MODELS: dict[Tier, ModelSpec] = {
    Tier.DETERMINISTIC: ModelSpec(Tier.DETERMINISTIC, None),
    Tier.CHEAP: ModelSpec(Tier.CHEAP, "claude-haiku-4-5-20251001",
                          input_cost_per_mtok=1.0, output_cost_per_mtok=5.0,
                          max_tokens=1024),
    Tier.STRONG: ModelSpec(Tier.STRONG, "claude-opus-4-5",
                           input_cost_per_mtok=15.0, output_cost_per_mtok=75.0,
                           max_tokens=2048),
}


# Which tier each task needs. Anything absent defaults to DETERMINISTIC, so a
# new task must consciously opt into spending money.
TASK_TIERS: dict[str, Tier] = {
    "anomaly_detection": Tier.DETERMINISTIC,
    "threshold_check": Tier.DETERMINISTIC,
    "bracket_construction": Tier.DETERMINISTIC,
    "probability_fusion": Tier.DETERMINISTIC,
    "agent_routing": Tier.DETERMINISTIC,
    "scoring": Tier.DETERMINISTIC,
    "calibration": Tier.DETERMINISTIC,
    "review_gate": Tier.DETERMINISTIC,

    "headline_classification": Tier.CHEAP,
    "filing_extraction": Tier.CHEAP,
    "summarisation": Tier.CHEAP,
    "entity_linking": Tier.CHEAP,

    "investigation_planning": Tier.STRONG,
    "hypothesis_pruning": Tier.STRONG,
    "contradiction_resolution": Tier.STRONG,
    "red_team_reasoning": Tier.STRONG,
    "narrative": Tier.STRONG,
    "postmortem": Tier.STRONG,
    "research_hypothesis": Tier.STRONG,
}


@dataclass
class Budget:
    daily_usd: float = 2.00
    per_run_usd: float = 0.50
    spent_today: float = 0.0
    spent_this_run: float = 0.0
    day: str = field(default_factory=lambda: dt.date.today().isoformat())
    calls: int = 0

    def roll_day(self) -> None:
        today = dt.date.today().isoformat()
        if today != self.day:
            self.day = today
            self.spent_today = 0.0

    def can_afford(self, est: float) -> bool:
        self.roll_day()
        return (self.spent_today + est <= self.daily_usd
                and self.spent_this_run + est <= self.per_run_usd)

    def charge(self, amount: float) -> None:
        self.roll_day()
        self.spent_today += amount
        self.spent_this_run += amount
        self.calls += 1

    def start_run(self) -> None:
        self.spent_this_run = 0.0

    def snapshot(self) -> dict:
        return {
            "day": self.day,
            "spent_today": round(self.spent_today, 4),
            "daily_cap": self.daily_usd,
            "spent_this_run": round(self.spent_this_run, 4),
            "run_cap": self.per_run_usd,
            "calls": self.calls,
            "headroom_today": round(max(0.0, self.daily_usd - self.spent_today), 4),
        }


@dataclass
class RoutingDecision:
    task: str
    tier: Tier
    model_id: str | None
    reason: str
    estimated_cost: float = 0.0
    downgraded: bool = False

    @property
    def use_llm(self) -> bool:
        return self.tier is not Tier.DETERMINISTIC and self.model_id is not None


class ModelRouter:
    def __init__(self, api_key: str | None = None, budget: Budget | None = None,
                 models: dict[Tier, ModelSpec] | None = None,
                 enabled: bool = True):
        self.api_key = api_key
        self.enabled = enabled and bool(api_key)
        self.budget = budget or Budget()
        self.models = models or dict(DEFAULT_MODELS)
        self._client = None

    def route(
        self,
        task: str,
        importance: float = 0.5,        # 0-1: financial significance of the call
        uncertainty: float = 0.5,       # 0-1: how unclear the situation is
        disagreement: float = 0.0,      # 0-1: how much the specialists conflict
        estimated_input_tokens: int = 2000,
        estimated_output_tokens: int = 600,
    ) -> RoutingDecision:
        """Pick a tier. Escalation requires a reason; so does staying cheap."""
        base = TASK_TIERS.get(task, Tier.DETERMINISTIC)

        if base is Tier.DETERMINISTIC:
            return RoutingDecision(task, Tier.DETERMINISTIC, None,
                                   "deterministic logic is sufficient and exact")

        if not self.enabled:
            return RoutingDecision(task, Tier.DETERMINISTIC, None,
                                   "no API key configured — degrading to deterministic",
                                   downgraded=True)

        tier = base
        reason = f"task '{task}' is mapped to {base.value}"

        # Escalate a cheap task when the situation is genuinely hard.
        if base is Tier.CHEAP and (disagreement > 0.6 or (importance > 0.75
                                                          and uncertainty > 0.6)):
            tier = Tier.STRONG
            reason = (f"escalated: importance {importance:.2f}, uncertainty "
                      f"{uncertainty:.2f}, disagreement {disagreement:.2f}")

        # Demote a strong task when nothing is at stake.
        if base is Tier.STRONG and importance < 0.25 and disagreement < 0.2:
            tier = Tier.CHEAP
            reason = f"demoted: low importance ({importance:.2f}) and no disagreement"

        spec = self.models[tier]
        est = spec.estimate_cost(estimated_input_tokens, estimated_output_tokens)

        if not self.budget.can_afford(est):
            cheap = self.models[Tier.CHEAP]
            cheap_est = cheap.estimate_cost(estimated_input_tokens, estimated_output_tokens)
            if tier is Tier.STRONG and self.budget.can_afford(cheap_est):
                return RoutingDecision(task, Tier.CHEAP, cheap.model_id,
                                       f"budget: ${est:.4f} unaffordable, using cheap tier",
                                       cheap_est, downgraded=True)
            return RoutingDecision(task, Tier.DETERMINISTIC, None,
                                   f"budget exhausted ({self.budget.snapshot()}) — "
                                   f"degrading to deterministic",
                                   downgraded=True)

        return RoutingDecision(task, tier, spec.model_id, reason, est)

    # ------------------------------------------------------------------

    def _client_or_none(self):
        if self._client is not None:
            return self._client
        if not self.enabled:
            return None
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        except Exception as exc:  # noqa: BLE001
            log.info("Anthropic client unavailable: %s", exc)
            return None
        return self._client

    def complete(self, decision: RoutingDecision, system: str, user: str,
                 max_tokens: int | None = None) -> str | None:
        """Execute a routed call. Returns None on any failure — every caller
        must have a deterministic path, so a model outage degrades rather than
        breaks."""
        if not decision.use_llm:
            return None
        client = self._client_or_none()
        if client is None:
            return None

        spec = self.models[decision.tier]
        try:
            resp = client.messages.create(
                model=decision.model_id,
                max_tokens=max_tokens or spec.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            in_tok = getattr(resp.usage, "input_tokens", 0) if hasattr(resp, "usage") else 0
            out_tok = getattr(resp.usage, "output_tokens", 0) if hasattr(resp, "usage") else 0
            self.budget.charge(spec.estimate_cost(in_tok, out_tok))
            return "".join(b.text for b in resp.content
                           if getattr(b, "type", "") == "text").strip() or None
        except Exception as exc:  # noqa: BLE001
            log.warning("model call failed (%s): %s", decision.model_id, exc)
            return None

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "budget": self.budget.snapshot(),
            "models": {t.value: (s.model_id or "n/a") for t, s in self.models.items()},
        }
