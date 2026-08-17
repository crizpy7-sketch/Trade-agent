"""One API for "how much should I trust this agent, here, now?".

Before this module there were two weight stores and no rule for choosing
between them:

    LearningEngine.agent_weights()      Brier-derived, global, one number
    ContributionTracker.get_weight()    ablation-derived, contextual, sparse

Both are useful and they measure different things. The Brier weight says an
agent has been *accurate* overall. The contribution weight says an agent has
*added value beyond the others* in a specific regime and event type, which is
the question that actually matters when you are deciding whether to spend four
cost units running it. An agent can be individually accurate and contribute
nothing, because two other agents already said the same thing.

The rule implemented here, in order:

  1. If contextual contribution has enough observations for this exact
     context, use it — the most specific measurement wins.
  2. Otherwise back off through broader contexts (regime+event → regime → all),
     which `ContextKey.generalisations()` already orders.
  3. Otherwise fall back to the global Brier weight.
  4. Otherwise 1.0, the honest prior for something never measured.

Consumers ask this resolver and nothing else. That is the point: five call
sites independently picking a weight store is how a system ends up with five
different opinions about the same agent.
"""

from __future__ import annotations

import logging
import sqlite3

from .contribution import WEIGHT_CEILING, WEIGHT_FLOOR, ContextKey, ContributionTracker

log = logging.getLogger("marketswarm.memory.resolver")


class ContextualWeightResolver:
    """Combines baseline reliability and contextual contribution.

    Deliberately read-only: it resolves weights, it never writes them. Writing
    is `ContributionTracker.update_from_resolved`, which runs in the scoring
    pass where outcomes are actually known.
    """

    def __init__(self, conn: sqlite3.Connection,
                 baseline_weights: dict[str, float] | None = None):
        self.conn = conn
        self.tracker = ContributionTracker(conn)
        self.baseline = dict(baseline_weights or {})
        self._explain: dict[str, str] = {}

    # ------------------------------------------------------------------

    def get_agent_weight(
        self,
        agent: str,
        regime: str = "",
        event_type: str = "",
        horizon: str = "intraday",
        default: float = 1.0,
    ) -> float:
        """The weight to use for this agent in this context.

        Bounded to the same floor and ceiling the tracker enforces, so a
        resolver bug cannot hand fusion a weight the learning layer would
        never have produced.
        """
        ctx = ContextKey(regime=regime or "any", event_type=event_type or "any",
                         horizon=horizon or "intraday")

        contextual = self.tracker.get_weight(agent, ctx, default=None)
        if contextual is not None:
            self._explain[agent] = f"contextual({ctx.key()})"
            return _clamp(contextual)

        if agent in self.baseline:
            self._explain[agent] = "baseline(brier)"
            return _clamp(self.baseline[agent])

        self._explain[agent] = "prior(unmeasured)"
        return _clamp(default)

    def weights_for(self, agents: list[str], regime: str = "",
                    event_type: str = "", horizon: str = "intraday") -> dict[str, float]:
        return {a: self.get_agent_weight(a, regime, event_type, horizon) for a in agents}

    def routing_weights(self, agents: list[str], regime: str = "",
                        event_type: str = "") -> dict[str, float]:
        """Weights for the capability registry's value-per-cost ranking.

        Same numbers as fusion uses. Routing an agent in because it is
        reliable, then down-weighting its output because it is not, would be
        two systems disagreeing about the same measurement.
        """
        return self.weights_for(agents, regime=regime, event_type=event_type)

    # ------------------------------------------------------------------

    def provenance(self) -> dict[str, str]:
        """Which store answered for each agent asked so far. Recorded in the
        run trace so a weight change can be attributed."""
        return dict(self._explain)

    def contexts_available(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT context_key) FROM agent_context_scores").fetchone()
        return int(row[0]) if row else 0

    def summary(self) -> dict:
        contextual = sum(1 for v in self._explain.values() if v.startswith("contextual"))
        return {
            "agents_resolved": len(self._explain),
            "from_contextual_learning": contextual,
            "from_baseline": sum(1 for v in self._explain.values() if v.startswith("baseline")),
            "unmeasured": sum(1 for v in self._explain.values() if v.startswith("prior")),
            "contexts_in_store": self.contexts_available(),
        }


def _clamp(w: float) -> float:
    return max(WEIGHT_FLOOR, min(WEIGHT_CEILING, float(w)))
