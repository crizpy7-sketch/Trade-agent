"""The Chief Investigator.

Sits above the specialist swarm and answers one question the 1.x pipeline never
asked: *what needs investigating right now?*

The division of labour is deliberate:

  deterministic  which events fired, which agents can address them, what the
                 budget allows, when to stop. All testable, all reproducible,
                 all free.
  LLM (optional) narrowing a long hypothesis list to the plausible few, and
                 reading a contradiction that the rules cannot resolve.

The Chief works fully without an LLM. The model sharpens the plan; it is not
load-bearing, and the system must never require it to function.

Recursion is bounded by construction: plans carry a `PlanBudget`, follow-up
investigations are depth-limited, and nothing here can spawn work without
charging it against a cap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .event_brain import DetectedEvent, EventBrain, EventPriority, EventType
from .plan import InvestigationPlan, PlanBudget, StopReason
from .registry import REGISTRY, CapabilityRegistry

log = logging.getLogger("marketswarm.chief")


# Effort allocated by priority. This is the mechanism that stops a quiet AAPL
# costing the same as an NVDA earnings gap.
PRIORITY_BUDGETS: dict[EventPriority, dict] = {
    EventPriority.CRITICAL: {"max_iterations": 3, "max_agents": 14, "max_cost": 40.0,
                             "max_seconds": 180.0},
    EventPriority.HIGH:     {"max_iterations": 2, "max_agents": 11, "max_cost": 28.0,
                             "max_seconds": 120.0},
    EventPriority.ELEVATED: {"max_iterations": 2, "max_agents": 9, "max_cost": 20.0,
                             "max_seconds": 90.0},
    EventPriority.NORMAL:   {"max_iterations": 1, "max_agents": 7, "max_cost": 12.0,
                             "max_seconds": 60.0},
    EventPriority.LOW:      {"max_iterations": 1, "max_agents": 5, "max_cost": 6.0,
                             "max_seconds": 30.0},
}

# Evidence each event type demands before a conclusion is defensible.
EVIDENCE_REQUIREMENTS: dict[EventType, list[str]] = {
    EventType.PRICE_GAP: [
        "identify the catalyst, or state explicitly that none was found",
        "distinguish company-specific movement from sector or market beta",
        "confirm whether the pre-market move has real volume behind it",
    ],
    EventType.EARNINGS_REACTION: [
        "the reported numbers versus consensus",
        "guidance direction, which frequently contradicts the headline",
        "whether the gap agrees or disagrees with the surprise",
    ],
    EventType.SEC_FILING: [
        "which 8-K items were filed and what they mean in plain English",
        "whether the tape has already reacted to the filing",
    ],
    EventType.OPTIONS_ANOMALY: [
        "whether the activity is opening or closing (often undeterminable — say so)",
        "whether it is a hedge rather than a directional view",
        "the implied move versus the proposed target",
    ],
    EventType.MACRO_RELEASE: [
        "release time relative to the open",
        "consensus expectation where available",
        "how the same release was handled recently",
    ],
    EventType.VOLATILITY_SPIKE: [
        "VIX term structure — backwardation or contango",
        "whether the spike is index-wide or concentrated",
    ],
    EventType.SECTOR_DIVERGENCE: [
        "the sector's own move for comparison",
        "any company-specific catalyst explaining the spread",
    ],
    EventType.COMPANY_NEWS: [
        "corroboration across independent outlets",
        "whether the tape moved before or after the headline",
    ],
    EventType.RELATIVE_VOLUME: [
        "what is driving the participation",
    ],
}


@dataclass
class InvestigationResult:
    plan: InvestigationPlan
    reports: dict = field(default_factory=dict)
    stop_reason: StopReason = StopReason.COMPLETED
    iterations: int = 0
    contradictions: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.stop_reason in (StopReason.SUFFICIENT_EVIDENCE,
                                    StopReason.COMPLETED,
                                    StopReason.NO_NEW_INFORMATION)


class ChiefInvestigator:
    def __init__(
        self,
        registry: CapabilityRegistry | None = None,
        brain: EventBrain | None = None,
        narrator=None,                       # optional LLM for hypothesis pruning
        max_parallel_investigations: int = 6,
        max_followup_depth: int = 1,
    ):
        self.registry = registry or REGISTRY
        self.brain = brain or EventBrain()
        self.narrator = narrator
        self.max_parallel_investigations = max_parallel_investigations
        self.max_followup_depth = max_followup_depth

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------

    def plan_session(self, snapshot: dict, agent_weights: dict[str, float] | None = None
                     ) -> list[InvestigationPlan]:
        """Turn a cheap market snapshot into a prioritised set of plans."""
        if agent_weights:
            self.registry.update_reliability(agent_weights)

        triage = self.brain.triage(snapshot)
        plans: list[InvestigationPlan] = []

        # 1. The market itself always gets a plan — the index read is the
        #    backbone every symbol decision leans on.
        market_events = [e for e in triage["events"] if e.subject == "MARKET"]
        plans.append(self._build_plan("MARKET", market_events, "session_open"))

        # 2. One plan per symbol that actually did something, most urgent first.
        symbol_events = {s: evs for s, evs in triage["by_subject"].items()
                         if s != "MARKET"}
        ordered = sorted(
            symbol_events.items(),
            key=lambda kv: -max(e.priority.rank for e in kv[1]),
        )
        for subject, events in ordered[: self.max_parallel_investigations]:
            plans.append(self._build_plan(subject, events, "anomaly"))

        # 3. Quiet names get one cheap shared plan rather than N expensive ones.
        quiet = triage["quiet"]
        if quiet:
            plans.append(self._build_quiet_plan(quiet))

        log.info("planned %d investigations from %d events (%d quiet names)",
                 len(plans), triage["n_events"], len(quiet))
        return plans

    def _build_plan(self, subject: str, events: list[DetectedEvent],
                    trigger: str) -> InvestigationPlan:
        priority = (max(events, key=lambda e: e.priority.rank).priority
                    if events else EventPriority.NORMAL)
        event_types = {e.event_type for e in events}

        budget = PlanBudget(**PRIORITY_BUDGETS[priority])
        selected, skipped = self.registry.select(
            event_types, budget=budget.max_cost, max_agents=budget.max_agents,
        )
        required = [n for n in selected if self.registry.get(n).always_run]
        optional = [n for n in selected if n not in required]

        evidence: list[str] = []
        for et in event_types:
            evidence.extend(EVIDENCE_REQUIREMENTS.get(et, []))
        evidence = list(dict.fromkeys(evidence))

        hypotheses: list[str] = []
        for e in events:
            hypotheses.extend(e.suggested_hypotheses)
        hypotheses = self._prune_hypotheses(subject, events, list(dict.fromkeys(hypotheses)))

        plan = InvestigationPlan(
            subject=subject,
            trigger=trigger,
            priority=priority,
            hypotheses=hypotheses,
            required_agents=required,
            optional_agents=optional,
            evidence_needed=evidence,
            stop_conditions=self._stop_conditions(priority, event_types),
            events=events,
            budget=budget,
        )
        if skipped:
            plan.notes.append(
                f"skipped {len(skipped)} agents not relevant to "
                f"{', '.join(sorted(t.value for t in event_types)) or 'a quiet tape'}"
            )
        return plan

    def _build_quiet_plan(self, symbols: list[str]) -> InvestigationPlan:
        budget = PlanBudget(**PRIORITY_BUDGETS[EventPriority.LOW])
        selected, _ = self.registry.select(
            {EventType.QUIET}, budget=budget.max_cost, max_agents=budget.max_agents)
        required = [n for n in selected if self.registry.get(n).always_run]
        return InvestigationPlan(
            subject=f"QUIET:{len(symbols)}",
            trigger="routine_scan",
            priority=EventPriority.LOW,
            hypotheses=["nothing is happening in these names today"],
            required_agents=required,
            optional_agents=[n for n in selected if n not in required],
            evidence_needed=["confirm the absence of a catalyst rather than assuming it"],
            stop_conditions=["no anomaly detected — lightweight pass only"],
            budget=budget,
            notes=[f"covers {', '.join(symbols[:12])}"
                   + (" …" if len(symbols) > 12 else "")],
        )

    def _stop_conditions(self, priority: EventPriority,
                         event_types: set[EventType]) -> list[str]:
        out = [
            "every required evidence item is answered, or explicitly marked unknown",
            "an additional iteration produced no new information",
        ]
        if priority.rank >= EventPriority.HIGH.rank:
            out.append("a contradiction remains unresolved after one follow-up")
        if EventType.MACRO_RELEASE in event_types:
            out.append("the release is pending — conviction stays capped until it prints")
        return out

    def _prune_hypotheses(self, subject: str, events: list[DetectedEvent],
                          hypotheses: list[str]) -> list[str]:
        """Trim the candidate list. Rules first; the model only when available.

        Rules handle the obvious cases deterministically — if an earnings event
        fired, "no identifiable catalyst" is not a live hypothesis and no model
        call is needed to work that out.
        """
        types = {e.event_type for e in events}
        pruned = list(hypotheses)

        if EventType.EARNINGS_REACTION in types:
            pruned = [h for h in pruned if "no identifiable catalyst" not in h]
        if EventType.SEC_FILING in types:
            pruned = [h for h in pruned if "no identifiable catalyst" not in h]
        if EventType.INDEX_MOVE not in types and EventType.MACRO_RELEASE not in types:
            pruned = [h for h in pruned if "repriced the entire market" not in h]

        if self.narrator and len(pruned) > 6:
            picked = self._llm_prune(subject, events, pruned)
            if picked:
                return picked
        return pruned[:8]

    def _llm_prune(self, subject: str, events: list[DetectedEvent],
                   hypotheses: list[str]) -> list[str] | None:
        """Ask the model to rank hypotheses. Best-effort and never load-bearing.

        Only hypotheses the model returns *verbatim from the supplied list* are
        accepted — it cannot invent a new explanation here, which is exactly the
        failure mode that produces confident fiction.
        """
        try:
            import json
            client = getattr(self.narrator, "_get_client", lambda: None)()
            if client is None:
                return None
            payload = {
                "subject": subject,
                "observations": [e.description for e in events],
                "candidate_hypotheses": hypotheses,
            }
            resp = client.messages.create(
                model=getattr(self.narrator, "model", "claude-opus-4-5"),
                max_tokens=500,
                system=(
                    "You rank candidate explanations for a market move. Return ONLY a "
                    "JSON array of the 3-6 most plausible hypotheses, copied VERBATIM "
                    "from candidate_hypotheses. Invent nothing. Add nothing. If the "
                    "observations do not discriminate between them, return the input "
                    "list unchanged."
                ),
                messages=[{"role": "user", "content": json.dumps(payload)[:12000]}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            start, end = text.find("["), text.rfind("]")
            if start < 0 or end < 0:
                return None
            picked = json.loads(text[start:end + 1])
            valid = [h for h in picked if h in hypotheses]     # verbatim only
            return valid[:6] or None
        except Exception as exc:  # noqa: BLE001 — pruning is an optimisation
            log.debug("LLM hypothesis pruning unavailable: %s", exc)
            return None

    # ------------------------------------------------------------------
    # execution control
    # ------------------------------------------------------------------

    def should_continue(self, plan: InvestigationPlan, evidence_graph,
                        new_evidence_this_pass: int) -> tuple[bool, StopReason | None]:
        """Decide whether another iteration is justified."""
        breach = plan.budget.exceeded()
        if breach:
            return False, breach
        if new_evidence_this_pass == 0 and plan.budget.iterations_used > 0:
            return False, StopReason.NO_NEW_INFORMATION
        if evidence_graph is not None and self.evidence_is_sufficient(plan, evidence_graph):
            return False, StopReason.SUFFICIENT_EVIDENCE
        return True, None

    def evidence_is_sufficient(self, plan: InvestigationPlan, graph) -> bool:
        """Enough independent evidence, and no open contradiction.

        Independence is what counts, not volume. Ten correlated momentum reads
        do not answer a question that one filing would.
        """
        try:
            independent = graph.effective_independent_count()
        except Exception:  # noqa: BLE001
            return False

        need = {
            EventPriority.CRITICAL: 4.0,
            EventPriority.HIGH: 3.0,
            EventPriority.ELEVATED: 2.5,
            EventPriority.NORMAL: 2.0,
            EventPriority.LOW: 1.0,
        }[plan.priority]

        if independent < need:
            return False
        if graph.contradictions() and plan.budget.iterations_used < 1:
            return False       # one pass to try to resolve it
        return True

    def followup_plan(self, parent: InvestigationPlan, questions: list[str],
                      depth: int = 1) -> InvestigationPlan | None:
        """A bounded child investigation. Returns None past the depth limit."""
        if depth > self.max_followup_depth:
            log.info("follow-up depth limit reached for %s", parent.subject)
            return None
        if parent.budget.exceeded():
            return None

        remaining_cost = max(1.0, parent.budget.max_cost - parent.budget.cost_used)
        agents = self._agents_for_questions(questions)
        budget = PlanBudget(
            max_iterations=1,
            max_agents=max(1, min(4, parent.budget.max_agents - parent.budget.agents_used)),
            max_cost=min(remaining_cost, 12.0),
            max_seconds=max(15.0, parent.budget.max_seconds - parent.budget.elapsed),
        )
        return InvestigationPlan(
            subject=parent.subject,
            trigger=f"followup:{parent.id}",
            priority=parent.priority,
            hypotheses=[],
            required_agents=[],
            optional_agents=agents,
            evidence_needed=list(questions),
            stop_conditions=["answer the specific questions or report them unanswered"],
            budget=budget,
            notes=[f"follow-up depth {depth} for {parent.id}"],
        )

    def _agents_for_questions(self, questions: list[str]) -> list[str]:
        """Map plain-English evidence requests onto capabilities.

        Keyword routing rather than a model call: the vocabulary is small,
        fixed, and defined by us, so a lookup is both cheaper and more reliable.
        """
        text = " ".join(questions).lower()
        picks: list[str] = []
        routes = [
            (("filing", "8-k", "sec", "insider", "13d"), ["sec_filings", "institutional"]),
            (("earnings", "guidance", "consensus", "eps"), ["earnings"]),
            (("news", "headline", "catalyst", "corroborat"), ["breaking_news"]),
            (("option", "implied move", "hedge", "iv", "skew"), ["options_flow"]),
            (("volume", "participation", "rvol"), ["overnight_scan"]),
            (("sector", "index", "beta", "relative"), ["global_markets", "futures"]),
            (("vix", "volatility", "term structure"), ["volatility_regime"]),
            (("level", "support", "resistance", "technical"), ["technicals"]),
            (("macro", "release", "cpi", "fomc", "payroll"), ["econ_calendar"]),
        ]
        for keys, agents in routes:
            if any(k in text for k in keys):
                picks.extend(a for a in agents if a in self.registry)
        return list(dict.fromkeys(picks))

    # ------------------------------------------------------------------

    def detect_contradictions(self, reports: dict) -> list[str]:
        """Find specialists that disagree, so the disagreement is surfaced
        rather than averaged away."""
        out: list[str] = []
        bulls, bears = [], []
        for name, rep in reports.items():
            for s in getattr(rep, "signals", []) or []:
                if s.probability > 0.55:
                    bulls.append((name, s))
                elif s.probability < 0.45:
                    bears.append((name, s))
        for bn, bs in bulls:
            for rn, rs in bears:
                if abs(bs.probability - rs.probability) > 0.18:
                    out.append(
                        f"{bn}:{bs.name} ({bs.probability:.0%}) contradicts "
                        f"{rn}:{rs.name} ({rs.probability:.0%})"
                    )
        return out[:6]
