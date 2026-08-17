# Chief Investigator

Sits above the specialist swarm and answers the question 1.x never asked:
*what needs investigating right now?*

## Division of labour

**Deterministic:** which events fired, which agents address them, what the
budget allows, when to stop. Testable, reproducible, free.

**LLM (optional):** narrowing a long hypothesis list, reading a contradiction
rules cannot resolve. The Chief works fully without it — the model sharpens the
plan and is never load-bearing.

## Effort allocation

| Priority | Iterations | Agents | Cost units | Seconds |
|---|---|---|---|---|
| CRITICAL | 3 | 14 | 40 | 180 |
| HIGH | 2 | 11 | 28 | 120 |
| ELEVATED | 2 | 9 | 20 | 90 |
| NORMAL | 1 | 7 | 12 | 60 |
| LOW | 1 | 5 | 6 | 30 |

A quiet AAPL no longer costs the same as an NVDA earnings gap. Quiet names
share **one** cheap plan rather than getting one each.

## Bounded recursion

`PlanBudget.exceeded()` is checked every iteration and returns a `StopReason`.
Follow-up investigations are depth-limited (`max_followup_depth`, default 1)
and inherit the parent's remaining budget. `max_iterations=0` raises.

Stop reasons: `SUFFICIENT_EVIDENCE`, `NO_NEW_INFORMATION`, `ITERATION_LIMIT`,
`COST_LIMIT`, `TIME_LIMIT`, `AGENT_LIMIT`, `CONTRADICTION_UNRESOLVED`,
`INSUFFICIENT_DATA`.

## Sufficiency

`evidence_is_sufficient()` measures **effective independent** evidence, not
volume: 4.0 for CRITICAL down to 1.0 for LOW, plus one pass to resolve any
contradiction. Ten correlated momentum reads do not answer a question one
filing would.

## Hypothesis safety

`_llm_prune` accepts only hypotheses returned **verbatim from the supplied
list**. The model cannot invent an explanation here —
`test_hypothesis_pruning_cannot_introduce_new_hypotheses`.
