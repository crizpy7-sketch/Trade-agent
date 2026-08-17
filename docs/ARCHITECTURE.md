# MarketSwarm 2.0 — Architecture

Research and decision-support only. No order execution, no funds movement.

## The pipeline

```
                         MARKETSWARM 2.0
                                |
                          EVENT BRAIN            cheap deterministic scan
                                |                classify + prioritise
                      CHIEF INVESTIGATOR         what needs investigating?
                                |                select agents, set budgets
                    INVESTIGATION PLANNER        bounded plans
                                |
        +-----------------------+-----------------------+
        |                       |                       |
   MARKET AGENTS          EVENT AGENTS           COMPANY AGENTS
   futures, global,       news, econ,            earnings, filings,
   volatility             options                institutional
        |                       |                       |
        +-----------------------+-----------------------+
                                |
                        EVIDENCE GRAPH            nodes, edges, 6 score dims,
                                |                 correlation clusters
                     FORECAST / SYNTHESIS         logit fusion + haircut
                                |
                   RECOMMENDATION ENGINE          conviction incl. "don't know"
                                |
                           RISK AGENT
                                |
                            RED TEAM              adversarial findings
                                |
                          REVIEW GATE  ←──────┐   APPROVE / REDUCE / MODIFY
                                |             │   / RESEARCH / REJECT
                +---------------+--------+    │
                |                        |    │
            APPROVED            MORE RESEARCH─┘   (bounded, max 2 iterations)
                |
                v
        FINAL RECOMMENDATION → LIVE MONITORING → OUTCOME
                                                    |
                                        AFTER-ACTION REVIEW
                                                    |
                                          LEARNING ENGINE
                                     (contribution, not agreement)
                                                    |
                                     INSTITUTIONAL MEMORY
                                   (7 categories + mistakes)
                                                    |
                                     EXPERIMENT LABORATORY
                                    CHAMPION / CHALLENGER
                                   (human approval to promote)
```

## Deterministic vs LLM

The durable intelligence lives in data, evaluation and tests — not in one
model. The model is replaceable; the evidence is not.

| Deterministic (no LLM) | LLM |
|---|---|
| anomaly detection, thresholds | investigation planning |
| agent routing, budgets | hypothesis pruning (ranking only) |
| probability fusion, calibration | contradiction resolution |
| bracket construction, EV, Kelly | red-team reasoning |
| review-gate policy | narrative explanation |
| scoring, migrations, persistence | research hypothesis generation |
| retries, permissions | post-mortem |

`ModelRouter` enforces this. Anything absent from `TASK_TIERS` defaults to
deterministic, so spending money is an explicit opt-in. Budgets are enforced,
and every LLM path degrades to a deterministic one when the model is
unavailable.

## Modules

| Module | Responsibility |
|---|---|
| `investigation/event_brain.py` | detect and prioritise anomalies |
| `investigation/chief.py` | build bounded investigation plans |
| `investigation/registry.py` | agent capabilities and team selection |
| `investigation/plan.py` | plans and enforced budgets |
| `evidence/graph.py` | evidence nodes, edges, scoring, independence |
| `review/gate.py` | binding review decisions |
| `review/loop.py` | generate → attack → revise, bounded |
| `recommend/engine.py` | conviction and recommendation types |
| `memory/institutional.py` | 7 memory categories + mistakes |
| `memory/contribution.py` | contextual, ablation-based learning |
| `memory/migrations.py` | versioned schema |
| `experiments/lab.py` | champion/challenger with promotion gates |
| `experiments/scientist.py` | propose-only research agent |
| `after_action.py` | post-outcome review and lesson extraction |
| `observability.py` | traces, costs, structured logs |
| `resilience.py` | circuit breakers, degradation levels |
| `security.py` | redaction, injection defence, allowlists |
| `llm_router.py` | tiered, budgeted model routing |
| `api.py` | read-only query surface |
| `pipeline2.py` | wires all of the above onto the 1.x swarm |

## Preserved from 1.x

Unchanged and still load-bearing: `clock.py`, `stats/*`, `providers/*`,
`agents/*`, `backtest/*`, `memory/store.py`, `report.py`, `monitor.py`.
The 2.0 layers wrap them; they were not rewritten.
