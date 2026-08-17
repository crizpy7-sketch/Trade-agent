# Runtime Control Path

What actually executes, in order, with the responsible module named at every
step. If a claim in another document contradicts this one, this one is wrong
too — go read the code. The point of this file is to make a bypass obvious.

Verify any of it on a live install:

```bash
marketswarm run --json | jq .control_path
sqlite3 ~/.marketswarm/marketswarm.db \
  "SELECT orchestration_mode, agents_executed, agents_skipped, \
          recommendations_rejected, legacy_fallback_used \
   FROM run_control_path ORDER BY run_id DESC LIMIT 5;"
```

---

## Part 1 — `marketswarm run` to publication

### 0. Entry

| Step | Module | Function |
|---|---|---|
| CLI parse | `cli.py` | `cmd_run` |
| Calendar refusal | `clock.py` | `day_status` — returns early unless `--force` |
| Mode resolution | `orchestrator.py` | `Swarm.run` — `--mode` › `config.orchestration_mode` › `dynamic` |
| Schema | `memory/migrations.py` | `migrate` on `Swarm.__init__` |

### 1. Load prior learning

`Swarm._resolve_agent_weights` → `memory/resolver.py::ContextualWeightResolver`

Contextual contribution first, global Brier weight as fallback, 1.0 for
anything never measured. This is where the previous session's scoring enters
this session.

**Called twice, deliberately.** Stage 1 runs here, before the scan, with the
previous session's regime and no event type. Stage 2 runs in §4 below, once the
Event Brain has classified today — and *that* is the resolution that routes.
Resolving only once, before detection, meant the store could hold "options_flow
is worth 1.6 around earnings" and the run could never ask for it, because at
resolution time it did not yet know it was an earnings day.

The stage-2 dict is used for both routing (via the capability registry) and
fusion (via `ctx.agent_weights`), because two independent opinions about the
same agent is how a system contradicts itself.

Also loaded: `learning.load_calibrator()`, `store.active_lessons()`.

### 2. Cheap broad scan

`Swarm._execute_swarm` → `Swarm._run_agents(SCAN_AGENTS)`

```
SCAN_AGENTS = overnight_scan, futures, volatility_regime, technicals
```

Foundational, `always_run` in the registry, and enough for the Event Brain to
see gaps, index moves, volatility and relative volume. Circuit breakers are
consulted first (`resilience.BREAKERS.open_circuits()`), so an agent whose
provider has been failing is skipped rather than retried into a timeout.

### 3. Detect

`orchestrator._snapshot_from` → `Pipeline2.build_snapshot` (a staticmethod, so
there is exactly one snapshot builder) → `investigation/event_brain.py::EventBrain`

Deterministic detectors only. No model call decides whether something happened.

### 3b. Resolve the event context, then route on it

`EventBrain.scan(snapshot)` → `orchestrator._dominant_event_type(events)` →
`_resolve_agent_weights(regime, event_type)` (stage 2).

The dominant event type is the highest-priority one detected, ties broken by
count, and `"any"` when nothing was classified — an honest absence rather than
a guessed label, so the resolver falls through its normal hierarchy instead of
routing on a context that was never observed.

Recorded as `ControlPathTrace.event_context_resolved`.

### 4. Plan

`investigation/chief.py::ChiefInvestigator.plan_session(snapshot, agent_weights)`

Produces `InvestigationPlan`s with hypotheses, priority and a `PlanBudget`
(iterations, agents, cost units, seconds) taken from `PRIORITY_BUDGETS`.

### 5. Select

`investigation/registry.py::CapabilityRegistry.select(event_types, budget, max_agents)`

Called from inside `_build_plan`. `update_reliability(agent_weights)` is applied
first, so a learned weight changes what gets run. Returns `(selected, skipped)`
and pulls in declared dependencies.

An optional agent whose measured reliability has fallen to
`registry.UNRELIABLE_THRESHOLD` (0.25) is **excluded**, not merely ranked last.
Before that, a learned weight could only change the order in which agents were
considered — and since the budget rarely binds on a busy session, ordering
changed nothing at all. `always_run` agents are never excluded on reliability.

### 6. Execute the selected specialists

`Swarm._run_agents(specialists)`

**Agents that were not selected are never constructed.** Not run-and-discard,
not run-with-a-flag: `_run_agents` filters `ALL_AGENTS` by name before
instantiating anything. `tests/test_control_path.py` spies on `__init__` to
prove it.

### 7. Synthesis — always

```
DECISION_AGENTS = cross_verify, risk, playbook, red_team
```

These always run. Without them there is no candidate and nothing to review.

### 8. Decide

`Swarm._decide` → `pipeline2.py::Pipeline2.run` (on a worker thread via
`asyncio.to_thread`, so its synchronous review state machine can schedule real
agent work back onto the event loop)

```
build_graph            evidence/graph.py     clustered, ρ-discounted
build_recommendations  recommend/engine.py   ignorance checked before edge
  └─ per candidate:
       review/loop.py::ReviewLoop.run(idea, critic, investigator)
            critic       = pipeline2.py::CandidateReview.critic
                           ← re-reads the CURRENT graph and the CURRENT
                             red-team report every round; nothing is cached
            investigator = pipeline2.py::CandidateReview.investigate
                           → Swarm._make_investigator  ← runs real agents,
                             then re-runs cross_verify + red_team
            gate         = review/gate.py::ReviewGate.review
```

The gate's five outcomes:

| Status | Effect |
|---|---|
| `APPROVE` | published as-is |
| `APPROVE_WITH_REDUCED_CONFIDENCE` | `gate.apply` rewrites confidence, then published |
| `MODIFY` | modifications applied, then published |
| `REQUEST_MORE_RESEARCH` | `Swarm._make_investigator` asks `chief.followup_plan` which agents answer the open questions and runs the ones that have not run. If any produce usable evidence, the graph is **rebuilt** (`graph_version` increments) and `cross_verify` + `red_team` are **re-run**, so the next round argues against the new picture. If none do, `followup_failed` is set and the next round receives a HIGH finding saying the demanded corroboration never arrived. |
| `REJECT` | `outcome.final_idea is None` → a `RejectedCandidate`, never a recommendation |
| `REVIEW_INCOMPLETE` | the reviewer did not complete — see below. Not a verdict, so the candidate is stored `SUPPRESSED`, not `REJECTED`, and the whole publication is suppressed. |

**Adversarial review must succeed.** `redteam_execution_status(report)` classifies
every round as `COMPLETED`, `COMPLETED_BY_FALLBACK`, `FAILED`, `UNAVAILABLE`,
`TIMED_OUT` or `INVALID`. `ReviewGate.review` treats an empty findings list as
clean **only** for the first two. When the agent does not complete,
`review/fallback.py::deterministic_review` runs the structural checks directly
from the reports; if that also yields nothing, publication is suppressed with a
stated reason. `red_team` is in `resilience.CRITICAL_AGENTS`.

Full state machine, including every failure case:
`docs/REVIEW_AND_FOLLOWUP_STATE_MACHINE.md`.

### 9. The publication authority

`publication.py::PublicationSet`

Built by `Pipeline2.run` and immediately checked by `assert_no_leak()`, which
raises `PublicationError` if:

- an active recommendation's **`candidate_id`** appears among the rejected
  candidate ids (matching on `rec.id` instead was a no-op, because `rec_...`
  and `cand_...` are different namespaces and never collide),
- its `revision_parent_id` descends from a rejected candidate,
- an id is duplicated, or
- the legacy view disagrees with its own recommendation about confidence.

**Everything downstream reads this object and nothing else.**

```
result.publication          the authority
result.ideas                a read-only @property → publication.ideas_view()
result.recommendations      → publication.active()
```

`SwarmResult.ideas` has **no setter**. The pre-fix bug was one assignment
statement, so the attribute no longer accepts one; the test
`test_result_ideas_cannot_be_assigned` pins that.

### 10. Consumers

| Consumer | Module | Reads |
|---|---|---|
| Markdown report | `report.py::render_markdown` | `result.publication` + `result.ideas` |
| HTML report | `report.py::render_html` | same |
| JSON | `cli.py::cmd_run` | `publication.active()`, `.rejected`, `.summary()`, `trace` |
| Webhook | `notify.py::summarize` | `result.ideas`, suppression state |
| Narrative | `llm.py::Narrator` | `result.ideas`; model tier chosen by `llm_router.ModelRouter` |
| Predictions table | `orchestrator._persist_predictions` | `result.ideas` |
| Recommendations table | `orchestrator._persist_recommendations` | approved **and** rejected, distinguished by `status` |
| Control-path trace | `orchestrator._persist_trace` | `result.trace` |

### 11. Failure and degradation

| Failure | Result |
|---|---|
| Pipeline2 raises | `PublicationSet.suppressed_set(...)` — **nothing published**. The pre-review playbook does not stand in. |
| Critical agent lost | `resilience.DegradationTracker` → conviction forced to `INSUFFICIENT_EVIDENCE`, `suppressed_outputs()` named in the report |
| Every candidate rejected | empty publication; `may_publish_index_call()` returns False, so the session-level SPY call is not filed either |
| Provider repeatedly failing | `resilience.BREAKERS` opens; the agent is skipped with a stated reason |
| `mode = legacy` | analysis only. Publication suppressed unless `allow_unreviewed_publication = true`, and then every recommendation is stamped `review_status = NOT_REVIEWED` |

There is no path from a 2.0 failure to a published unreviewed idea.

---

## Part 2 — `marketswarm score` to the next session

### 1. Resolve

`cli.py::cmd_score` → `memory/learning.py::LearningEngine.resolve_prediction`
→ `store.resolve(...)`. Unchanged from 1.x, and still correct.

### 2. Baseline statistics

`LearningEngine.score_and_learn()` — Brier, log loss, skill, Platt
recalibration, global agent weights. Its job is baseline reliability and
calibration; it keeps it.

### 3. The closed loop

`closed_loop.py::ClosedLoop.run()`, called from `cmd_score`:

```
_newly_resolved()          resolved predictions with no after-action review yet
                           (the LEFT JOIN is what makes re-running idempotent —
                            learning twice from one outcome is double-counting)
    ↓
after_action.py::AfterActionReviewer.review + .persist
    → after_action_reviews, mistakes (11-code taxonomy), memories
    ↓
memory/contribution.py::ContributionTracker.update_from_resolved
    → agent_context_scores, per (regime, event_type, horizon)
    → MIN_OBSERVATIONS=12, shrinkage at n=30, floor 0.15, ceiling 1.75,
      max step 0.25 per update
    ↓
memory/institutional.py::InstitutionalMemory.prune()   TTL expiry
    ↓
_maybe_propose()   only when ≥30 resolved AND a taxonomy code recurs ≥5 times
    → ResearchScientist.propose → register_proposals → ExperimentLab
```

`ClosedLoop` has no promotion path. `ResearchScientist` has no `promote`,
`deploy`, or `apply_to_production` method at all — asserted by
`test_research_scientist_cannot_promote_itself`. Promotion is
`ExperimentLab.promote`, gated and human-approved.

### 4. Back into the next run

`agent_context_scores` → `ContextualWeightResolver` → §1 of Part 1.

The loop is closed. `test_learning_from_session_a_changes_session_b` asserts on
the resolved weight changing, and on the agent that helped outranking the one
that hurt — not on a row existing.

---

## Where a bypass would show up

If someone reintroduces one, these are the tripwires:

1. `PublicationSet.assert_no_leak()` raises on every run.
2. `SwarmResult.ideas` has no setter — an assignment is an `AttributeError`.
3. `run_control_path.legacy_fallback_used` is recorded per run.
4. `agent_execution_reason` carries a reason for every one of the 16 agents.
5. `recommendations.status` distinguishes `APPROVED` from `REJECTED`; the API's
   `ACTIVE_STATUSES` filter means a rejected row cannot be served as live.
6. `tests/test_control_path.py` and `tests/test_review_loop_integration.py` —
   56 tests that drive `Swarm.run()` and assert on instantiation, database
   rows and rendered output.
7. `run_control_path.graph_versions == 1` on a run with
   `followup_requests > 0` means follow-up research bought evidence that never
   reached a second round — the 2.0.1 stale-loop bug returning.
8. `review_rounds.red_team_execution_status` is recorded per round. Anything
   other than `completed` / `completed_by_fallback` must coincide with a
   suppressed publication.
9. `run_control_path.event_context_resolved` shows which (regime, event) the
   session actually routed on. `event=any` on a session with detected events
   means stage-2 resolution did not happen.
