# MarketSwarm 2.0 — Integration Acceptance Matrix

"Implemented" means the class exists. "On the production path" means
`marketswarm run` or `marketswarm score` reaches it without a flag. "E2E
tested" means a test in `tests/test_control_path.py` or
`tests/test_review_loop_integration.py` drives the real entry point and asserts on the observable result — instantiation, a database row, or
rendered output — not on a mock.

A component is only complete when all three are yes.

Legend: **Y** yes · **N** no · **—** not applicable

---

## Detection and routing

| Capability | Implemented | On production path | E2E tested | Fallback behaviour | Authoritative source of truth |
|---|---|---|---|---|---|
| Event Brain | Y | Y — `_execute_swarm` step 3, before specialists | Y — `test_critical_event_expands_the_swarm_within_budget` | No events → quiet plan, scan agents only | `investigation/event_brain.py` |
| Chief Investigator | Y | Y — plans *before* execution | Y — `test_chief_plans_before_specialists_execute` | Plan failure → scan + decision agents only | `investigation/chief.py::InvestigationPlan` |
| Capability Registry | Y | Y — via `chief._build_plan` → `registry.select` | Y — `test_dynamic_routing_skips_agents_that_never_execute` | `always_run` set is never dropped | `investigation/registry.py::DEFAULT_CAPABILITIES` |
| Dynamic routing | Y | Y — default mode | Y — asserts on `__init__` spies, 8 of 16 skipped on a quiet tape | `full` mode available for benchmarking | `ControlPathTrace.agent_execution_reason` |
| Budget enforcement | Y | Y — `PlanBudget` per plan | Y — `test_critical_event_expands_the_swarm_within_budget` | `StopReason` ends the investigation | `investigation/plan.py::PlanBudget` |
| Circuit breakers | Y | Y — consulted in `_run_agents`, fed after every agent | Partial — wiring covered, open-state path not forced in a test | Open circuit → agent skipped with a stated reason | `resilience.py::BREAKERS` |

## Evidence and recommendation

| Capability | Implemented | On production path | E2E tested | Fallback behaviour | Authoritative source of truth |
|---|---|---|---|---|---|
| Evidence Graph | Y | Y — `Pipeline2.build_graph` | Y — `test_followup_research_grows_the_evidence_graph` asserts node **count and version both increase**, with provenance | Empty graph → `effective_independent_evidence` 0 → `INSUFFICIENT_EVIDENCE` | `evidence/graph.py::EvidenceGraph` |
| Correlation discount | Y | Y — `effective_independent_count` feeds every recommendation | Y — via conviction assertions | Assumed ρ priors, not estimates (see Limitations) | `evidence/graph.py::CLUSTERS` |
| Recommendation Engine | Y | Y — the only builder of `Recommendation` | Y — `test_report_renders_with_playbook_and_disclaimer` | Ignorance checked before edge | `recommend/engine.py::Recommendation` |
| Red Team | Y | Y — `DECISION_AGENTS`, always runs; **critical to publication** | Y — `test_red_team_objections_reach_the_report` | Absent red team → deterministic fallback, else suppression. **Never APPROVE.** | `agents/redteam.py` |
| Review Gate | Y | **Y — authoritative** | Y — `test_rejected_recommendation_can_never_reenter_publication_path` | Gate failure ⇒ suppression, never bypass | `review/gate.py::ReviewDecision` |
| Revision loop | Y | Y — bounded by `max_iterations=2` | Y — `test_modified_recommendation_replaces_the_original` | No-progress guard terminates | `review/loop.py::ReviewOutcome` |
| Follow-up research | Y | Y — `Swarm._make_investigator` runs real agents | Y — `test_request_more_research_executes_a_real_followup` | No fresh agent → questions published unanswered | `investigation/chief.py::followup_plan` |
| **Follow-up evidence ingestion** | Y | Y — `CandidateReview.refresh_evidence` rebuilds the graph before the next round | Y — `test_followup_research_grows_the_evidence_graph` (count, version, provenance) | Nothing produced → `followup_failed`, HIGH finding injected | `CandidateReview.graph_version` |
| **Fresh Red Team after follow-up** | Y | Y — the investigator re-runs `cross_verify` then `red_team` | Y — `test_followup_triggers_a_fresh_red_team_pass` counts agent invocations | Refresh fails → `red_team_failures`, stale objections never silently reused | `review_rounds.red_team_execution_status` |
| **Red Team execution status** | Y | Y — `redteam_execution_status` on every round | Y — `test_gate_rejects_every_untrustworthy_execution_status` covers all six | Untrustworthy → `REVIEW_INCOMPLETE` | `review/gate.py::ReviewExecutionStatus` |
| **Review-failure suppression** | Y | Y — `Pipeline2.run` suppresses the whole publication | Y — `test_red_team_failure_suppresses_publication` checks report, JSON, DB, API, webhook, monitor | Deterministic fallback first; then suppress | `PublicationSet.suppression_reason` |
| **Deterministic fallback reviewer** | Y | Y — used when the agent does not complete | Y — `test_red_team_failure_falls_back_to_deterministic_review` | Produces nothing → suppress | `review/fallback.py` |
| **Review round history** | Y | Y — persisted per round | Y — `test_each_round_is_preserved_not_overwritten` | Round 1 never overwritten | `review_rounds` table |
| **Event-aware contextual routing** | Y | Y — stage 2 resolves after the Event Brain | Y — `test_event_specific_weights_change_which_agents_execute` asserts **execution**, not weights | Unclassified → `event=any`, hierarchical backoff | `ControlPathTrace.event_context_resolved` |
| **Event context persisted for learning** | Y | Y — `features.event_type` written explicitly | Y — `test_event_context_is_persisted_for_later_learning` | Absent → falls back to `setup` | `predictions.features` |
| **Candidate/recommendation lineage** | Y | Y — `assert_no_leak` matches on `candidate_id` | Y — `test_leak_check_uses_candidate_lineage_not_recommendation_id` with `rec_456` / `cand_123` | Also checks `revision_parent_id` | `publication.py::assert_no_leak` |

## Publication and persistence

| Capability | Implemented | On production path | E2E tested | Fallback behaviour | Authoritative source of truth |
|---|---|---|---|---|---|
| Publication authority | Y | Y | Y — 8 separate leak assertions | `suppressed_set()` publishes nothing | `publication.py::PublicationSet` |
| Legacy `ideas` view | Y | Y — read-only `@property` | Y — `test_result_ideas_cannot_be_assigned` | Derived, never assigned | `publication._payload_from` |
| Reports (MD + HTML) | Y | Y | Y | Suppression renders "Recommendations withheld" | `report.py` |
| JSON output | Y | Y | Y — assertion 5 | — | `cli.py::cmd_run` |
| Notifications | Y | Y | Y — assertion 8 | Suppression states it; `high_confidence` policy sends nothing | `notify.py::summarize` |
| Persistence: predictions | Y | Y — from `result.ideas` only | Y — assertion 6 | Nothing published ⇒ nothing persisted | `predictions` table |
| Persistence: recommendations | Y | Y — approved **and** rejected, by `status` | Y — assertion 7 + audit | — | `recommendations.status` |
| Status lifecycle | Y | Y | Y | `CANDIDATE/APPROVED/MODIFIED/REJECTED/SUPPRESSED/EXPIRED/INVALIDATED/RESOLVED` | `publication.py::RecordStatus` |
| Lineage | Y | Y — written on every recommendation | Y — audit assertions | `candidate_id`, `evidence_graph_id`, `review_decision_id`, `revision_parent_id` | migration 005 |
| Monitoring | Y | Y — reads `predictions` | Y — `test_scenario_i_...` | Rejected ideas are absent by construction | `monitor.py::InvalidationMonitor` |
| Read-only API | Y | Y | Y — `test_scenario_i_...` | `ACTIVE_STATUSES` filter; `rejected()` is separate | `api.py::ReadOnlyAPI` |

## Learning

| Capability | Implemented | On production path | E2E tested | Fallback behaviour | Authoritative source of truth |
|---|---|---|---|---|---|
| Scoring | Y | Y — `cmd_score` | Y (1.x tests) | — | `memory/learning.py::LearningEngine` |
| AfterActionReviewer | Y | Y — `ClosedLoop.run`, automatic | Y — `test_after_action_review_runs_as_part_of_scoring` | One bad row is logged and skipped | `after_action_reviews` table |
| ContributionTracker | Y | Y — `ClosedLoop.run` | Y — `test_learning_from_session_a_changes_session_b` | <12 observations ⇒ no weight change | `agent_context_scores` table |
| Contextual weights | Y | Y — `_resolve_agent_weights` on every run | Y — asserts next-run behaviour changes | Resolver failure ⇒ baseline Brier weights | `memory/resolver.py::ContextualWeightResolver` |
| Regime separation | Y | Y | Y — `test_scenario_d_contextual_learning_separates_regimes` | Backs off to broader context | `ContextKey.generalisations()` |
| Institutional memory | Y | Y — recalled in `_prior_failures`, written by the loop | Y — `test_prior_mistake_is_retrieved_for_a_later_investigation` | Recall failure is logged, never fatal | `memories` table |
| Mistake taxonomy | Y | Y — written by the reviewer | Y — same test | — | `mistakes` table |
| Idempotent learning | Y | Y | Y — second `ClosedLoop.run` reviews 0 | LEFT JOIN on `after_action_reviews` | `closed_loop._newly_resolved` |
| Research Scientist | Y | Y — thresholded (≥30 resolved, ≥5 recurrences) | Y — `test_closed_loop_registers_challengers_but_never_promotes` | Below threshold ⇒ silent | `experiments/scientist.py` |
| Experiment Lab | Y | Y — receives challengers | Y — same test | Registration only | `experiments/lab.py` |
| **Promotion stays human** | Y | **Deliberately not automated** | Y — `test_research_scientist_cannot_promote_itself` | No automatic path exists | `ExperimentLab.promote` |

## Cross-cutting

| Capability | Implemented | On production path | E2E tested | Fallback behaviour | Authoritative source of truth |
|---|---|---|---|---|---|
| Observability | Y | Y — `run_control_path` row per run | Y — `test_control_path_trace_is_persisted_...` | — | `ControlPathTrace` |
| Degradation | Y | Y | Y — `test_scenario_h_...` | Suppresses the affected output class | `resilience.py::DegradationTracker` |
| LLM routing | Y | Y — `Narrator._route_for` | Partial — routing verified directly, not through a live API call | No key ⇒ deterministic, `downgraded=True` | `llm_router.py::ModelRouter` |
| Security boundary | Y | Y | Y — `test_no_trading_capability_was_introduced` | — | `security.py::FORBIDDEN_CAPABILITIES` |
| Migrations | Y | Y — on `Swarm.__init__` | Y (existing tests) | Newer schema ⇒ refuses to act | `memory/migrations.py` |

---

## Not complete, stated plainly

| Item | Status | Why |
|---|---|---|
| Circuit-breaker open path | Wired, not E2E tested | The wiring and the skip reason are covered; no test yet forces a breaker open across a full run |
| Red-team refresh *inside* one candidate's loop | Wired, tested at run level | The re-run is triggered once per follow-up and shared across candidates in that run, not re-run per candidate |
| Sector context | Column exists, always `any` | The event brain does not classify sector; the column is reserved rather than populated |
| LLM routing under a live key | Wired, not E2E tested | Would require a real API call in CI |
| `Observatory.trace_agent` | Available, unused by the orchestrator | Per-agent DB traces are written by `record_report`; the finer-grained context manager is not on the path |
| Cluster correlation priors | Assumed, not estimated | `CLUSTERS` holds defensible priors, not measurements from realised data |
| Any claim of predictive edge | **Not made** | Deflated Sharpe 0.004. Nothing in this release changes that, and none of it has been validated forward |
