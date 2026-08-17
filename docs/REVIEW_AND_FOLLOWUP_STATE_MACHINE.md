# The Review and Follow-Up State Machine

How a candidate becomes a published recommendation, a rejection, or a
suppression — and what happens when the machinery itself fails.

Everything here is executable. `tests/test_review_loop_integration.py` drives
the real `Swarm.run()` and asserts each transition.

---

## The states

```
                          ┌─────────────┐
                          │  CANDIDATE  │  playbook built it; nothing has
                          └──────┬──────┘  looked at it yet
                                 │
                     ┌───────────▼───────────┐
                     │  ROUND n: ADVERSARIAL │  CandidateReview.critic()
                     │        REVIEW         │  reads the CURRENT graph and
                     └───────────┬───────────┘  the CURRENT red-team report
                                 │
              ┌──────────────────┼──────────────────┬────────────────┐
              │                  │                  │                │
     execution_status      COMPLETED /         COMPLETED /      COMPLETED /
     NOT trustworthy       no findings         findings         findings
              │                  │                  │                │
              ▼                  ▼                  ▼                ▼
     ┌────────────────┐   ┌───────────┐   ┌──────────────┐   ┌────────────┐
     │REVIEW_INCOMPLETE│  │  APPROVE  │   │  REDUCE /    │   │   REJECT   │
     │                │   │           │   │  MODIFY /    │   │            │
     │  SUPPRESSED    │   │ published │   │  REQUEST     │   │  audit only│
     └────────────────┘   └───────────┘   └──────┬───────┘   └────────────┘
                                                 │
                                    REQUEST_MORE_RESEARCH
                                                 │
                                    ┌────────────▼────────────┐
                                    │  FOLLOW-UP RESEARCH     │
                                    │  chief.followup_plan()  │
                                    │  → run selected agents  │
                                    └────────────┬────────────┘
                                                 │
                          ┌──────────────────────┴──────────────────────┐
                          │                                             │
                  agents produced                              nothing produced
                  usable evidence                              (or they failed)
                          │                                             │
                          ▼                                             ▼
              ┌───────────────────────┐                  ┌──────────────────────┐
              │ EVIDENCE REFRESH      │                  │ followup_failed=True │
              │ graph v(n) → v(n+1)   │                  │ next round gets a    │
              │ recompute independence│                  │ HIGH finding:        │
              │                       │                  │ "corroboration was   │
              │ ADVERSARIAL REFRESH   │                  │  demanded and could  │
              │ re-run cross_verify   │                  │  not be obtained"    │
              │ re-run red_team       │                  └──────────┬───────────┘
              └───────────┬───────────┘                             │
                          └──────────────────┬──────────────────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  ROUND n+1      │  (loop, capped at
                                    └─────────────────┘   max_iterations)
```

---

## Why round 2 used to be pointless

Before 2.0.2 the critic was a closure:

```python
rt = reports.get("red_team")            # captured once, before the loop
def critic(current, iteration, _rt=rt, _sym=symbol):
    return findings_from_redteam_report(_rt, symbol=_sym)
```

and the evidence graph was built once per session, outside the per-candidate
loop. Follow-up agents really did run and really did write into `ctx.reports`.
Round 2 then re-derived its findings from `_rt` — the same object — and
produced byte-identical objections. The loop's own no-progress guard saw the
identical signature and terminated.

Measured on the fixture before the fix:

```
follow-up agents executed: ['sec_filings', 'institutional']
round-0 findings: ('4 signals collapse to 1.9 independent ones…',
                   'Every stock setup is long…')
round-1 findings: ('4 signals collapse to 1.9 independent ones…',
                   'Every stock setup is long…')
IDENTICAL across rounds: True
```

The research was bought and thrown away, and it was guaranteed to be by
construction — no market condition could have made it otherwise.

After the fix, same probe:

```
follow-up: requests=1 executed=2 failed=0
red team:  attempts=1 successes=1
graph:     v2, nodes 19 → 21
QQQ round 1: graph v1, 19 nodes, 2 findings, completed
QQQ round 2: graph v2, 21 nodes, 2 findings, completed
```

Two nodes arrived, the graph versioned, and the red team ran a second time.
The objection count is unchanged here because on this fixture the new filings
do not answer the objection ("every setup is long" is still true) — which is
the correct outcome, not a stale one. `test_new_evidence_can_change_the_verdict`
and `test_contradictory_followup_evidence_changes_the_verdict` cover the cases
where the evidence does bear on the objection, in both directions.

---

## Evidence refresh: rebuild, not incremental

`CandidateReview.refresh_evidence()` rebuilds the graph from the live reports
via `Pipeline2.build_graph`, rather than inserting new nodes into the existing
one.

`build_graph` is already a pure function of the report set. A second,
incremental insertion path would be a second definition of what the graph
means, and the two would eventually disagree — most likely about correlation
clusters, which is exactly where a disagreement would be hardest to notice.
Rebuilding costs a few hundred node constructions on a session that has
already decided to spend money on more research.

Each rebuild increments `graph_version`. Every round records the version it
saw, and the published recommendation cites the version of its **final** round.
A round-2 record carrying `graph_version: 1` would be the stale-loop bug
returning, and is directly assertable.

---

## Review execution status

The axis that 2.0.1 was missing entirely.

| `ReviewExecutionStatus` | Meaning | Empty findings means |
|---|---|---|
| `COMPLETED` | the agent ran and produced a verdict | reviewed, nothing found → **may approve** |
| `COMPLETED_BY_FALLBACK` | the deterministic reviewer stood in | reviewed structurally → **may approve, labelled degraded** |
| `FAILED` | the agent raised | **nothing** |
| `UNAVAILABLE` | never ran, or absent from the report set | **nothing** |
| `TIMED_OUT` | exceeded its budget | **nothing** |
| `INVALID` | claimed success, emitted no readable `objections` | **nothing** |

`ReviewGate.review` checks this **before** anything else:

```python
if not execution_status.is_trustworthy:
    return ReviewDecision(status=ReviewStatus.REVIEW_INCOMPLETE, ...)
```

`REVIEW_INCOMPLETE` is deliberately not `REJECT`. A rejection is a verdict; this
is the absence of one. The candidate is stored with `status = SUPPRESSED` rather
than `REJECTED`, because recording a verdict nobody reached would be a lie in
the audit trail.

The parameter defaults to `COMPLETED` so unit tests that already know the
reviewer ran do not have to say so twice. Production always passes the measured
value, from `redteam_execution_status(report)`.

---

## The deterministic fallback

`review/fallback.py` re-implements the red team's *structural* checks directly
against the agent reports: independence collapse, one-sided book, ignored event
risk, degraded inputs. It has no dependency on the red-team agent, so a failure
inside that agent cannot take the fallback down with it.

It is a subset — it cannot read a headline and notice the thesis is nonsense —
so it always appends one finding of its own:

> Adversarial review ran in deterministic fallback mode: the red-team agent did
> not complete, so only structural checks were applied and no argument about the
> substance was made.

That finding travels into the report. Degraded review is not review, and the
reader is told which one they got.

---

## Failure cases, in full

| Failure | Behaviour | Test |
|---|---|---|
| Red team raises, fallback available | `COMPLETED_BY_FALLBACK`, publication continues, degradation stated | `test_red_team_failure_falls_back_to_deterministic_review` |
| Red team raises, fallback produces nothing | `REVIEW_INCOMPLETE` → whole publication suppressed | `test_red_team_failure_suppresses_publication` |
| Red team returns `{}` with no `objections` key | `INVALID` → suppressed | `test_malformed_red_team_output_is_invalid_not_clean` |
| Red team completes with `objections: []` | `APPROVE` permitted | `test_completed_review_with_no_findings_may_approve` |
| Follow-up agents fail | `followup_failed`, HIGH finding injected next round | `test_failed_followup_is_not_treated_as_evidence` |
| Follow-up runs but every agent is unusable | same — the request is not treated as satisfied | same |
| No agent can answer the question | same | same |
| Pipeline2 raises anywhere | `PublicationSet.suppressed_set` — nothing published | `test_scenario_b_pipeline_crash_publishes_nothing` |
| Every candidate rejected | empty publication; the index call is withheld too | `test_scenario_a_red_team_rejects_everything` |

The rule underneath all of it:

```
REVIEW FAILURE   ≠  APPROVAL
FOLLOW-UP FAILURE ≠  EVIDENCE RECEIVED
2.0 FAILURE       ≠  LEGACY PUBLICATION
```

---

## What is recorded

Per round, in memory (`CandidateReview.rounds`) and in the `review_rounds`
table (migration 006):

```
round_number  graph_version  evidence_nodes  effective_independent
red_team_execution_status  n_findings  findings  followup_agents
```

Round 1 is never overwritten by round 2. After-action review needs to see the
whole arc — objection raised, research demanded, evidence arrived, objection
withdrawn — and a single final-state row cannot express any of it.

Per run, in `run_control_path`:

```
review_rounds  red_team_attempts  red_team_successes  red_team_failures
review_incomplete  followup_requests  followup_agents_selected
followup_agents_executed  followup_agents_failed
evidence_nodes_before_followup  evidence_nodes_after_followup
graph_versions  event_context_resolved  contextual_weights_used
```

`graph_versions == 1` on a run with `followup_requests > 0` means the stale
loop is back. That is the single number to watch.
