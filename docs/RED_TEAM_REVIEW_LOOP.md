# Red Team and Review Gate

## The defect this fixes

In 1.x the Red Team ran last, produced objections, and **nothing consumed
them**. The playbook was already final. Criticism that cannot change the output
is theatre.

## The flow

```
Playbook → Red Team → Review Gate → [Revision] → Final
```

`ReviewGate.review()` returns a `ReviewDecision` the pipeline must apply:

| Status | Effect |
|---|---|
| `APPROVE` | published unchanged |
| `APPROVE_WITH_REDUCED_CONFIDENCE` | confidence lowered, reasons attached |
| `REQUEST_MORE_RESEARCH` | follow-up investigation, then re-review |
| `MODIFY` | target/stop/size changed |
| `REJECT` | **not published at all** |

## Guarantees (all test-enforced)

1. **A CRITICAL finding can never be approved.** No policy configuration
   permits it — `test_no_policy_configuration_lets_a_critical_through`.
2. **Enough HIGH findings force rejection** (default 3).
3. **A HIGH finding caps confidence** (default 55).
4. **Below the confidence floor the idea is rejected**, not published weakly.
5. **A general objection naming no ticker applies to every idea.** The original
   adapter required an explicit symbol match and silently dropped critical
   findings — fixed, and covered by
   `test_general_critical_objection_is_never_dropped`.

## Termination

Bounded three ways: `max_iterations` (default 2), a latch so research is
requested at most once, and a signature guard that stops when the same verdict
repeats. `ReviewLoop(max_iterations=0)` raises.

## Audit trail

Every iteration is recorded: original snapshot, findings, decision, what was
demanded, what changed, and why. `ReviewOutcome.audit_trail()` returns the
sequence; `explain()` renders it.

## Configuration

`GatePolicy(critical_rejects, high_findings_to_reject, high_confidence_penalty,
medium_confidence_penalty, low_confidence_penalty, min_confidence_to_survive,
max_confidence_after_high, research_severity)`.

Defaults are deliberately strict: the failure mode this exists to prevent is
waving things through.
