# Operations

## Daily cycle

| Time (ET) | Action |
|---|---|
| 08:15 | `marketswarm run` — research pass, writes the report |
| 09:30–16:00 | `marketswarm monitor` — invalidation watch |
| 16:45 | `marketswarm score` — resolve, score, learn |
| any time | `marketswarm dashboard` — operator overview |

The daemon runs the first and third automatically on trading days.

## Commands

```bash
marketswarm migrate [--check]     # apply/inspect schema migrations
marketswarm dashboard [--json]    # system, agents, performance, failures, costs
marketswarm investigate           # what the Event Brain and Chief would do
marketswarm review <rec_id>       # full audit trail for a recommendation
marketswarm memory [--category X] [--subject Y] [--prune]
marketswarm experiments [--propose] [--register]
marketswarm calibration           # reliability curve, Brier decomposition
marketswarm backtest --n-trials N # purged walk-forward validation
```

## Observability

`agent_runs` records status, latency, retries, errors, providers, tokens and
cost per agent per run. `system_events` records structured events. Both are
queryable through `ReadOnlyAPI`.

Statuses: `IDLE`, `PLANNING`, `INVESTIGATING`, `WAITING`, `REVIEWING`,
`RED_TEAM`, `REVISING`, `COMPLETE`, `FAILED`, `SKIPPED`.

## Degradation

| Level | Meaning | Publishes ideas? |
|---|---|---|
| `HEALTHY` | everything reported | yes |
| `DEGRADED` | optional inputs missing | yes, with a stated gap |
| `CRITICALLY_DEGRADED` | a capability-bearing agent died | that output class is suppressed |
| `UNUSABLE` | `technicals` or `cross_verify` died | no directional ideas at all |

Missing evidence is stated in the report, never silently omitted.

## Circuit breakers

Per dependency. 4 consecutive failures opens the circuit for 120s; calls then
short-circuit instead of timing out repeatedly. Two successful probes close it.
`marketswarm dashboard` shows open circuits.

## Costs

`ModelRouter` enforces daily and per-run USD caps. When exhausted it degrades
to deterministic. Most work never calls a model at all.

## Backups

Everything learned lives in `~/.marketswarm/memory.db` (or
`/var/lib/marketswarm/memory.db`). Copy that file and you keep the entire track
record, calibration, memory and experiment history.
