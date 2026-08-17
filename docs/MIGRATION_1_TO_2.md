# Migrating 1.1.2 → 2.0

## TL;DR

```bash
marketswarm migrate --check    # see what will apply
marketswarm migrate            # apply — additive, nothing is dropped
marketswarm status             # verify
```

Migrations run automatically on the next `Swarm(...)` construction as well.

## What changes

**Nothing is removed.** Migrations are additive `CREATE TABLE IF NOT EXISTS`
plus a version ledger. All 1.x tables (`runs`, `predictions`, `evidence`,
`agent_scores`, `source_scores`, `calibration_state`, `lessons`) are untouched,
and existing history keeps scoring and calibrating exactly as before —
`test_migrations_preserve_existing_1x_data`.

## New tables

`investigations` · `evidence_nodes` · `evidence_edges` · `recommendations` ·
`recommendation_revisions` · `agent_context_scores` · `market_regimes` ·
`memories` · `mistakes` · `experiments` · `experiment_results` ·
`calibration_results` · `after_action_reviews` · `agent_runs` ·
`system_events` · `schema_migrations`

## Public interface changes

| Item | Change |
|---|---|
| `SwarmResult` | new optional field `v2` (Pipeline2Result or None) |
| `_build_bracket` | new optional params `min_risk_atr`, `max_risk_atr`, `buffer_atr` — defaults reproduce 1.x geometry |
| `edge.evaluate_trade` | unchanged; `evaluate_bracket` added for the three-way distribution |
| `barrier_probabilities` | same signature; now cached and deterministic; two new result fields |
| CLI | 6 new commands; all existing ones unchanged |

No existing signature was broken.

## Rollback

The pre-upgrade state is tagged `v1.1.2-baseline`. A 2.0 database is readable
by 1.x code — the new tables are simply ignored — so rolling back the code does
not require rolling back the schema.

## Behaviour you will notice

1. **Some ideas now get rejected.** The review gate removes what it cannot
   defend. Fewer, better-qualified outputs is the intent.
2. **`INSUFFICIENT_EVIDENCE` and `NO_EDGE` appear.** These are answers, not
   errors.
3. **Confidence numbers are lower.** Correlated evidence is discounted properly.
4. **Quiet names cost less.** Effort now follows the events.
