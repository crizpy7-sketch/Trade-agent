# Institutional Memory

## Categories and retention

| Category | TTL | Holds |
|---|---|---|
| `episodic` | 365d | what happened on a specific day |
| `semantic` | none | recurring patterns |
| `company` | 180d | what we know about an issuer |
| `agent` | 120d | which specialists work when |
| `strategy` | none | hypotheses already tested |
| `failure` | **never expires** | mistakes |
| `experiment` | none | what was tried |

## Hygiene

Storing everything forever is a landfill, not a memory — recall degrades as the
store fills with things that were never useful. `prune()` deactivates expired
entries, unused low-evidence entries older than 90 days, and contradicted
low-confidence entries. Failures and experiments are never pruned.

## Versioning, not overwriting

`supersede()` writes a new version and deactivates the old one, which stays
readable. An after-action review needs to know what we *used* to believe.

`record_contradiction()` flags and halves confidence rather than deleting — a
contradicted memory is information.

## Mistake taxonomy

```
overweighted_correlated_evidence · ignored_contradicting_evidence
acted_on_stale_information · misread_the_regime · red_team_objection_ignored
confidence_exceeded_evidence · failed_to_find_the_catalyst
poor_entry_or_stop_placement · underestimated_trading_costs
genuinely_unpredictable_event · bad_or_missing_data
```

The taxonomy makes failure **countable**. `mistake_frequency()` is the most
valuable query in the system: a code that keeps recurring is a process defect,
not bad luck.

**Only avoidable mistakes become lessons.** A shock nobody could forecast is
recorded but produces no memory — treating it as a lesson teaches the system to
fear the wrong things.
