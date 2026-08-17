# Recommendation Engine

## Predictions vs recommendations

1.x emitted predictions: direction, probability, bracket. Every candidate
became long or short because the vocabulary contained nothing else. A system
that cannot say "I don't know" will say something else instead.

## Conviction levels

| Conviction | Meaning |
|---|---|
| `HIGH_CONVICTION` | broad independent evidence, no contradictions, no pending event |
| `MODERATE_CONVICTION` | actionable but qualified |
| `LOW_CONVICTION` | a case exists, not enough to act |
| `INSUFFICIENT_EVIDENCE` | too little independent evidence — about *us*, not the security |
| `CONFLICTING_EVIDENCE` | credible evidence both ways, unresolved |
| `NO_ACTIONABLE_EDGE` | discernible direction, not enough to clear costs |

## Recommendation types

`WATCH` · `INVESTIGATE` · `FAVORABLE` · `UNFAVORABLE` · `AVOID` ·
`WAIT_FOR_CONFIRMATION` · `NO_EDGE` · `INSUFFICIENT_EVIDENCE`

Only `FAVORABLE` and `UNFAVORABLE` imply a direction.

## Assessment order

Ignorance is checked **before** edge, because a confident number computed from
nothing is the most dangerous output the system can produce:

1. adequate evidence base? → `INSUFFICIENT_EVIDENCE`
2. does evidence contradict itself? → `CONFLICTING_EVIDENCE`
3. edge after costs? → `NO_ACTIONABLE_EDGE`
4. graded conviction

## Output sections

Observation · Interpretation · Prediction · Recommendation · Confidence ·
Supporting Evidence · Contradicting Evidence · Key Risks · Invalidation
Conditions · What Would Change Our Mind · Historical Analogues · Data Quality ·
Uncertainty

These are kept separate so a reader can see *which part* they disagree with.
