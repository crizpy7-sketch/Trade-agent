# Learning Engine

## From agreement to contribution

1.x asked "was this agent pointing the right way?" That rewards an agent saying
+0.5 every day in a bull market — right often, useful never.

2.0 asks: **did the forecast get better because this agent was in it?**

`ablation_contribution()` recomputes the fused probability with each agent's
log-odds removed and compares log-loss. Positive means the agent helped.
Negative means removing it would have improved the forecast.

## Contextual weights

One scalar per agent cannot express that options flow is strong into earnings
and weak in a macro shock. Scores are stored per `(agent, context)` where
context is `regime | event_type | horizon`, with back-off to broader slices when
a cell is thin.

## Small-sample safeguards (all tested)

| Safeguard | Value |
|---|---|
| Minimum observations before any deviation | 12 |
| Shrinkage toward neutral | `n / (n + 30)` |
| Weight bounds | 0.15 – 1.75 |
| Max change per update | ±0.25 |

Three lucky calls cannot produce a 2× weight —
`test_small_samples_cannot_move_weights`.

## Calibration

Brier decomposition (reliability / resolution / uncertainty), log loss, Platt
recalibration and a Wald SPRT on whether skill exists at all — all inherited
unchanged from 1.x and still the arbiter. Results are segmented into
`calibration_results` by regime, event type and horizon.

`underperformers()` lists agents measurably making forecasts worse. They are
candidates for removal, not for a smaller weight.
