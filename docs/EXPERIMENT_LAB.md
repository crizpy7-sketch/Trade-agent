# Experiment Laboratory

## The rule

**SELF-IMPROVEMENT IS NOT SELF-DEPLOYMENT.**

`ResearchScientist` can propose, experiment, backtest and compare. It has no
`promote`, `deploy`, `apply_to_production` or `set_champion` method — verified
by `test_research_scientist_cannot_promote`.

## Champion protection

`Challenger.apply_to(champion_config)` returns a **copy**. The champion object
is never mutated, tested in
`test_champion_is_never_mutated_by_a_challenger`.

## Promotion gates — all must pass

| Gate | Default |
|---|---|
| minimum samples | 200 |
| minimum trades | 100 |
| mean R improvement | ≥ +0.01 absolute |
| absolute mean R | must be > 0 |
| drawdown worsening | ≤ 20% relative |
| Brier worsening | ≤ 0.005 |
| deflated Sharpe | ≥ 0.90 |
| PBO | ≤ 0.35 |

One improved metric is not evidence. A challenger with better mean R and a
much worse drawdown is rejected.

## Human approval

`require_human_approval=True` by default. Passing every gate marks a challenger
eligible; a human must still call `promote(..., human_approved=True)`. There is
no override parameter — an override is the first thing over-eager automation
reaches for.

## Research Scientist

Proposals come from the measured failure taxonomy, and only for failures that
have recurred at least 5 times. One unlucky call changes nothing.

```bash
marketswarm experiments --propose            # analyse and report
marketswarm experiments --propose --register # file as inert challengers
```
