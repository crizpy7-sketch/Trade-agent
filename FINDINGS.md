# Backtest findings

What happened when the agent's logic was replayed over five years of real
market data, validated properly.

**Headline: no demonstrated edge.** Details below, including the numbers that
say so and the four bugs the exercise uncovered.

---

## Setup

| | |
|---|---|
| Data | 505 S&P 500 constituents, daily OHLCV, Feb 2013 – Feb 2018 (619,040 bars) + SPY 2008–2018 |
| Source | `plotly/datasets` `all_stocks_5yr.csv`, `mplfinance` SPY set — static files, so runs are byte-reproducible |
| Universe tested | 31 liquid large caps (AAPL, MSFT, NVDA, AMD, JPM, XOM, …) |
| Samples | 36,581 point-in-time decisions across 1,181 sessions |
| Entry | the session's opening print |
| Bracket | built by the **production** `_build_bracket`, not a reimplementation |
| Resolution | actual session OHLC; both-barriers-touched scored as a **loss** |
| Validation | 5-fold purged walk-forward, 5-day embargo, expanding window |
| Costs | spread + √-law market impact + commission, per symbol |

Features are computed only from bars strictly *before* the decision date, plus
that day's opening price. The store raises `LookaheadError` rather than
returning future data, so this is enforced rather than intended.

---

## Result

Baseline (what any strategy must beat):

```
always_long   n=36,581   mean +0.0127R   hit 48.6%
random_entry  n=18,215   mean +0.0131R   hit 48.4%
```

Walk-forward, the default configuration:

```
Trades taken       1,519 of 36,581 candidates
Net mean           -0.0712R per trade
Hit rate           43.4%
Profit factor      0.84
Annualised Sharpe  -1.15
Max drawdown       -38.4R
Cost drag          -0.0565R per trade
Brier              0.2499   skill score -0.0070
Forecast 50.9% vs actual 45.7%
```

The model's *resolution* — its ability to separate winners from losers — is
0.0003. That is not a small edge. That is no edge.

### Pre-registered configuration sweep

Twelve configurations declared before running, **all reported**:

| config | n | net R | hit | ann. Sharpe |
|---|---|---|---|---|
| t0.8_s0.5_th0.1 | 3 | +0.5157 | 66.7% | — |
| t2.0_s0.8_th0.1 | 49 | +0.0588 | 57.1% | 1.21 |
| t1.2_s0.8_th0.1 | 14 | +0.0490 | 50.0% | — |
| t1.2_s0.5_th0.1 | 90 | +0.0353 | 54.4% | 0.66 |
| t1.2_s0.8_th0.0 | 1,716 | −0.0191 | 51.3% | −0.46 |
| t2.0_s0.8_th0.0 | 2,762 | −0.0362 | 49.8% | −0.91 |
| t1.2_s0.5_th0.0 | 3,706 | −0.0470 | 47.7% | −0.89 |
| t2.0_s0.5_th0.0 | 4,799 | −0.0472 | 47.8% | −0.88 |
| t0.8_s0.5_th0.0 | 1,873 | −0.0488 | 48.2% | −0.90 |
| t2.0_s0.5_th0.1 | 190 | −0.0660 | 50.5% | −1.21 |

*(t = target ATR multiple, s = stop ATR multiple, th = minimum expected-R gate)*

The four positive configurations all share a raised EV gate — and all have tiny
samples. The top line is three trades. Presenting that as "+0.52R per trade"
would be the exact fraud this harness exists to prevent.

The arbiters:

```
PBO             0.457 over 70 combinatorial splits
                → "the winner is partly luck"
Deflated Sharpe raw 0.021 vs a 0.060 hurdle for the best of 12 random tries
                → DSR 0.131, not significant
```

A PBO near 0.5 means picking the in-sample winner conveys almost no information
about out-of-sample performance. **Conclusion: no edge demonstrated, in any
configuration tested.**

---

## Four bugs the backtest found

This is why you build the harness before the models.

**1. Expected value treated flat days as maximum losses.** `evaluate_trade`
received `p_target_first` and charged −1R to everything else — including the
~50% of sessions that touch neither barrier and close near flat. This
systematically understated EV and is why the live playbook kept reporting
"nothing clears the cost bar." Fixed by `evaluate_bracket`, which uses the full
three-way path distribution.

**2. Market impact was dimensionally wrong.** The √-law term omitted
volatility, so the coefficient had no units and trading 0.02% of ADV was
charged ~50 bps. Round-trip friction came out above 1.0R, rejecting every
candidate. Now `impact ≈ coef × σ_daily × √participation`, giving 4–7% of R for
liquid names — in line with the 6% placeholder it replaces.

**3. A swept parameter was inert.** The first sweep produced *identical* results
for `stop_atr=0.5` and `stop_atr=0.8`, because the value never reached the
bracket builder. The sweep was reporting on configurations it had not tested.
`_build_bracket` now takes explicit risk multiples.

**4. Reports were not reproducible.** Monte Carlo seeds came from Python's
`hash()`, which is randomised per process, so the same inputs produced
different published probabilities on different days. Now deterministic.

Bugs 1 and 2 both biased the system toward *inaction*, which is why they
survived: a system that refuses to trade looks conservative rather than broken.

---

## One thing that got much faster

Barrier probabilities are scale-free — they depend only on where the barriers
sit in σ units and the drift per σ. Normalising and caching on that geometry
took the Monte Carlo from ~3 ms to **21 µs** per call at a 99.5% hit rate, a
~150× speedup, with no loss of fidelity. That is what made a 12-configuration
sweep over 36,581 samples feasible at all.

---

## What this means for the live agent

1. **Do not trade this.** Paper-trade it, watch `marketswarm calibration`, and
   require the SPRT to say "skill confirmed" before risking money. On this
   evidence it will not.
2. **Daily technical features do not predict next-session bracket outcomes.**
   The strongest coefficients (distance from the 50-EMA at −0.18 log-odds/SD,
   21-EMA at +0.12) are weak and partly contradictory. This matches the
   literature; it is not a bug in the features.
3. **Costs dominate.** −0.057R of drag per trade against a gross edge of
   roughly zero. Any real edge must clear friction first, and at this horizon
   friction is most of the game.
4. **The gate matters more than the signal.** Every positive configuration came
   from raising the EV threshold — trading less, not predicting better. That is
   a genuine (if unproven) direction: selectivity is cheaper to improve than
   accuracy.

### What would actually move the needle

In the order I would attempt them:

- **Real options data.** The single largest missing input. Volume/OI inference
  is not order flow, and the flow agent is currently guessing.
- **Intraday bars.** Daily OHLC cannot say whether the high or the low came
  first, which forces the pessimistic tie-break and discards real information.
- **Cross-sectional (long/short) construction.** Every idea here is a market-beta
  bet in disguise; the correlated-heat warning fires for a reason.
- **Event-conditioned samples.** Trading only around earnings, filings and
  macro prints — where information genuinely arrives — rather than every
  symbol every day.

---

## Reproducing this

```bash
marketswarm fetch --source github_sp500
marketswarm backtest --folds 5 --embargo 5 --n-trials 12
```

`--n-trials` is the deflation hurdle. Set it to the number of configurations
you have *actually* tried, not the number you are reporting. Lying to it only
means lying to yourself.

---

*Research and educational analysis. Not financial advice. Past performance —
including backtested performance, which is worth considerably less — does not
indicate future results.*

---

# MarketSwarm 2.0 — backtest comparison

Same data, same universe, same folds, same seeds. Only the decision layer
differs. Nothing was cherry-picked and nothing below shows an edge.

## Measured results

| | 1.1.2 as documented | after the 2.0 bug fixes | 2.0 decision layer |
|---|---|---|---|
| Candidates | 36,581 | 19,075 | 19,075 |
| Trades taken | 1,519 | 1,354 | 1,354 |
| Net mean R | −0.0712 | −0.0223 | −0.0223 |
| Hit rate | 43.4% | 49.0% | 50.8% |
| Annualised Sharpe | −1.15 | −0.43 | −0.43 |
| Max drawdown | −38.4R | −69.7R | −60.2R |
| Brier skill | −0.0070 | +0.0025 | +0.0025 |
| Deflated Sharpe | 0.131 | 0.004 | 0.004 |

## What actually changed, and what did not

**The improvement from −0.0712R to −0.0223R is a bug fix, not the new
architecture.** The 1.1.2 measurement was taken while `stop_atr` never reached
the bracket builder, so the sweep was reporting on configurations it had not
tested. Wiring that parameter through changed the bracket geometry and the
measured baseline. It is a more honest number, not a better system.

**The 2.0 decision layer filtered nothing on this dataset — 100% of candidates
passed every gate.** That is a real and important negative result, and the
reason is instructive: the 2.0 gates operate on *evidence quality*, and a
daily-bar backtest contains no evidence to vary. There are no headlines, no
filings, no option chains and no corroboration counts in the 2013–2018 OHLCV
set, so every candidate presents the engine with the same three thin clusters
(technical, volatility, market beta) and receives the same verdict.

The gates that would discriminate — `INSUFFICIENT_EVIDENCE` when independent
evidence is thin, `CONFLICTING_EVIDENCE` when sources disagree,
red-team rejection on a live objection — are inert when every input is
identical. **The 2.0 architecture cannot be validated on this data.** Claiming
otherwise from these numbers would be exactly the self-deception the validation
layer exists to prevent.

## Honest conclusion

- **No edge is demonstrated, before or after.** Deflated Sharpe 0.004 against a
  12-trial hurdle. The system does not beat random selection on this data.
- The 2.0 work is an **architecture and correctness upgrade**: the red team can
  now change output, evidence is traceable, correlated signals are discounted,
  the system can say "I don't know", and four real bugs are fixed.
- **Whether it forecasts better is unmeasured**, and honestly cannot be
  measured until the evidence layer has data to work on. That needs intraday
  bars, real option flow and timestamped news — the Tier 1 items in the
  recommendations, none of which the free daily dataset provides.

Validating 2.0 properly requires forward paper-trading with the live evidence
pipeline, scored through `marketswarm score` and judged by
`marketswarm calibration`. Expect that to take months, and expect the honest
answer to remain "no demonstrated edge" until the data says otherwise.
