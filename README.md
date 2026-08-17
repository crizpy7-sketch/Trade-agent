# MarketSwarm

An autonomous pre-market research agent. It wakes before the U.S. open, gathers
evidence from a dozen independent sources, fuses it with Bayesian statistics
into a calibrated probability, builds concrete bracketed trade ideas whose odds
come from a path simulation rather than from a narrative, writes a report — and
then, after the close, scores itself and updates its own weights.

**Research and educational software. Nothing it produces is financial advice.**

---

## What it actually does

Sixteen agents run in dependency-ordered stages, each publishing findings,
evidence with provenance, and directional signals:

| Stage | Agents |
|---|---|
| 1 (parallel) | overnight scan · global markets · futures · volatility regime · breaking news · economic calendar · SEC filings · options flow · institutional activity |
| 2 | earnings · technicals · sentiment |
| 3 | cross-verification |
| 4 | risk assessment |
| 5 | day trading playbook |
| 6 | red team (argues against the playbook) |

Any agent may fail. A failure degrades the report and is stated in it; it never
takes down the run.

### The statistics

This is the part that separates it from a headline summarizer.

- **Logit-pooled Bayesian fusion.** Signals combine as log-odds, weighted by
  learned reliability. Probabilities are never averaged — independent
  likelihood ratios add in log-odds space, and linear pooling is systematically
  under-confident at the tails.
- **Correlation haircut.** Futures, global indices and pre-market breadth all
  read the same overnight risk appetite. The effective sample size is shrunk by
  `n / (1 + ρ(n−1))`, so twelve correlated reads do not masquerade as twelve
  independent votes. This is the single largest source of overconfidence in
  naive signal stacks.
- **Beta-Binomial source reliability.** Each source carries a posterior over
  its hit rate under a Jeffreys prior. A source at 60% over 200 calls outweighs
  one at 90% over four, and a source at 50% carries no weight at all.
- **Empirical-Bayes shrinkage.** Per-agent hit rates are pulled toward the
  pooled mean in proportion to their noise (Stein).
- **Student-t barrier Monte Carlo.** Every idea's probability is
  P(target touched before stop) from 8,000 simulated paths with df=4
  innovations. Not a normal distribution — intraday returns have fat tails, and
  sizing a stop off a Gaussian quantile under-prices the move that actually
  stops you out.
- **Black-Scholes option repricing.** Premium targets and stops are computed by
  repricing the contract at the bracket levels with half its remaining life
  burned, using each contract's own IV. Strikes are chosen by computed delta,
  not by distance from spot.
- **GARCH(1,1) volatility forecast** and a Hurst exponent for regime
  classification; VIX term structure for the risk backdrop.
- **Expected value and fractional Kelly.** Every idea is scored in R-multiples
  net of friction. Sizing is quarter-Kelly — full Kelly is optimal only if the
  probability is exactly right, and it never is.
- **Correlated portfolio heat.** Three 1% ideas in the same direction are not
  3% of diversified risk, and the report says so.

### The learning loop

`marketswarm score` runs after the close and closes the loop:

1. Resolves each published idea against real intraday bars — target *before*
   stop, in sequence, exactly as the Monte Carlo priced it.
2. Scores with Brier, log loss, and Murphy's reliability/resolution/uncertainty
   decomposition, plus a Wald SPRT on whether the system has demonstrated skill
   at all.
3. Credit-assigns outcomes to the agents that pushed the call and updates each
   agent's weight through its Beta posterior.
4. Refits a Platt recalibrator on the raw-vs-realized history. If the agent has
   been overconfident, tomorrow's probabilities are shrunk toward 50%
   automatically.
5. Mines conditional lessons — "in stress regimes this setup hits 38% against a
   55% baseline over 19 calls" — but only promotes them above a minimum sample
   size and effect size, so it learns rather than overfits.

Everything is persisted in SQLite and read back by the next morning's run. The
agent's behaviour changes over time without anyone editing code.


---

## Historical validation

The agent ships with a replay harness so you do not have to wait a year to find
out whether it works. It reruns the **production** decision code over years of
real data with purged walk-forward validation, realistic costs, and overfitting
controls.

```bash
marketswarm fetch --source github_sp500      # 505 names, daily OHLCV, 2013-2018
marketswarm backtest --folds 5 --n-trials 12
```

Design points that make it honest rather than flattering:

- **Lookahead is structurally impossible.** `PointInTimeStore` raises
  `LookaheadError` when asked for data past the decision timestamp. Features
  see prior bars plus that day's opening print — nothing else.
- **It tests the real code.** The replay imports the live `_build_bracket`,
  barrier Monte Carlo and EV gate. A backtest of a reimplementation measures
  the reimplementation.
- **Ambiguity resolves pessimistically.** Daily bars cannot say whether the
  high or the low came first; sessions touching both barriers score as losses.
- **Purged, embargoed folds.** Training samples whose outcome window overlaps
  the test block are dropped, plus an embargo either side.
- **Selection bias is priced in.** Deflated Sharpe against the best-of-N-random
  hurdle, and CSCV probability of backtest overfitting.
- **Baselines are always reported.** Always-long and random-entry run alongside,
  because "positive" means nothing until it beats doing something stupid.

**Results on 2013-2018 are in [`FINDINGS.md`](FINDINGS.md): no demonstrated
edge**, and four real bugs found along the way. Read it before trusting
anything the agent prints.

## Intraday monitoring

The morning report is stale by 9:35. Every idea carries machine-checkable
invalidation conditions, and this checks them live:

```bash
marketswarm monitor            # poll every 5 minutes
marketswarm monitor --once     # single check
```

It reports each idea as pending / working / target hit / stopped / invalidated,
tracks maximum favourable and adverse excursion, and closes out resolved ideas
so the learning loop sees the same outcome you did. It never places an order.

---

## Install on a VPS

### systemd (recommended)

```bash
git clone https://github.com/crizpy7-sketch/trade-agent.git
cd trade-agent
sudo ./deploy/install.sh
sudo nano /etc/marketswarm/env        # set MARKETSWARM_CONTACT at minimum
sudo systemctl start marketswarm
sudo journalctl -u marketswarm -f
```

### Docker

```bash
cp .env.example .env && nano .env
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml logs -f
```

### Local / development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
marketswarm status
marketswarm run
```

---

## Usage

```bash
marketswarm run                  # pre-market research pass (exits quietly if the market is closed)
marketswarm run --force          # run anyway, e.g. on a weekend for testing
marketswarm run --json           # machine-readable output
marketswarm score                # resolve yesterday's ideas and learn
marketswarm score --postmortem   # add an LLM review of the track record
marketswarm calibration          # reliability curve, Brier decomposition, learned lessons
marketswarm status               # session, config, integrations, track record
marketswarm holidays 2026        # market closures and early closes
marketswarm daemon               # scheduler (what systemd runs)

marketswarm fetch --source github_sp500   # download historical data
marketswarm backtest --n-trials 12        # purged walk-forward validation
marketswarm monitor                       # intraday invalidation watch
```

The daemon runs the research pass at `MARKETSWARM_RUN_TIME` ET on trading days
and the scoring pass after the close. It computes NYSE holidays and early
closes from first principles, so it needs no calendar service and never
publishes a report on a day the market is shut.

---

## Configuration

Secrets come from the environment (see `.env.example`). Everything else lives
in `~/.marketswarm/config.yaml`, created by `marketswarm init`:

| Key | Default | Meaning |
|---|---|---|
| `universe` | 18 liquid names | What to scan. Only optionable, tight-spread names belong here. |
| `signal_correlation` | `0.35` | Assumed pairwise signal correlation. Raise it to become more conservative. |
| `base_risk_pct` | `0.75` | Per-idea risk budget before today's conditions cut it. |
| `friction_r` | `0.06` | Round-trip cost as a fraction of R. |
| `run_time_et` / `score_time_et` | `08:15` / `16:45` | Daemon schedule, Eastern. |
| `notify_on` | `always` | `always`, `high_confidence`, or `never`. |

Only `MARKETSWARM_CONTACT` is genuinely required — the SEC throttles clients
that do not identify themselves. Without an Anthropic key the agent runs in
full and simply omits the written narrative; the numbers do not come from a
language model.

---

## Data sources

| Layer | Source | Prior reliability |
|---|---|---|
| Filings, insider activity | SEC EDGAR | 0.98 |
| Rates, macro | U.S. Treasury, FRED | 0.97 |
| Quotes, OHLCV, option chains | Public market data endpoints | 0.90–0.93 |
| Newswires | Reuters, AP | 0.82 |
| Financial media | CNBC, MarketWatch | 0.70 |
| Aggregators | Google News, Yahoo | 0.58–0.62 |
| Social sentiment | *deliberately excluded* | — |

Priors are cold-start values only; they are replaced by scored performance as
the agent accumulates history. Social sentiment is excluded on purpose: it is
reflexive, trivially manipulated, and has no stable relationship to next-session
returns.

---

## Honest limitations

- **The edge is unproven, and measured at zero.** Five years of purged
  walk-forward on real data produced −0.07R per trade net of costs, a Brier
  skill score of −0.007, PBO 0.46 and a deflated Sharpe of 0.13. See
  [`FINDINGS.md`](FINDINGS.md). The agent will regularly report that no idea
  clears expected value — that is the system working, not failing. Run
  `marketswarm calibration` and believe the SPRT verdict over the narrative.
- **"Options flow" is inferred, not observed.** Free chains give volume and
  open interest, not the trade tape. Unusual activity here means high
  volume/OI turnover; it cannot distinguish an institutional buy from a dealer
  hedge or a closing trade.
- **13F data is up to 45 days stale** and can never inform an intraday
  decision. Only 13D/G and Form 4 filings are timely.
- **Option premium levels are model estimates.** Verify against the live chain;
  expect a worse fill than the model shows.
- **No execution.** This agent researches and writes. It does not place orders,
  and adding that is not a small change — it is a different risk category.

---

## Tests

```bash
pip install pytest pytest-asyncio
pytest -q
```

101 tests cover the market calendar (including observed-holiday and Good
Friday rules), every statistical routine, config handling, a full end-to-end
swarm run against synthetic providers, and the backtest machinery — including
negative tests that the store *refuses* lookahead, that purged folds do not
leak, and that ambiguous bars resolve pessimistically.

## Licence

MIT — see `LICENSE`.
