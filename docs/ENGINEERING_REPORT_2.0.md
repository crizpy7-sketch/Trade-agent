# MarketSwarm 2.0 — Final Engineering Report

Version 2.0.0 · built from `v1.1.2-baseline` · 57 files changed, 9,609 insertions

---

## 1. Executive Summary

MarketSwarm 1.x was a **fixed pipeline**: the same 14 agents ran in the same
order every morning, produced a report, and stored it. It had good statistics
underneath and one fatal structural flaw — the adversarial reviewer wrote
prose that nothing read. It could criticise its own conclusions at length and
publish them unchanged.

MarketSwarm 2.0 is a **decision system**. The loop is now:

```
observe → detect → investigate → delegate → gather evidence → cross-verify
   → predict → attack its own thesis → revise → publish → monitor → score
   → learn → experiment → improve
```

The changes that matter, in order of importance:

1. **The red team can now change the output.** A `ReviewGate` converts
   objections into a typed decision — approve, reduce confidence, request more
   research, modify, or reject — and a rejection means the idea is not
   published. This was the single largest defect in 1.x.
2. **The system can say "I don't know."** `Conviction.INSUFFICIENT_EVIDENCE`
   and `NO_ACTIONABLE_EDGE` are first-class outputs. 1.x always produced a
   directional view because it had no way to represent ignorance.
3. **Evidence is a graph, not a list.** Sources are clustered, correlated
   evidence is discounted via `n / (1 + ρ(n−1))`, and six score dimensions
   (reliability, corroboration, timeliness, specificity, directness,
   independence) are tracked separately rather than collapsed into one number.
4. **Investigation is dynamic.** An event brain detects anomalies; a chief
   investigator forms hypotheses, assigns priority, and spends a bounded budget
   of agents, iterations, cost and wall-clock on each.
5. **Memory is institutional.** Lessons, mistakes (11-code taxonomy),
   contradictions and supersessions persist with per-category TTL and feed a
   propose-only Research Scientist.
6. **Four real bugs were found and fixed**, three of them by tests written
   during this work (see §6).

**What this report does not claim.** No predictive edge is demonstrated.
Deflated Sharpe is **0.004** against a 12-trial hurdle. The measured
improvement from −0.0712R to −0.0223R per trade is attributable to a bug fix
in the backtest, **not** to the new architecture, and the 2.0 decision gates
filtered exactly zero candidates on the available data. §9 explains why in
detail. This is an architecture and correctness upgrade whose forecasting
value is, at present, **unmeasured**.

---

## 2. Architecture Before (1.1.2)

```
clock → providers → 14 agents (fixed DAG) → synthesis → report → sqlite
```

| Property | 1.1.2 |
|---|---|
| Agent selection | Static list, identical every run |
| Investigation | None — no concept of a question |
| Evidence | Flat list of signals with weights |
| Correlation handling | Single global haircut in cross-verify |
| Red team | Wrote text; **nothing consumed it** |
| Uncertainty | Every run produced a directional probability |
| Memory | Lessons table, append-only, no taxonomy |
| Schema | `CREATE TABLE IF NOT EXISTS`, no version tracking |
| Self-improvement | None |
| Failure handling | Provider exception → agent returns empty |
| Security | Ad-hoc; no allowlist, no injection defence |
| LLM use | Single model, optional, no routing or budget |

Strengths worth preserving (and preserved verbatim): logit-pooled Bayesian
fusion, Kish effective sample size, Beta-Binomial posteriors with Jeffreys
prior, Student-t barrier Monte Carlo, Black-Scholes greeks, dealer GEX,
purged walk-forward CV with embargo, deflated Sharpe, CSCV, point-in-time
data store that raises on lookahead.

---

## 3. Architecture After (2.0.0)

```
                       ┌──────────────── observability ────────────────┐
clock → providers → EVENT BRAIN ──anomalies──→ CHIEF INVESTIGATOR
                                                   │  hypotheses + budget
                                                   ▼
                                        CAPABILITY REGISTRY → agents
                                                   │
                                                   ▼
                                          EVIDENCE GRAPH  (clustered,
                                                   │       ρ-discounted)
                                                   ▼
                                          cross-verify / fusion
                                                   ▼
                                       RECOMMENDATION ENGINE
                                     (conviction incl. "no edge")
                                                   ▼
                              ┌────── RED TEAM ⇄ REVIEW GATE ──────┐
                              │   generate → attack → revise (≤N)   │
                              └──────────────┬──────────────────────┘
                                     approve │ reject → not published
                                             ▼
                                    report / api / notify
                                             ▼
                              monitor → score → AFTER-ACTION REVIEW
                                             ▼
                        INSTITUTIONAL MEMORY ⇄ CONTRIBUTION SCORING
                                             ▼
                        RESEARCH SCIENTIST → EXPERIMENT LAB
                             (propose only)   (champion/challenger,
                                               human-approved promotion)
```

Cross-cutting: `security.py` (allowlists, injection defusal, redaction),
`resilience.py` (circuit breakers, degradation levels), `observability.py`
(run traces, agent timings, system events), `llm_router.py` (model tiering
and cost budget).

`orchestrator.py` runs Pipeline 2 inside a `try/except`. If any 2.0 layer
fails, the run degrades to the 1.x path rather than producing nothing.

---

## 4. Files Changed

**44 new files, 13 modified.** No 1.x module was rewritten.

### New — core (20)

| File | Purpose |
|---|---|
| `memory/migrations.py` | 4 versioned migrations, forward-only, refuses to downgrade |
| `evidence/graph.py` | Evidence graph, clusters, ρ-discounted effective N |
| `review/gate.py` | Typed review decisions; rejection suppresses publication |
| `review/loop.py` | Bounded generate → attack → revise cycle |
| `investigation/event_brain.py` | Deterministic anomaly detectors |
| `investigation/chief.py` | Hypotheses, priority, budget allocation |
| `investigation/registry.py` | Agent capability registry and selection |
| `investigation/plan.py` | Investigation plan and budget accounting |
| `recommend/engine.py` | Conviction / recommendation types incl. "no edge" |
| `memory/institutional.py` | Categories, TTL, mistake taxonomy, supersession |
| `memory/contribution.py` | Ablation-based per-agent contribution scoring |
| `experiments/lab.py` | Champion/challenger, promotion gates |
| `experiments/scientist.py` | Propose-only research scientist |
| `security.py` | Allowlists, sanitisation, redaction, path/URL guards |
| `resilience.py` | Circuit breakers, degradation tracking |
| `observability.py` | Run traces, agent runs, system events |
| `llm_router.py` | Model tiering, cost budget, fallback |
| `after_action.py` | Post-resolution review of published calls |
| `api.py` | Read-only HTTP surface |
| `pipeline2.py` | Wires all of the above |

Plus 4 `__init__.py` files for the new packages.

### New — tests (4)

`tests/test_review.py`, `tests/test_pipeline2.py`, `tests/test_memory2.py`,
`tests/test_adversarial.py` — 136 new tests.

### New — docs (16)

15 subsystem documents plus this report.

### Modified (13)

| File | Change |
|---|---|
| `orchestrator.py` | Runs migrations on construction; adds `result.v2`; degrades safely |
| `cli.py` | New `migrate`, `investigate`, `review`, `experiments`, `memory` commands |
| `report.py` | Renders red-team section (§4) and review decisions |
| `agents/redteam.py` | Emits structured objections consumable by the gate |
| `agents/synthesis.py` | Consumes evidence-graph output |
| `memory/store.py` | Delegates schema creation to migrations |
| `backtest/replay.py` | `evaluate_bracket()`, corrected impact model, `stop_atr` wired |
| `models/logistic.py` | Deterministic seeding |
| `stats/gex.py` | Deterministic seeding, note wording |
| `README.md`, `FINDINGS.md`, `pyproject.toml`, `__init__.py` | Docs and version |

---

## 5. New Components

**Event Brain** — deterministic detectors only (no LLM): gaps normalised by
the symbol's own ATR, volume anomalies, sector divergence, volatility regime
shifts, options anomalies. Options anomalies are tagged `inferred: True` and
carry the caveat that they are derived from volume/open-interest, not observed
trade flow.

**Chief Investigator** — converts anomalies into hypotheses and allocates a
budget by priority:

| Priority | Iterations | Agents | Cost | Seconds |
|---|---|---|---|---|
| CRITICAL | 3 | 14 | 40 | 180 |
| HIGH | 2 | 10 | 25 | 120 |
| MEDIUM | 2 | 8 | 15 | 60 |
| LOW | 1 | 5 | 6 | 30 |

The optional LLM pruning pass accepts **only** hypotheses returned verbatim
from the supplied list, so a prompt-injected response cannot introduce one.

**Evidence Graph** — 11 clusters with assumed intra-cluster ρ (market_beta
0.85 … insider 0.30), cross-cluster ρ 0.15. Effective independent count is
computed within clusters and then again across them. `EvidenceScore.composite()`
returns **`None`** when nothing is known — never a fabricated 0.5.

**Review Gate** — the highest-priority fix. Statuses: `APPROVE`,
`APPROVE_WITH_REDUCED_CONFIDENCE`, `REQUEST_MORE_RESEARCH`, `MODIFY`,
`REJECT`. Default policy: any CRITICAL finding rejects; 3 HIGH findings
reject; each HIGH costs 25 confidence points; anything under 15 confidence
dies; nothing survives a HIGH above 55. `gate.apply()` returns `None` on
reject, and Pipeline 2 runs the gate **before** publication.

**Recommendation Engine** — ignorance is checked *before* edge. Conviction:
HIGH / MODERATE / LOW / INSUFFICIENT_EVIDENCE / CONFLICTING_EVIDENCE /
NO_ACTIONABLE_EDGE. Types: WATCH, INVESTIGATE, FAVORABLE, UNFAVORABLE, AVOID,
WAIT_FOR_CONFIRMATION, NO_EDGE, INSUFFICIENT_EVIDENCE.

**Institutional Memory** — 7 categories with per-category TTL, an 11-code
mistake taxonomy, `supersede()`, `record_contradiction()`, `prune()`,
`mistake_frequency()`.

**Contribution Scoring** — ablation log-loss delta per agent, with
`MIN_OBSERVATIONS=12`, empirical-Bayes shrinkage at n=30, weight floor 0.15,
ceiling 1.75, and a maximum step of 0.25 per update. An agent cannot be
deleted or doubled on twelve noisy observations.

**Experiment Lab / Research Scientist** — the scientist proposes from measured
recurring failures (≥5 occurrences) and from calibration statistics (n≥50).
`ResearchScientist` has **no reference to the champion config and no method
that writes production state**. Promotion lives on `ExperimentLab`, behind
gates and human approval. This is structural, not instructional.

**Resilience** — `CircuitBreaker` (CLOSED/OPEN/HALF_OPEN) plus a
`DegradationTracker`. `CRITICAL_AGENTS = {"technicals", "cross_verify"}`; when
critical capability is lost the system suppresses idea publication and says
so in plain language rather than publishing thinner ideas silently.

---

## 6. Bugs Fixed

### The Red Team decision-path issue (the important one)

**1.x:** `RedTeamAgent` produced prose. `report.py` never rendered it and
`synthesis.py` never read it. The system could identify that its own
conviction rested on one correlated signal and publish that conviction at full
confidence anyway.

**Fix:** objections are now structured findings with severity; `ReviewGate`
maps them to a typed decision; `review/loop.py` runs a bounded
generate → attack → revise cycle; `report.py` renders the outcome. Rejection
means the idea does not reach the reader.

**Second-order bug found while testing that fix** —
`findings_from_redteam_report` required an explicit symbol match, so a general
objection ("every setup is long") matched *no* symbol and was silently dropped
for all of them. **CRITICAL findings were being discarded.** Fixed with
`_applies_to()`: an objection that names tickers applies to those tickers; an
objection that names none is general and applies to all. Three regression
tests cover it.

### 1.x bugs found by tests and backtest

| Bug | Effect | Fix |
|---|---|---|
| `evaluate_trade` charged −1R for "neither barrier hit" | ~50% of sessions penalised as losses; this is why the live playbook kept concluding "nothing clears the cost bar" | `evaluate_bracket()` using the three-way outcome distribution |
| Market impact model omitted σ | ~50bps charged for 0.02% of ADV; `cost_r > 1.0`; every idea rejected | `impact ≈ coef × σ_daily × √participation` |
| `stop_atr` never reached `_build_bracket` | Parameter sweep reported results for configurations it had not tested | Wired through, plus `min_risk_atr` / `max_risk_atr` / `buffer_atr` |
| Bracket geometry placed stops inside support | Stops sat where they were most likely to be run | Rewrote `_build_bracket`: buffer beyond structure, min/max risk in ATR, returns `None` when there is no room |
| Hurst exponent ≈ 0 on a deterministic ramp | Degenerate std | Guarded and clamped; test rewritten with a noisy series |
| MC seeded from `hash()` | Reports not reproducible (PYTHONHASHSEED randomises per process) | `zlib.crc32`, then `seed=0` |
| `MARKETSWARM_DATA_DIR` did not move reports | A sandboxed systemd unit would fail to write | Fixed in `Config.load`; 6 config tests |
| Red-team output rendered nowhere | See above | Report §4 plus regression test |

### 2.0 bugs found by the new tests

| Bug | Fix |
|---|---|
| `Observatory.record_report` crashed on a malformed report | Defensive `getattr` |
| After-action review misclassified wins — matched the word "worked", but a win reports "target reached" | Explicit `succeeded: bool` field |
| Path traversal: `..\..\etc` is a legal POSIX filename | `safe_output_path` now rejects backslashes, absolute paths, and any `..` component |

### Test assumptions that were wrong (code was correct)

Recorded for honesty: the review loop terminated via the "evidence already
requested" latch rather than the signature guard (test rewritten); AAPL flat
against a −1.8% index *is* a sector divergence (test corrected); decision
objects legitimately appear twice per iteration (test now filters on
`stage == "red_team"`).

---

## 7. Database Changes

1.x created tables with `CREATE TABLE IF NOT EXISTS` and tracked no version,
so a schema change on a live VPS was an unmanaged event. 2.0 adds
`memory/migrations.py`: forward-only, ordered, recorded, and **refusing to act
when the database is newer than the code**:

```python
if have > LATEST_VERSION:
    log.warning("database schema v%d is newer than this code (v%d); leaving it alone", ...)
    return []
```

| # | Name | Adds |
|---|---|---|
| 001 | `baseline_1x` | Adopts the existing 1.x schema under version control |
| 002 | `investigations_and_evidence_graph` | `investigations`, `evidence_nodes`, `evidence_edges`, `recommendations`, `recommendation_revisions` |
| 003 | `learning_memory_experiments` | `agent_context_scores`, `market_regimes`, `memories`, `mistakes`, `experiments`, `experiment_results`, `calibration_results`, `after_action_reviews` |
| 004 | `observability` | `agent_runs`, `system_events` |

15 new tables. Migration is idempotent, runs automatically on
`Orchestrator` construction, and is exposed as `marketswarm migrate`.
An existing 1.x database upgrades in place with no data loss —
verified on a copy of a populated 1.x database.

---

## 8. Tests

| | Count |
|---|---|
| Tests before (1.1.2) | **103** |
| Tests after (2.0.0) | **239** |
| Passed | **239** |
| Failed | **0** |
| New test code | 3,011 lines across 4 new files |

```
239 passed in 12.36s
```

Coverage of the new surface: review gate policy and rejection paths, the
`_applies_to` regression set, evidence-graph correlation discounting, chief
investigator budget exhaustion, recommendation engine ignorance paths,
migration idempotence and downgrade refusal, memory TTL and supersession,
contribution-scoring safeguards, circuit-breaker state machine, degradation
suppression, prompt-injection defusal, path-traversal and SSRF guards, and a
static assertion that no shell/eval path exists anywhere in the package.

pyflakes is clean apart from intentional `__init__.py` re-exports.

---

## 9. Backtest Comparison

Same data (619,040 daily bars, 505 tickers, 2013–2018), same universe, same
folds, same seeds. Only the decision layer differs.

| | 1.1.2 as documented | after the 2.0 bug fixes | 2.0 decision layer |
|---|---|---|---|
| Candidates | 36,581 | 19,075 | 19,075 |
| Trades taken | 1,519 | 1,354 | 1,354 |
| Net mean R | −0.0712 | −0.0223 | −0.0223 |
| Hit rate | 43.4% | 49.0% | 50.8% |
| Annualised Sharpe | −1.15 | −0.43 | −0.43 |
| Max drawdown | −38.4R | −69.7R | −60.2R |
| Brier skill | −0.0070 | +0.0025 | +0.0025 |
| **Deflated Sharpe** | 0.131 | **0.004** | **0.004** |

**No edge is demonstrated, before or after.** Deflated Sharpe 0.004 against a
12-trial hurdle: the system does not beat random selection on this data.

**The −0.0712R → −0.0223R improvement is a bug fix, not the architecture.**
The 1.1.2 figure was measured while `stop_atr` never reached the bracket
builder, so the sweep reported on configurations it had not tested. Wiring
the parameter through changed the bracket geometry and the measured baseline.
It is a more honest number, not a better system.

**The 2.0 decision layer filtered nothing — 100% of candidates passed every
gate.** This is a real negative result and the reason is instructive: the
gates operate on *evidence quality*, and a daily-bar backtest contains no
evidence to vary. There are no headlines, no filings, no option chains and no
corroboration counts in a 2013–2018 OHLCV set, so every candidate presents the
same three thin clusters (technical, volatility, market beta) and receives the
same verdict. `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE` and red-team
rejection are inert when every input is identical.

**The 2.0 architecture cannot be validated on this data.** Claiming otherwise
from these numbers would be exactly the self-deception the validation layer
exists to prevent. Validating it requires forward paper trading with the live
evidence pipeline, scored via `marketswarm score` and judged by
`marketswarm calibration`. Expect months, and expect the honest answer to stay
"no demonstrated edge" until the data says otherwise.

---

## 10. Security Review

**Hard boundary, enforced in code.** MarketSwarm is research and
decision-support only. `security.py` defines `FORBIDDEN_CAPABILITIES`
including `broker_order`, `funds_transfer`, `portfolio_allocate`. There is no
order path, no credential store for a broker, and `check_tool()` refuses
anything outside `ALLOWED_TOOLS`.

| Threat | Control |
|---|---|
| Command injection | `assert_no_shell_execution()` scans the whole package for `subprocess`, `eval`, `exec`, `os.system` and fails the test suite if any appears. Verified clean. |
| Prompt injection | `sanitise_external_text()` **defuses rather than deletes** — injected instructions are neutralised and preserved so they remain auditable. `fence_untrusted()` wraps all external content. LLM pruning accepts only verbatim items from a supplied list. |
| Secret exfiltration | `redact()` applied to logs, traces and any LLM payload |
| Path traversal | `safe_output_path()` rejects absolute paths, backslashes and `..` components |
| SSRF | `check_outbound_url()` — scheme and host allowlist, private ranges refused |
| Symbol injection | `safe_symbol()` — strict character class |
| Self-deployment | `ResearchScientist` has no promotion method and no champion reference; promotion is `ExperimentLab.promote`, gated and human-approved |
| Cascading failure | Circuit breakers per provider; degradation levels suppress publication instead of degrading silently |

**Verified in this build:**

```
═══ security gate ═══ no shell/eval anywhere
```

**Residual risks I want stated plainly.** The API in `api.py` is read-only but
has no authentication — bind it to localhost or put it behind a reverse proxy
with auth before exposing it. LLM providers see market context and agent
prose; if that matters to you, run without an API key (the mechanical layer
works without one). The SQLite database is unencrypted at rest.

---

## 11. Remaining Limitations

Explicitly, without hedging:

1. **No demonstrated predictive edge.** Deflated Sharpe 0.004. Do not trade
   this with real money on this evidence.
2. **The 2.0 gates are unvalidated.** They filtered nothing in the only
   backtest available, for the reason given in §9.
3. **Options flow is inferred, not observed.** Volume and open interest are
   not order flow. Every options-derived signal carries this caveat and it is
   a real weakness, not a formality.
4. **Daily bars cannot say whether the high or the low came first**, which
   forces a pessimistic tie-break in bracket evaluation and discards real
   information.
5. **Every idea is a market-beta bet in disguise.** There is no cross-sectional
   or long/short construction; the correlated-heat warning fires for a reason.
6. **Correlation coefficients in `CLUSTERS` are assumed, not estimated.** They
   are defensible priors, not measurements. Estimating them from realised data
   is a genuine improvement waiting to be made.
7. **The event brain has no news, filings or transcript feed** in this
   environment. Its detectors are price/volume only.
8. **The LLM layer is optional and unproven.** Everything with teeth is
   deterministic. That is deliberate, but it means the LLM adds prose more
   reliably than it adds skill.
9. **Contribution scoring needs history the system does not yet have.**
   `MIN_OBSERVATIONS=12` is a floor, not a sufficient sample.
10. **Calibration is untested live.** The SPRT will not confirm skill for
    months of paper trading, if ever.

---

## 12. Recommended Next Upgrades (ranked by expected value)

**Tier 1 — the data, which is the actual bottleneck**

1. **Real options order flow** (exchange or a paid vendor). The single largest
   missing input. It converts the flow agent from inference to observation and
   it is the one place where a genuine informational edge plausibly exists.
2. **Intraday bars (1-minute or better).** Removes the pessimistic tie-break,
   makes barrier evaluation honest, and gives the event brain something to
   detect intraday.
3. **Timestamped news and filings with point-in-time guarantees.** Without
   these the evidence graph has three clusters instead of eleven, and §9
   happens again.

**Tier 2 — construction**

4. **Cross-sectional long/short pairs**, so ideas are not all the same beta bet.
5. **Event-conditioned sampling** — trade only around earnings, filings and
   macro prints, where information genuinely arrives, rather than every symbol
   every day.
6. **Estimate cluster correlations from realised data** instead of using the
   assumed priors.

**Tier 3 — process**

7. **Forward paper-trading harness with automatic scoring**, run for at least
   six months before any capital decision.
8. **Authentication on the API** before any non-localhost exposure.
9. **Encrypt the SQLite database at rest** if the VPS is shared.

**Tier 4 — nice to have**

10. Regime-conditional model selection; richer LLM routing; a web dashboard
    beyond the current CLI/HTML report.

I would not spend effort on model sophistication before Tier 1. A better
model on the same data will not find an edge that is not in the data.

---

## 13. Run Instructions

### Install

```bash
unzip marketswarm-2.0.0.zip
cd marketswarm-2.0.0
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.11+. Dependencies are numpy plus stdlib; no compiler needed.

### Configure

```bash
marketswarm init                      # writes ~/.marketswarm/config.toml
export MARKETSWARM_DATA_DIR=/var/lib/marketswarm   # optional; moves db AND reports
export ANTHROPIC_API_KEY=...          # optional — everything with teeth runs without it
```

### Migrate

```bash
marketswarm migrate                   # idempotent; also runs automatically on startup
```

Expected on a fresh database:

```
Applied 4 migration(s): baseline_1x, investigations_and_evidence_graph,
learning_memory_experiments, observability
```

### Test

```bash
python -m pytest -q                   # expect: 239 passed
```

### Run once

```bash
marketswarm run                       # pre-market session for the next trading day
marketswarm run --force               # ignore the market calendar
marketswarm run --date 2026-08-18     # run as of a specific session
marketswarm run --no-llm              # mechanical layer only
marketswarm run --json                # machine-readable output
```

### Run as a daemon

```bash
marketswarm daemon                    # foreground scheduler
sudo cp deploy/marketswarm.service /etc/systemd/system/
sudo systemctl enable --now marketswarm
```

Or `docker compose -f deploy/docker-compose.yml up -d`.

### Inspect

```bash
marketswarm status                    # last run, degradation level, circuit breakers
marketswarm dashboard                 # terminal summary
marketswarm investigate               # what the Event Brain and Chief would do
marketswarm review <recommendation-id>       # the review-gate decision trail
marketswarm memory --category mistake --limit 20
marketswarm memory --prune            # expire memories past their TTL
marketswarm experiments               # champion/challenger state
marketswarm experiments --propose     # Research Scientist proposals (inert)
marketswarm score                     # resolve and score past calls
marketswarm calibration               # Brier, Murphy decomposition, SPRT verdict
marketswarm monitor                   # intraday invalidation checks
```

### Logs and reports

```
$MARKETSWARM_DATA_DIR/marketswarm.db          # sqlite, all state
$MARKETSWARM_DATA_DIR/reports/YYYY-MM-DD.html # rendered report
$MARKETSWARM_DATA_DIR/reports/YYYY-MM-DD.md
journalctl -u marketswarm -f                  # systemd
docker compose logs -f                        # docker
```

### Backtest

```bash
marketswarm fetch --source github_sp500
marketswarm backtest --folds 5 --embargo 5 --n-trials 12
```

Set `--n-trials` to the number of configurations you have *actually* tried,
not the number you are reporting. Lying to it only means lying to yourself.

---

*Research and educational analysis. Not financial advice. MarketSwarm does not
place orders, move funds, or allocate a portfolio, and no part of this system
should be treated as a recommendation to trade. Past performance — including
backtested performance, which is worth considerably less — does not indicate
future results.*
