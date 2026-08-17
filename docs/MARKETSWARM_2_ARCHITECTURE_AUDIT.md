# MarketSwarm 2.0 — Architecture Audit

Audit of MarketSwarm 1.1.2 conducted before the 2.0 upgrade.
Baseline commit: `v1.1.2-baseline`. Baseline test result: **103 passed**.

---

## 1. Existing structure

```
marketswarm/
  clock.py            NYSE calendar, sessions          208 loc
  config.py           YAML + env config                167
  orchestrator.py     staged swarm runner              260
  report.py           Markdown/HTML rendering          423
  monitor.py          intraday invalidation watch      257
  llm.py              optional narrative layer         153
  notify.py           webhook delivery                  66
  cli.py              command surface                  642
  agents/             16 specialist agents           ~1400
  providers/          market/news/econ/edgar/options ~1400
  stats/              bayes, calibration, MC, TA     ~1670
  memory/             SQLite store + learning engine   641
  models/             logistic regression              203
  backtest/           point-in-time replay + validation ~1250
tests/                103 tests                       ~1350
```

Total ~11,000 lines.

---

## 2. Existing strengths — DO NOT REWRITE

These components are correct, tested, and load-bearing. The 2.0 work must
build **around** them, not replace them.

| Component | Why it stays |
|---|---|
| `clock.py` | NYSE calendar computed from first principles, incl. observed-date and Good Friday rules. 12 tests. Zero dependencies. Correct. |
| `stats/bayes.py` | Logit pooling with correlation haircut and Beta-Binomial reliability. The mathematical core. |
| `stats/distributions.py` | Student-t barrier Monte Carlo, Black-Scholes, cached scale-free simulation (150× speedup). |
| `stats/calibration.py` | Brier decomposition, Platt recalibration, SPRT. |
| `backtest/datastore.py` | Point-in-time store that *raises* on lookahead. The single most valuable safety property in the repo. |
| `backtest/validation.py` | Purged walk-forward, deflated Sharpe, CSCV/PBO. |
| `providers/*` | Degrade-not-crash discipline throughout. |
| `memory/store.py` | Working SQLite schema with real history. Must be migrated, never dropped. |
| `agents/base.py` | Timeout + failure containment per agent. Sound contract. |

---

## 3. Weaknesses

### 3.1 Red Team has no decision path — CRITICAL

`RedTeamAgent` runs last, produces objections, and **nothing consumes them.**
The playbook is already final by the time it is criticised. Even
`recommend_stand_down` is advisory prose. A high-severity objection cannot
lower a confidence score, modify a bracket, or remove an idea.

This is the highest-priority defect. Criticism that cannot change the output is
theatre.

### 3.2 Orchestration is static

`stage_agents()` topologically sorts a **fixed roster** and runs all 16 agents
every session regardless of conditions. A quiet day for AAPL costs exactly as
much as an NVDA earnings gap. There is no mechanism to:
- notice that something unusual happened,
- decide what to investigate,
- select a subset of specialists,
- request follow-up evidence,
- stop early when evidence is sufficient.

### 3.3 Evidence is a flat list

`Evidence` carries `claim / source / url / reliability / tags`. There is no
graph: no way to express that evidence A *contradicts* B, that claim C
*supports* hypothesis H, or that two items derive from the same underlying
observation. Provenance exists; structure does not.

### 3.4 One reliability number conflates distinct properties

`SOURCE_RELIABILITY` collapses "is this true?" with "does this predict
anything?". SEC EDGAR scores 0.98 — correct as *factual reliability*, and
close to meaningless as *predictive utility*. Timeliness, novelty, market
impact and independence are not represented at all.

### 3.5 Correlation handling is a single global constant

`signal_correlation = 0.35` is applied uniformly. In reality futures / global
indices / breadth are ~0.9 correlated with each other and ~0.1 with an SEC
filing. One scalar cannot express that. The haircut is directionally right and
quantitatively crude.

### 3.6 Learning measures correctness, not contribution

`_update_agent_weights` credits an agent when its LLR pointed the right way.
That rewards agents that agree with the outcome, **not** agents that improved
the forecast. An agent that always says +0.5 in a bull market scores well while
adding nothing. There is no ablation.

### 3.7 Reliability is context-free

One scalar weight per agent. The options agent is plausibly excellent into
earnings and useless in a macro shock; the current model cannot represent that.

### 3.8 No institutional memory

`lessons` is a single flat table of mined conditional statements. There is no
episodic record, no per-company memory, no mistake taxonomy, no experiment
history, no staleness or contradiction tracking.

### 3.9 No experiment isolation

Any change to weights or thresholds edits production directly. There is no
champion/challenger separation and no promotion gate.

### 3.10 Forced binary output

Every idea becomes long or short. There is no vocabulary for
`INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`, or `NO_EDGE`. The
`clears_bar` flag is the only hedge, and it is about cost, not knowledge.

### 3.11 Observability is print statements

No run/investigation IDs, no per-agent latency or cost accounting, no
structured logs, no queryable status.

### 3.12 No schema migrations

`store.py` runs `CREATE TABLE IF NOT EXISTS`. Adding a column to an existing
install would silently do nothing.

---

## 4. Technical debt

- `stats/regime.py` classifies with hard thresholds and returns no confidence.
- `report.py` `SECTION_ORDER` is a hand-maintained list; a new agent renders
  nowhere unless someone remembers to add it (this exact bug shipped in 1.1.1
  with the red team).
- `orchestrator.py` `_persist_predictions` reaches into agent `data` dicts by
  string key — untyped coupling.
- `agents/synthesis.py` is 500+ lines doing fusion, bracket construction,
  option selection and ranking.
- `cli.py` at 642 lines mixes argument parsing with business logic.

---

## 5. Data limitations

- Free chain data: volume and OI only. "Options flow" is **inferred**, not
  observed. Correctly disclaimed in the report, but the ceiling is real.
- Daily bars only in backtest — cannot order intraday high vs low, forcing the
  pessimistic tie-break.
- 13F lags 45 days and cannot inform an intraday decision.
- No intraday bars, no microstructure, no order book.

---

## 6. Security review (pre-2.0)

| Area | Finding |
|---|---|
| Secrets | Env-only, never serialised. `to_yaml()` excludes keys — verified by test. **Good.** |
| Subprocess | None. No shell execution anywhere. **Good.** |
| SQL | Parameterised throughout. **Good.** |
| Network | Outbound only, no inbound listener. |
| Untrusted input | **Gap:** RSS/EDGAR/news text flows into LLM prompts with no sanitisation. Prompt-injection surface. |
| Path handling | **Gap:** report paths derive from config without traversal checks. |
| Webhook | URL from env, no signing, no allowlist. |
| systemd | Hardened: `ProtectSystem=strict`, empty capability set, syscall filter. **Good.** |

---

## 7. Reliability concerns

- Per-agent timeout and containment exist. **No circuit breaker** — a dead
  provider is retried on every symbol, every run.
- No distinction between *degraded* and *critically degraded*. A run that lost
  the options agent still publishes option ideas built on nothing.
- Missing evidence is absent rather than explicitly represented.

---

## 8. Measured performance (the honest baseline)

From `FINDINGS.md`, 36,581 point-in-time decisions, 2013–2018, purged
walk-forward:

```
Net mean         −0.0712R per trade
Hit rate          43.4%
Brier skill      −0.0070   (no skill vs. base rate)
PBO               0.457    (winner is partly luck)
Deflated Sharpe   0.131    (not significant)
```

**MarketSwarm 1.1.2 has no demonstrated predictive edge.** The 2.0 work is an
architecture upgrade. It is not expected, and must not be claimed, to create an
edge that the data does not support.

---

## 9. What 2.0 must change, ranked

1. Red Team → Review Gate → Revision (criticism must bite)
2. Chief Investigator + Event Brain (spend effort where it matters)
3. Evidence Graph with multi-dimensional scoring
4. Cluster-aware correlation instead of one constant
5. Contextual, contribution-based learning
6. Institutional + mistake memory
7. Recommendation vocabulary including "we don't know"
8. Champion/challenger isolation
9. Observability and migrations
10. Security hardening of untrusted text paths

---

*Audit complete. Baseline preserved at tag `v1.1.2-baseline`.*
