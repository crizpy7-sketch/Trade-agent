# Testing

```bash
pytest -q                      # full suite
pytest tests/test_review.py -q # one area
```

## Coverage

| File | Tests | Area |
|---|---|---|
| `test_clock.py` | 12 | NYSE calendar, observed holidays, Good Friday, sessions |
| `test_stats.py` | 42 | Bayes, calibration, distributions, technicals, edge, regime |
| `test_config.py` | 6 | config precedence, secret handling |
| `test_backtest.py` | 28 | point-in-time store, purged folds, deflation, PBO, fills |
| `test_pipeline.py` | 17 | 1.x swarm end-to-end on synthetic providers |
| `test_review.py` | 27 | review gate and revision loop |
| `test_pipeline2.py` | 39 | event brain, chief, evidence graph, engine, resilience, integration |
| `test_memory2.py` | 38 | migrations, memory, contribution, AAR, lab, router, API |
| `test_adversarial.py` | 32 | the gauntlet |
| **Total** | **239** | |

## Negative tests

The most valuable ones assert that something is *impossible*:

- the store **refuses** lookahead (`LookaheadError`)
- a CRITICAL finding can **never** be approved, under any policy
- the research scientist has **no** promote method
- a challenger **cannot** mutate the champion
- unknown tools are denied by default
- no `subprocess`/`eval`/`exec` exists anywhere in the package
- unknown evidence scores stay `None` rather than becoming 0.5

## The gauntlet

`test_adversarial.py` asks hostile questions: how could correlated signals fake
confidence? how could stale data look current? how could an LLM invent
evidence? what if half the providers die? Findings are fixed in code, not
documented as caveats — three real bugs were found and fixed this way.

## Determinism

Monte Carlo seeds are fixed. Same inputs produce byte-identical reports.
Historical data comes from static files, so backtests are reproducible across
machines.
