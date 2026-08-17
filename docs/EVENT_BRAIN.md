# Event Brain

Cheap broad scan → anomaly detection → targeted depth.

## Detectors

price gap · relative volume · volatility spike · index move · earnings reaction ·
earnings upcoming · SEC filing · macro release · options anomaly ·
sector divergence · company news · correlation breakdown

Priorities: `LOW`, `NORMAL`, `ELEVATED`, `HIGH`, `CRITICAL`.

## Gaps are normalised by the symbol's own volatility

A raw percentage compares nothing across symbols. A 2% gap on a 0.8%-ATR name
is a 2.5σ event; on a 4%-ATR name it is noise. Both are tested.

Thresholds live in `TriggerThresholds` — one dataclass, tunable, not buried in
a function.

## Inferred vs observed

Options anomalies are tagged `inferred: True` with an explicit caveat:
*derived from volume/open-interest, not observed trade flow*. The system never
claims retail chain data is order flow.

## Example

```
NVDA  gap -7.2% (3.0 ATR) + rvol 5.5 + 8-K + P/C 1.9   → HIGH, deep investigation
JPM   earnings reaction, surprise +8.4%                → HIGH, earnings workflow
AAPL  gap +0.1%, rvol 1.0                              → quiet, cheap pass
SPY   CPI at 08:30                                     → CRITICAL, macro workflow
```
