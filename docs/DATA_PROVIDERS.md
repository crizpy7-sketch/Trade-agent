# Data Providers

## Current (free tier)

| Layer | Source | Reliability | Limitation |
|---|---|---|---|
| Quotes, OHLCV | public chart API | 0.93 | 15-min delay possible; daily bars in backtest |
| Option chains | public options API | 0.90 | volume + OI only — **no trade tape** |
| Filings | SEC EDGAR | 0.98 | requires a declared contact |
| Rates / macro | Treasury, FRED | 0.97 | FRED needs a free key |
| News | RSS (Reuters, AP, CNBC, MarketWatch) | 0.58–0.85 | no timestamps finer than publication |
| Earnings | Nasdaq calendar | 0.88 | timing occasionally wrong |

## Observed vs inferred

The distinction is enforced in the data model. `EvidenceNode.is_inferred` and
`DetectedEvent.evidence["inferred"]` mark anything derived rather than
measured, and the report says so.

**"Options flow" from a free chain is inferred.** Volume/OI turnover cannot
distinguish an institutional buy from a dealer hedge or a closing trade. The
system never claims otherwise.

## Adapter architecture

Providers are constructor-injected (`MarketData(client)`, `OptionsData(client)`
…) so a paid feed is a drop-in replacement, not a rewrite. `backtest/fetch.py`
already implements four interchangeable historical sources
(`github_sp500`, `github_spy`, `yahoo`, `stooq`, `csv`).

## Graceful degradation

Every provider returns `None` rather than raising. A dead source narrows the
report and is named in it. Circuit breakers stop retrying a source that is
down.

## Upgrade path, in value order

1. **Real options data** (Polygon.io, Databento) — trades with exchange and
   condition codes, so buy/sell classification and sweep detection become
   observation rather than inference. The single largest missing input.
2. **Intraday bars** — daily OHLC cannot order the high and low, which forces
   the pessimistic tie-break and discards real information.
3. **Microstructure** — order book imbalance, Lee-Ready classification, VPIN.
4. **Timestamped news** with publication-time precision.
