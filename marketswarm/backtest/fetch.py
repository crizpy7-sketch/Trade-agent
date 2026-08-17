"""Historical data acquisition.

Several sources, deliberately ordered by what survives a restricted network:

  github_sp500   505 S&P names, daily OHLCV, 2013-2018. A static file on a CDN,
                 so it works from anywhere and is byte-identical every time —
                 which also makes backtests reproducible across machines.
  yahoo          Any symbol, any range, but rate-limited and occasionally
                 reshapes its JSON without notice.
  stooq          CSV, no key, good coverage of US equities and indices.
  csv            Whatever you already have on disk.

Reproducibility matters more than freshness here: a backtest you cannot re-run
to the same number is not evidence of anything.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("marketswarm.backtest.fetch")

SP500_5YR = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"
SPY_10YR = ("https://raw.githubusercontent.com/matplotlib/mplfinance/master/"
            "examples/data/yahoofinance-SPY-20080101-20180101.csv")
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
STOOQ = "https://stooq.com/q/d/l/"


def _http_get(url: str, params: dict | None = None, timeout: float = 300) -> bytes:
    import httpx

    with httpx.Client(timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": "marketswarm/1.0 (backtest data fetch)"}) as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        return r.content


def fetch_github_sp500(cache_dir: Path | None = None) -> pd.DataFrame:
    """505 S&P 500 constituents, daily OHLCV, Feb 2013 - Feb 2018 (~619k rows)."""
    cache = Path(cache_dir).expanduser() / "all_stocks_5yr.csv" if cache_dir else None
    if cache and cache.exists():
        log.info("using cached %s", cache)
        raw = cache.read_bytes()
    else:
        log.info("downloading S&P 500 five-year daily set (~29 MB)")
        raw = _http_get(SP500_5YR)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(raw)

    df = pd.read_csv(io.BytesIO(raw))
    df = df.rename(columns={"Name": "symbol"})
    return df[["symbol", "date", "open", "high", "low", "close", "volume"]]


def fetch_github_spy(cache_dir: Path | None = None) -> pd.DataFrame:
    """SPY daily OHLCV 2008-2018 — spans the GFC, 2011 and 2015 vol shocks.

    Worth having precisely because it contains regimes the 2013-2018 set does
    not: a strategy validated only on a bull market has been validated on
    nothing.
    """
    cache = Path(cache_dir).expanduser() / "spy_2008_2018.csv" if cache_dir else None
    if cache and cache.exists():
        raw = cache.read_bytes()
    else:
        log.info("downloading SPY 2008-2018 daily")
        raw = _http_get(SPY_10YR)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(raw)

    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["symbol"] = "SPY"
    return df[["symbol", "date", "open", "high", "low", "close", "volume"]]


def fetch_yahoo(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame | None:
    """Daily bars from Yahoo. Works on an unrestricted network (i.e. your VPS)."""
    try:
        import json

        raw = _http_get(
            YAHOO_CHART.format(symbol=symbol),
            params={
                "period1": int(dt.datetime.combine(start, dt.time()).timestamp()),
                "period2": int(dt.datetime.combine(end, dt.time()).timestamp()),
                "interval": "1d",
            },
        )
        payload = json.loads(raw)
        result = payload["chart"]["result"][0]
        q = result["indicators"]["quote"][0]
        stamps = result["timestamp"]
    except Exception as exc:  # noqa: BLE001
        log.warning("yahoo fetch failed for %s: %s", symbol, exc)
        return None

    df = pd.DataFrame({
        "date": [dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).date() for t in stamps],
        "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
        "close": q.get("close"), "volume": q.get("volume"),
    })
    df["symbol"] = symbol.upper()
    return df.dropna(subset=["close"])[["symbol", "date", "open", "high", "low", "close", "volume"]]


def fetch_stooq(symbol: str) -> pd.DataFrame | None:
    """Full daily history from Stooq. US tickers take a `.us` suffix."""
    sym = symbol.lower()
    if "." not in sym:
        sym += ".us"
    try:
        raw = _http_get(STOOQ, params={"s": sym, "i": "d"})
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        log.warning("stooq fetch failed for %s: %s", symbol, exc)
        return None
    if df.empty or "Close" not in df.columns:
        return None
    df.columns = [c.strip().lower() for c in df.columns]
    df["symbol"] = symbol.upper()
    return df[["symbol", "date", "open", "high", "low", "close", "volume"]]


def fetch_csv(path: Path, symbol: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "symbol" not in df.columns:
        if not symbol:
            raise ValueError("CSV has no symbol column — pass --symbol")
        df["symbol"] = symbol.upper()
    if "name" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"name": "symbol"})
    return df[["symbol", "date", "open", "high", "low", "close", "volume"]]


def load_into_store(store, source: str, symbols: list[str] | None = None,
                    start: dt.date | None = None, end: dt.date | None = None,
                    cache_dir: Path | None = None, path: Path | None = None) -> dict:
    """Fetch from `source` and ingest. Returns a coverage summary."""
    if source == "github_sp500":
        df = fetch_github_sp500(cache_dir)
        if symbols:
            df = df[df["symbol"].isin([s.upper() for s in symbols])]
    elif source == "github_spy":
        df = fetch_github_spy(cache_dir)
    elif source == "yahoo":
        if not symbols:
            raise ValueError("yahoo source needs --symbols")
        start = start or dt.date.today() - dt.timedelta(days=365 * 5)
        end = end or dt.date.today()
        frames = [f for f in (fetch_yahoo(s, start, end) for s in symbols) if f is not None]
        if not frames:
            raise RuntimeError("no data retrieved from yahoo")
        df = pd.concat(frames, ignore_index=True)
    elif source == "stooq":
        if not symbols:
            raise ValueError("stooq source needs --symbols")
        frames = [f for f in (fetch_stooq(s) for s in symbols) if f is not None]
        if not frames:
            raise RuntimeError("no data retrieved from stooq")
        df = pd.concat(frames, ignore_index=True)
    elif source == "csv":
        if not path:
            raise ValueError("csv source needs --path")
        df = fetch_csv(Path(path).expanduser())
    else:
        raise ValueError(f"unknown source {source!r}")

    if start is not None:
        df = df[pd.to_datetime(df["date"]).dt.date >= start]
    if end is not None:
        df = df[pd.to_datetime(df["date"]).dt.date <= end]

    n = store.ingest_frame(df, source=source)
    log.info("ingested %d bars from %s", n, source)
    return store.coverage()
