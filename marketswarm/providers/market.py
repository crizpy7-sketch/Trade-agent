"""Quotes, OHLCV history, futures, indices and volatility products.

Primary transport is the public Yahoo Finance chart/quote API (no key). Every
call degrades to None rather than raising, so a partial outage narrows the
report instead of killing the run.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np

from .base import DataClient, Evidence, ProviderError

log = logging.getLogger("marketswarm.market")

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Index futures — the pre-market read on U.S. direction.
FUTURES = {
    "ES=F": "S&P 500 E-mini",
    "NQ=F": "Nasdaq 100 E-mini",
    "YM=F": "Dow E-mini",
    "RTY=F": "Russell 2000 E-mini",
    "CL=F": "WTI Crude",
    "GC=F": "Gold",
    "SI=F": "Silver",
    "ZN=F": "10Y T-Note",
    "DX=F": "US Dollar Index",
    "BTC-USD": "Bitcoin",
}

# Overnight global session — closed before the U.S. open, so these are facts
# rather than forecasts by the time the report is written.
GLOBAL_INDICES = {
    "^N225": "Nikkei 225 (Japan)",
    "^HSI": "Hang Seng (Hong Kong)",
    "000001.SS": "Shanghai Composite",
    "^KS11": "KOSPI (South Korea)",
    "^AXJO": "ASX 200 (Australia)",
    "^STOXX50E": "Euro Stoxx 50",
    "^GDAXI": "DAX (Germany)",
    "^FTSE": "FTSE 100 (UK)",
    "^FCHI": "CAC 40 (France)",
}

VOLATILITY = {"^VIX": "VIX", "^VIX3M": "VIX3M", "^VVIX": "VVIX", "^VXN": "VXN (Nasdaq)"}

RATES_FX = {"^TNX": "US 10Y yield", "^FVX": "US 5Y yield", "^IRX": "US 13W bill",
            "EURUSD=X": "EUR/USD", "USDJPY=X": "USD/JPY"}


@dataclass
class Quote:
    symbol: str
    name: str
    price: float
    previous_close: float
    change: float
    change_pct: float
    day_high: float | None = None
    day_low: float | None = None
    volume: float | None = None
    market_state: str = "UNKNOWN"
    currency: str = "USD"
    as_of: str = ""

    @property
    def direction(self) -> str:
        return "up" if self.change_pct > 0.05 else "down" if self.change_pct < -0.05 else "flat"

    def to_evidence(self) -> Evidence:
        return Evidence(
            claim=f"{self.name} ({self.symbol}) {self.change_pct:+.2f}% at {self.price:,.2f}",
            source="exchange_data",
            url=f"https://finance.yahoo.com/quote/{self.symbol}",
            reliability=0.93,
            value={"symbol": self.symbol, "price": self.price, "change_pct": self.change_pct},
            tags=["quote"],
        )


@dataclass
class OHLCV:
    symbol: str
    timestamps: list[int] = field(default_factory=list)
    opens: np.ndarray = field(default_factory=lambda: np.array([]))
    highs: np.ndarray = field(default_factory=lambda: np.array([]))
    lows: np.ndarray = field(default_factory=lambda: np.array([]))
    closes: np.ndarray = field(default_factory=lambda: np.array([]))
    volumes: np.ndarray = field(default_factory=lambda: np.array([]))
    interval: str = "1d"

    def __len__(self) -> int:
        return int(self.closes.size)

    @property
    def last(self) -> float:
        return float(self.closes[-1]) if self.closes.size else float("nan")

    def returns(self) -> np.ndarray:
        return np.diff(np.log(self.closes)) if self.closes.size > 1 else np.array([])


def _extract_quote(payload: dict, symbol: str, name: str | None = None) -> Quote | None:
    try:
        result = payload["chart"]["result"][0]
        meta = result["meta"]
        price = meta.get("regularMarketPrice")
        prev = meta.get("chartPreviousClose") or meta.get("previousClose")
        if price is None or prev in (None, 0):
            return None
        return Quote(
            symbol=symbol,
            name=name or meta.get("shortName") or symbol,
            price=float(price),
            previous_close=float(prev),
            change=float(price) - float(prev),
            change_pct=(float(price) / float(prev) - 1) * 100,
            day_high=meta.get("regularMarketDayHigh"),
            day_low=meta.get("regularMarketDayLow"),
            volume=meta.get("regularMarketVolume"),
            market_state=meta.get("marketState", "UNKNOWN"),
            currency=meta.get("currency", "USD"),
            as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        log.debug("quote parse failed for %s: %s", symbol, exc)
        return None


class MarketData:
    def __init__(self, client: DataClient):
        self.client = client

    async def quote(self, symbol: str, name: str | None = None) -> Quote | None:
        try:
            payload = await self.client.get_json(
                CHART.format(symbol=symbol),
                params={"range": "5d", "interval": "1d", "includePrePost": "true"},
            )
        except ProviderError as exc:
            log.warning("quote %s unavailable: %s", symbol, exc)
            return None
        return _extract_quote(payload, symbol, name)

    async def quotes(self, symbols: dict[str, str]) -> dict[str, Quote]:
        results = await self.client.gather(
            [self.quote(sym, nm) for sym, nm in symbols.items()], label="quotes"
        )
        return {q.symbol: q for q in results if isinstance(q, Quote)}

    async def history(self, symbol: str, range_: str = "6mo", interval: str = "1d") -> OHLCV | None:
        try:
            payload = await self.client.get_json(
                CHART.format(symbol=symbol),
                params={"range": range_, "interval": interval, "includePrePost": "false"},
            )
            result = payload["chart"]["result"][0]
            q = result["indicators"]["quote"][0]
            ts = result["timestamp"]
        except (ProviderError, KeyError, IndexError, TypeError) as exc:
            log.warning("history %s unavailable: %s", symbol, exc)
            return None

        def arr(key: str) -> np.ndarray:
            vals = q.get(key) or []
            return np.array([v if v is not None else np.nan for v in vals], dtype=float)

        closes = arr("close")
        mask = ~np.isnan(closes)
        if mask.sum() < 2:
            return None
        return OHLCV(
            symbol=symbol,
            timestamps=[t for t, m in zip(ts, mask) if m],
            opens=np.nan_to_num(arr("open")[mask], nan=0.0),
            highs=np.nan_to_num(arr("high")[mask], nan=0.0),
            lows=np.nan_to_num(arr("low")[mask], nan=0.0),
            closes=closes[mask],
            volumes=np.nan_to_num(arr("volume")[mask], nan=0.0),
            interval=interval,
        )

    async def premarket_profile(self, symbol: str) -> dict | None:
        """Pre/post-market extremes from 5-minute bars including extended hours.

        The overnight high/low are the first levels the regular session reacts
        to, and the overnight volume tells you whether the gap has sponsorship.
        """
        try:
            payload = await self.client.get_json(
                CHART.format(symbol=symbol),
                params={"range": "2d", "interval": "5m", "includePrePost": "true"},
                use_cache=False,
            )
            result = payload["chart"]["result"][0]
            meta, q = result["meta"], result["indicators"]["quote"][0]
            ts = result["timestamp"]
        except (ProviderError, KeyError, IndexError, TypeError) as exc:
            log.warning("premarket %s unavailable: %s", symbol, exc)
            return None

        prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
        pre_start = (meta.get("currentTradingPeriod", {}).get("pre", {}) or {}).get("start")
        reg_start = (meta.get("currentTradingPeriod", {}).get("regular", {}) or {}).get("start")

        highs, lows, vols, closes = [], [], [], []
        for i, t in enumerate(ts):
            if pre_start and t < pre_start:
                continue
            if reg_start and t >= reg_start:
                continue
            h, l, v, c = (q.get(k) or [None] * len(ts) for k in ("high", "low", "volume", "close"))
            if h[i] is None or l[i] is None:
                continue
            highs.append(h[i]); lows.append(l[i]); vols.append(v[i] or 0); closes.append(c[i])

        last = meta.get("regularMarketPrice")
        gap_pct = (last / prev_close - 1) * 100 if last and prev_close else None
        return {
            "symbol": symbol,
            "previous_close": prev_close,
            "last": last,
            "gap_pct": gap_pct,
            "premarket_high": max(highs) if highs else None,
            "premarket_low": min(lows) if lows else None,
            "premarket_volume": sum(vols) if vols else 0,
            "premarket_bars": len(highs),
            "market_state": meta.get("marketState"),
        }

    async def futures_board(self) -> dict[str, Quote]:
        return await self.quotes(FUTURES)

    async def global_board(self) -> dict[str, Quote]:
        return await self.quotes(GLOBAL_INDICES)

    async def volatility_board(self) -> dict[str, Quote]:
        return await self.quotes(VOLATILITY)

    async def rates_fx_board(self) -> dict[str, Quote]:
        return await self.quotes(RATES_FX)

    async def breadth_proxy(self, universe: list[str]) -> dict:
        """Share of a mega-cap universe gapping green — a crude but honest
        pre-market breadth read, since real advance/decline data is not
        available before the open."""
        quotes = await self.quotes({s: s for s in universe})
        if not quotes:
            return {"n": 0, "pct_up": None}
        ups = sum(1 for q in quotes.values() if q.change_pct > 0)
        return {
            "n": len(quotes),
            "pct_up": ups / len(quotes),
            "median_change_pct": float(np.median([q.change_pct for q in quotes.values()])),
            "leaders": sorted(quotes.values(), key=lambda q: -q.change_pct)[:3],
            "laggards": sorted(quotes.values(), key=lambda q: q.change_pct)[:3],
        }
