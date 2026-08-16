"""Synthetic providers so the full swarm can be exercised without network."""

from __future__ import annotations

import asyncio
import datetime as dt

import numpy as np

from marketswarm.providers.earnings import EarningsEvent
from marketswarm.providers.econ import EconEvent
from marketswarm.providers.edgar import Filing
from marketswarm.providers.market import OHLCV, Quote
from marketswarm.providers.news import Headline
from marketswarm.providers.options import Chain, Contract


class FakeClient:
    """Mimics DataClient.gather without any I/O."""

    async def gather(self, coros, label: str = "batch"):
        results = await asyncio.gather(*coros, return_exceptions=True)
        return [None if isinstance(r, Exception) else r for r in results]


def synthetic_history(symbol: str, seed: int = 0, days: int = 180,
                      start: float = 100.0, drift: float = 0.0006,
                      vol: float = 0.012) -> OHLCV:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, days)
    closes = start * np.exp(np.cumsum(rets))
    highs = closes * (1 + np.abs(rng.normal(0, 0.004, days)))
    lows = closes * (1 - np.abs(rng.normal(0, 0.004, days)))
    opens = np.concatenate([[start], closes[:-1]])
    volumes = rng.uniform(4e7, 9e7, days)
    base = int(dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc).timestamp())
    return OHLCV(
        symbol=symbol,
        timestamps=[base + i * 86400 for i in range(days)],
        opens=opens, highs=highs, lows=lows, closes=closes, volumes=volumes,
    )


def synthetic_chain(symbol: str, price: float, expiration: dt.date | None = None) -> Chain:
    exp = expiration or dt.date.today()
    strikes = [round(price * (1 + k / 100), 0) for k in range(-6, 7)]
    calls, puts = [], []
    for k in strikes:
        otm_c = max(0.0, price - k)
        otm_p = max(0.0, k - price)
        base = max(0.35, price * 0.004)
        calls.append(Contract(strike=k, last=base + otm_c, bid=base + otm_c - 0.03,
                              ask=base + otm_c + 0.03, volume=int(6000 - abs(k - price) * 90),
                              open_interest=int(14000 - abs(k - price) * 200),
                              implied_volatility=0.19 + abs(k - price) / price * 0.5,
                              in_the_money=k < price, kind="call"))
        puts.append(Contract(strike=k, last=base + otm_p, bid=base + otm_p - 0.03,
                             ask=base + otm_p + 0.03, volume=int(5200 - abs(k - price) * 80),
                             open_interest=int(15500 - abs(k - price) * 190),
                             implied_volatility=0.23 + abs(k - price) / price * 0.5,
                             in_the_money=k > price, kind="put"))
    return Chain(symbol, price, exp, 0.4, calls, puts)


class FakeMarket:
    def __init__(self, universe, gaps=None, seed=0):
        self.client = FakeClient()
        self.universe = universe
        self.seed = seed
        self.gaps = gaps or {}
        self._hist = {
            s: synthetic_history(s, seed=seed + i, drift=0.0009 if i % 3 else -0.0004)
            for i, s in enumerate(universe + ["SPY", "QQQ", "IWM"])
        }

    async def premarket_profile(self, symbol):
        h = self._hist.get(symbol)
        if h is None:
            return None
        prev = float(h.closes[-1])
        gap = self.gaps.get(symbol, 0.6 if symbol in ("SPY", "QQQ", "NVDA") else -0.4)
        return {
            "symbol": symbol, "previous_close": prev, "last": prev * (1 + gap / 100),
            "gap_pct": gap, "premarket_high": prev * (1 + abs(gap) / 100 + 0.001),
            "premarket_low": prev * (1 - 0.002), "premarket_volume": 850_000,
            "premarket_bars": 60, "market_state": "PRE",
        }

    async def history(self, symbol, range_="6mo", interval="1d"):
        return self._hist.get(symbol)

    async def _board(self, mapping):
        out = {}
        for i, (sym, name) in enumerate(mapping.items()):
            px = 100 + i * 7
            chg = ((i % 5) - 2) * 0.3
            out[sym] = Quote(sym, name, px * (1 + chg / 100), px, px * chg / 100, chg)
        return out

    async def global_board(self):
        from marketswarm.providers.market import GLOBAL_INDICES
        return await self._board(GLOBAL_INDICES)

    async def futures_board(self):
        from marketswarm.providers.market import FUTURES
        board = await self._board(FUTURES)
        board["ES=F"] = Quote("ES=F", "S&P 500 E-mini", 5432.0, 5410.0, 22.0, 0.41)
        board["NQ=F"] = Quote("NQ=F", "Nasdaq 100 E-mini", 19200.0, 19080.0, 120.0, 0.63)
        board["RTY=F"] = Quote("RTY=F", "Russell 2000 E-mini", 2210.0, 2205.0, 5.0, 0.23)
        return board

    async def volatility_board(self):
        return {
            "^VIX": Quote("^VIX", "VIX", 14.8, 15.6, -0.8, -5.1),
            "^VIX3M": Quote("^VIX3M", "VIX3M", 17.2, 17.3, -0.1, -0.6),
        }

    async def rates_fx_board(self):
        return {
            "^TNX": Quote("^TNX", "US 10Y yield", 4.21, 4.19, 0.02, 0.48),
            "EURUSD=X": Quote("EURUSD=X", "EUR/USD", 1.09, 1.088, 0.002, 0.18),
        }


class FakeOptions:
    def __init__(self, market: FakeMarket):
        self.client = FakeClient()
        self.market = market

    async def flow_snapshot(self, symbol):
        prof = await self.market.premarket_profile(symbol)
        if not prof:
            return None
        price = prof["last"]
        chain = synthetic_chain(symbol, price)
        im = chain.implied_move()
        return {
            "symbol": symbol, "underlying": price,
            "expiration": chain.expiration.isoformat(),
            "days_to_expiry": chain.days_to_expiry,
            "implied_move_pct": round(im.implied_move_pct * 100, 2) if im else None,
            "implied_move_abs": round(im.implied_move_abs, 2) if im else None,
            "implied_vol_annual": round(im.implied_vol_annual * 100, 1) if im else None,
            "put_call_volume_ratio": round(chain.put_call_volume_ratio, 2),
            "put_call_oi_ratio": round(chain.put_call_oi_ratio, 2),
            "skew": chain.skew_25d(), "walls": chain.oi_walls(),
            "unusual": chain.unusual_activity(),
            "liquidity_score": chain.liquidity_score(), "chain": chain,
        }


class FakeNews:
    async def overnight_headlines(self, max_age_hours=18.0):
        now = dt.datetime.now(dt.timezone.utc)
        return [
            Headline("Fed officials signal patience on rate cuts", "Reuters Markets",
                     "major_newswire", 0.85, "https://example.test/1", now - dt.timedelta(hours=3),
                     materiality="high", corroborations=3),
            Headline("NVDA raises full-year outlook after data-centre beat", "CNBC Markets",
                     "financial_media", 0.72, "https://example.test/2", now - dt.timedelta(hours=9),
                     tickers=["NVDA"], materiality="high", corroborations=2),
            Headline("Oil steadies as inventories draw", "Yahoo Finance", "aggregator", 0.62,
                     "https://example.test/3", now - dt.timedelta(hours=5), materiality="medium"),
        ]

    async def ticker_news(self, ticker, limit=8):
        return []


class FakeEcon:
    async def todays_calendar(self, day):
        return [EconEvent("Initial Jobless Claims", day, "08:30", "medium")]

    async def macro_dashboard(self):
        return {"available": True, "series": {
            "DGS10": {"label": "10-year Treasury", "value": 4.21, "date": "2026-08-14", "previous": 4.19}
        }}

    async def treasury_curve(self):
        return {"tenors": {"2YEAR": 3.92, "10YEAR": 4.21, "30YEAR": 4.48},
                "source": "U.S. Treasury par yield curve"}


class FakeEdgar:
    def __init__(self):
        self.client = FakeClient()

    async def scan_universe(self, tickers, lookback_hours=24.0):
        now = dt.datetime.now(dt.timezone.utc)
        return [
            Filing("NVDA", "NVIDIA Corp", "8-K", now - dt.timedelta(hours=11),
                   "0001045810-26-000123", "https://example.test/8k", items=["2.02", "7.01"]),
        ]

    async def insider_transactions(self, ticker, lookback_days=30):
        if ticker == "AMD":
            return {"ticker": ticker, "count": 5, "most_recent": "2026-08-12",
                    "urls": ["https://example.test/form4"],
                    "signal": "5 Form 4 filings in 30d — cluster activity worth reading",
                    "caveat": "10b5-1 plan sales carry no signal."}
        return {"ticker": ticker, "count": 0, "signal": "no recent Form 4 activity"}

    async def institutional_13f_hint(self, ticker):
        return {"ticker": ticker, "recent_13d_13g": [], "signal": "no recent >5% ownership changes on file",
                "staleness_warning": "13F lags 45 days."}


class FakeEarnings:
    async def relevant_window(self, today, prev_session, min_market_cap=5e9):
        return {
            "reacting_today": [
                EarningsEvent("NVDA", "NVIDIA Corp", prev_session, "amc",
                              eps_estimate=1.10, eps_actual=1.25, surprise_pct=13.6, market_cap=3e12)
            ],
            "reporting_tonight": [
                EarningsEvent("AMD", "Advanced Micro Devices", today, "amc",
                              eps_estimate=0.92, market_cap=2.6e11)
            ],
            "counts": {"prior_amc": 1, "today_bmo": 0, "today_amc": 1},
        }
