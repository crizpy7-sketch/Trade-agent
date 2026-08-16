"""Option chains: implied move, skew, open-interest walls and flow proxies.

Retail-accessible chain data gives volume and open interest but not the
trade-by-trade tape, so "flow" here is inferred: volume/OI ratios, skew, and
where dealer gamma is likely concentrated. The report labels these as
inferences, never as observed institutional prints.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass, field

from ..stats.distributions import bs_price_and_greeks, implied_move_from_straddle
from .base import DataClient, ProviderError

log = logging.getLogger("marketswarm.options")

CHAIN = "https://query2.finance.yahoo.com/v7/finance/options/{symbol}"


@dataclass
class Contract:
    strike: float
    last: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    in_the_money: bool
    kind: str  # "call" | "put"

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return (self.ask - self.bid) / m * 100 if m > 0 and self.ask > self.bid else float("nan")

    @property
    def turnover(self) -> float:
        """Volume / open interest. Above ~1 means today's activity is new
        positioning rather than existing holders trading around a position."""
        return self.volume / self.open_interest if self.open_interest > 0 else float("nan")


@dataclass
class Chain:
    symbol: str
    underlying_price: float
    expiration: dt.date
    days_to_expiry: float
    calls: list[Contract] = field(default_factory=list)
    puts: list[Contract] = field(default_factory=list)

    def atm(self, kind: str = "call") -> Contract | None:
        pool = self.calls if kind == "call" else self.puts
        if not pool:
            return None
        return min(pool, key=lambda c: abs(c.strike - self.underlying_price))

    def nearest(self, strike: float, kind: str = "call") -> Contract | None:
        pool = self.calls if kind == "call" else self.puts
        if not pool:
            return None
        return min(pool, key=lambda c: abs(c.strike - strike))

    def by_delta(self, target_delta: float = 0.40, kind: str = "call") -> Contract | None:
        """Pick the strike closest to a target delta.

        Free chains omit greeks, so delta is computed from Black-Scholes using
        each contract's own implied volatility. Selecting by delta rather than
        by distance-from-spot is what keeps the choice consistent across names
        with very different volatilities: a 1% OTM strike is a 45-delta on SPY
        and a 25-delta on TSLA.
        """
        pool = self.calls if kind == "call" else self.puts
        pool = [c for c in pool if c.implied_volatility > 0 and c.mid > 0]
        if not pool:
            return None

        best, best_gap = None, float("inf")
        for c in pool:
            g = bs_price_and_greeks(
                self.underlying_price, c.strike, self.days_to_expiry,
                c.implied_volatility, kind,
            )
            gap = abs(abs(g["delta"]) - target_delta)
            if gap < best_gap:
                best, best_gap = c, gap
        return best

    def greeks_for(self, contract: Contract) -> dict:
        return bs_price_and_greeks(
            self.underlying_price, contract.strike, self.days_to_expiry,
            contract.implied_volatility, contract.kind,
        )

    @property
    def total_call_volume(self) -> int:
        return sum(c.volume for c in self.calls)

    @property
    def total_put_volume(self) -> int:
        return sum(p.volume for p in self.puts)

    @property
    def put_call_volume_ratio(self) -> float:
        return self.total_put_volume / self.total_call_volume if self.total_call_volume else float("nan")

    @property
    def put_call_oi_ratio(self) -> float:
        c = sum(x.open_interest for x in self.calls)
        p = sum(x.open_interest for x in self.puts)
        return p / c if c else float("nan")

    def implied_move(self):
        ac, ap = self.atm("call"), self.atm("put")
        if not ac or not ap:
            return None
        straddle = ac.mid + ap.mid
        if straddle <= 0:
            return None
        return implied_move_from_straddle(self.underlying_price, straddle, self.days_to_expiry)

    def skew_25d(self) -> dict:
        """OTM put IV minus OTM call IV at roughly equal distance.

        Positive and rising skew means downside protection is being bid — a
        genuine positioning signal that precedes many risk-off sessions.
        """
        s = self.underlying_price
        put = self.nearest(s * 0.95, "put")
        call = self.nearest(s * 1.05, "call")
        if not put or not call or put.implied_volatility <= 0 or call.implied_volatility <= 0:
            return {"skew": None, "read": "insufficient chain data"}
        skew = (put.implied_volatility - call.implied_volatility) * 100
        if skew > 6:
            read = "steep put skew — downside protection aggressively bid"
        elif skew < 0:
            read = "call skew — upside speculation bid (squeeze/melt-up behaviour)"
        else:
            read = "normal equity skew"
        return {
            "skew": round(skew, 2),
            "put_iv": round(put.implied_volatility * 100, 1),
            "call_iv": round(call.implied_volatility * 100, 1),
            "read": read,
        }

    def oi_walls(self, top_n: int = 3) -> dict:
        """Strikes with the largest open interest.

        Dealers hedging short options at these strikes tend to dampen movement
        into them (pinning) and accelerate it once through — so they act as
        soft magnets and, on a break, as trigger levels.
        """
        calls = sorted(self.calls, key=lambda c: -c.open_interest)[:top_n]
        puts = sorted(self.puts, key=lambda p: -p.open_interest)[:top_n]
        return {
            "call_walls": [{"strike": c.strike, "oi": c.open_interest} for c in calls],
            "put_walls": [{"strike": p.strike, "oi": p.open_interest} for p in puts],
            "max_pain_proxy": self._max_pain(),
        }

    def _max_pain(self) -> float | None:
        """Strike at which the most option value expires worthless."""
        strikes = sorted({c.strike for c in self.calls} | {p.strike for p in self.puts})
        if not strikes:
            return None
        best, best_pain = None, float("inf")
        for k in strikes:
            pain = sum(max(0.0, k - c.strike) * c.open_interest for c in self.calls)
            pain += sum(max(0.0, p.strike - k) * p.open_interest for p in self.puts)
            if pain < best_pain:
                best, best_pain = k, pain
        return best

    def unusual_activity(self, min_volume: int = 500, min_turnover: float = 2.0) -> list[dict]:
        """Contracts where today's volume dwarfs existing open interest —
        the closest a free feed gets to 'unusual options activity'."""
        out = []
        for c in self.calls + self.puts:
            t = c.turnover
            if c.volume >= min_volume and not math.isnan(t) and t >= min_turnover:
                out.append(
                    {
                        "kind": c.kind,
                        "strike": c.strike,
                        "volume": c.volume,
                        "open_interest": c.open_interest,
                        "turnover": round(t, 1),
                        "iv": round(c.implied_volatility * 100, 1),
                        "notional_estimate": round(c.volume * c.mid * 100),
                    }
                )
        out.sort(key=lambda x: -x["notional_estimate"])
        return out[:10]

    def liquidity_score(self) -> float:
        """0-1 tradability score from ATM spread and volume.

        An idea in a chain that cannot be entered and exited at a reasonable
        price is not an idea, and this is what down-weights it in ranking.
        """
        ac = self.atm("call")
        if not ac:
            return 0.0
        sp = ac.spread_pct
        spread_score = 1.0 if math.isnan(sp) else max(0.0, 1.0 - sp / 15.0)
        vol_score = min(1.0, (self.total_call_volume + self.total_put_volume) / 50_000)
        return round(0.6 * spread_score + 0.4 * vol_score, 3)


def _parse_contract(raw: dict, kind: str) -> Contract | None:
    try:
        return Contract(
            strike=float(raw["strike"]),
            last=float(raw.get("lastPrice") or 0),
            bid=float(raw.get("bid") or 0),
            ask=float(raw.get("ask") or 0),
            volume=int(raw.get("volume") or 0),
            open_interest=int(raw.get("openInterest") or 0),
            implied_volatility=float(raw.get("impliedVolatility") or 0),
            in_the_money=bool(raw.get("inTheMoney", False)),
            kind=kind,
        )
    except (KeyError, TypeError, ValueError):
        return None


class OptionsData:
    def __init__(self, client: DataClient):
        self.client = client

    async def expirations(self, symbol: str) -> list[dt.date]:
        try:
            payload = await self.client.get_json(CHAIN.format(symbol=symbol))
            stamps = payload["optionChain"]["result"][0].get("expirationDates", [])
        except (ProviderError, KeyError, IndexError, TypeError) as exc:
            log.warning("expirations %s unavailable: %s", symbol, exc)
            return []
        return [dt.datetime.fromtimestamp(s, tz=dt.timezone.utc).date() for s in stamps]

    async def chain(self, symbol: str, expiration: dt.date | None = None) -> Chain | None:
        params = {}
        if expiration:
            params["date"] = int(
                dt.datetime.combine(expiration, dt.time(0, 0), tzinfo=dt.timezone.utc).timestamp()
            )
        try:
            payload = await self.client.get_json(CHAIN.format(symbol=symbol), params=params, use_cache=False)
            result = payload["optionChain"]["result"][0]
            quote = result.get("quote", {})
            opt = result["options"][0]
        except (ProviderError, KeyError, IndexError, TypeError) as exc:
            log.warning("chain %s unavailable: %s", symbol, exc)
            return None

        under = float(quote.get("regularMarketPrice") or 0)
        if under <= 0:
            return None
        exp_ts = opt.get("expirationDate")
        exp = dt.datetime.fromtimestamp(exp_ts, tz=dt.timezone.utc).date() if exp_ts else dt.date.today()
        dte = max((exp - dt.date.today()).days, 0)
        # 0DTE: express remaining life as the fraction of a session still ahead.
        dte_frac = dte if dte > 0 else 0.4

        calls = [c for c in (_parse_contract(r, "call") for r in opt.get("calls", [])) if c]
        puts = [p for p in (_parse_contract(r, "put") for r in opt.get("puts", [])) if p]
        if not calls and not puts:
            return None

        return Chain(symbol, under, exp, dte_frac, calls, puts)

    async def nearest_expirations(self, symbol: str, count: int = 2) -> list[Chain]:
        exps = await self.expirations(symbol)
        today = dt.date.today()
        upcoming = [e for e in exps if e >= today][:count]
        chains = await self.client.gather([self.chain(symbol, e) for e in upcoming], label="chains")
        return [c for c in chains if isinstance(c, Chain)]

    async def flow_snapshot(self, symbol: str) -> dict | None:
        """Everything the options layer contributes about one underlying."""
        chains = await self.nearest_expirations(symbol, 2)
        if not chains:
            return None
        near = chains[0]
        im = near.implied_move()
        return {
            "symbol": symbol,
            "underlying": near.underlying_price,
            "expiration": near.expiration.isoformat(),
            "days_to_expiry": near.days_to_expiry,
            "implied_move_pct": round(im.implied_move_pct * 100, 2) if im else None,
            "implied_move_abs": round(im.implied_move_abs, 2) if im else None,
            "implied_vol_annual": round(im.implied_vol_annual * 100, 1) if im else None,
            "put_call_volume_ratio": round(near.put_call_volume_ratio, 2)
            if not math.isnan(near.put_call_volume_ratio) else None,
            "put_call_oi_ratio": round(near.put_call_oi_ratio, 2)
            if not math.isnan(near.put_call_oi_ratio) else None,
            "skew": near.skew_25d(),
            "walls": near.oi_walls(),
            "unusual": near.unusual_activity(),
            "liquidity_score": near.liquidity_score(),
            "chain": near,
        }
