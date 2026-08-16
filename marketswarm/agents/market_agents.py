"""Price-and-structure agents: overnight tape, global session, futures,
volatility regime, and technical levels."""

from __future__ import annotations

import math

import numpy as np

from ..stats import distributions as dist
from ..stats import regime as rg
from ..stats import technicals as ta
from .base import AgentReport, BaseAgent, SwarmContext


class OvernightScanAgent(BaseAgent):
    name = "overnight_scan"
    description = "Overnight gaps and pre-market participation across the watchlist"

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Overnight scan")
        profiles = await ctx.market.client.gather(
            [ctx.market.premarket_profile(s) for s in ctx.universe], label="premarket"
        )
        good = [p for p in profiles if p and p.get("gap_pct") is not None]
        if not good:
            rep.status = "failed"
            rep.error = "no pre-market data returned"
            rep.headline = "Overnight scan: no pre-market data available"
            return rep

        movers = sorted(good, key=lambda p: -abs(p["gap_pct"]))
        gappers = [p for p in movers if abs(p["gap_pct"]) >= 1.0]

        for p in movers[:8]:
            vol_note = ""
            if p.get("premarket_volume"):
                vol_note = f", {p['premarket_volume']:,.0f} pre-market shares"
            rep.add(
                f"{p['symbol']}: {p['gap_pct']:+.2f}% from {p['previous_close']:.2f} "
                f"to {p['last']:.2f}{vol_note}"
            )
            rep.cite(
                f"{p['symbol']} gapping {p['gap_pct']:+.2f}% pre-market",
                source="exchange_data",
                url=f"https://finance.yahoo.com/quote/{p['symbol']}",
                reliability=0.93,
                value=p,
                tags=["premarket", p["symbol"]],
            )

        ups = [p for p in good if p["gap_pct"] > 0]
        tilt = len(ups) / len(good)
        rep.data = {
            "profiles": {p["symbol"]: p for p in good},
            "gappers": gappers,
            "breadth_pct_up": tilt,
            "biggest_mover": movers[0]["symbol"] if movers else None,
        }
        rep.headline = (
            f"{len(gappers)} of {len(good)} watchlist names gapping ≥1%; "
            f"{tilt:.0%} of the list is green pre-market"
        )
        rep.confidence = 0.8 if len(good) >= len(ctx.universe) * 0.7 else 0.55
        if len(good) < len(ctx.universe) * 0.7:
            rep.status = "degraded"

        # Pre-market breadth is a weak-but-real read on the open.
        if len(good) >= 6:
            p_up = 0.5 + (tilt - 0.5) * 0.30
            rep.signal(
                "premarket_breadth",
                p_up,
                weight=0.7 * ctx.weight_for(self.name),
                note=f"{tilt:.0%} of watchlist green pre-market",
            )
        return rep


class GlobalMarketsAgent(BaseAgent):
    name = "global_markets"
    description = "Asian and European session outcomes"

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Global markets")
        board = await ctx.market.global_board()
        if not board:
            rep.status = "failed"
            rep.error = "no global index data"
            rep.headline = "Global markets: data unavailable"
            return rep

        asia = [q for s, q in board.items() if s in ("^N225", "^HSI", "000001.SS", "^KS11", "^AXJO")]
        europe = [q for s, q in board.items() if s in ("^STOXX50E", "^GDAXI", "^FTSE", "^FCHI")]

        for q in board.values():
            rep.add(f"{q.name}: {q.change_pct:+.2f}%")
            rep.cite(q.to_evidence().claim, source="exchange_data",
                     url=f"https://finance.yahoo.com/quote/{q.symbol}", reliability=0.93,
                     tags=["global"])

        asia_avg = float(np.mean([q.change_pct for q in asia])) if asia else 0.0
        eu_avg = float(np.mean([q.change_pct for q in europe])) if europe else 0.0

        rep.data = {
            "asia_avg_pct": asia_avg,
            "europe_avg_pct": eu_avg,
            "asia": {q.symbol: q.change_pct for q in asia},
            "europe": {q.symbol: q.change_pct for q in europe},
            "aligned": (asia_avg > 0) == (eu_avg > 0),
        }
        rep.headline = f"Asia {asia_avg:+.2f}% average, Europe {eu_avg:+.2f}% average"
        rep.confidence = 0.75

        # Europe is still trading into the U.S. open, so it carries more
        # information about the open than the already-closed Asian session.
        blended = 0.35 * asia_avg + 0.65 * eu_avg
        if abs(blended) > 0.15:
            p = 0.5 + max(-0.12, min(0.12, blended * 0.05))
            rep.signal("global_session_tone", p,
                       weight=0.8 * ctx.weight_for(self.name),
                       note=f"Asia {asia_avg:+.2f}%, Europe {eu_avg:+.2f}%")
        if not rep.data["aligned"]:
            rep.add("Asia and Europe diverged overnight — a weaker directional read than usual")
            rep.confidence = 0.55
        return rep


class FuturesAgent(BaseAgent):
    name = "futures"
    description = "U.S. index futures, rates, dollar and commodities"

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Index futures")
        board = await ctx.market.futures_board()
        rates = await ctx.market.rates_fx_board()
        if not board:
            rep.status = "failed"
            rep.error = "no futures data"
            rep.headline = "Futures: data unavailable"
            return rep

        for q in board.values():
            rep.add(f"{q.name}: {q.change_pct:+.2f}% ({q.price:,.2f})")
            rep.cite(q.to_evidence().claim, source="exchange_data",
                     url=f"https://finance.yahoo.com/quote/{q.symbol}", reliability=0.93,
                     tags=["futures"])
        for q in rates.values():
            rep.add(f"{q.name}: {q.price:,.3f} ({q.change_pct:+.2f}%)")

        es = board.get("ES=F")
        nq = board.get("NQ=F")
        rty = board.get("RTY=F")
        equity = [q.change_pct for q in (es, nq, rty) if q]
        avg = float(np.mean(equity)) if equity else 0.0

        rotation = None
        if es and nq:
            spread = nq.change_pct - es.change_pct
            if abs(spread) > 0.25:
                rotation = "growth/tech leading" if spread > 0 else "value/defensive leading"
                rep.add(f"NQ minus ES spread {spread:+.2f}pp — {rotation}")

        rep.data = {
            "es_pct": es.change_pct if es else None,
            "nq_pct": nq.change_pct if nq else None,
            "rty_pct": rty.change_pct if rty else None,
            "avg_equity_pct": avg,
            "rotation": rotation,
            "tnx": rates.get("^TNX").price if rates.get("^TNX") else None,
            "dollar_pct": board["DX=F"].change_pct if "DX=F" in board else None,
            "crude_pct": board["CL=F"].change_pct if "CL=F" in board else None,
            "gold_pct": board["GC=F"].change_pct if "GC=F" in board else None,
            "board": {q.symbol: {"name": q.name, "pct": q.change_pct, "price": q.price}
                      for q in board.values()},
        }
        rep.headline = (
            f"ES {es.change_pct:+.2f}%, NQ {nq.change_pct:+.2f}%" if es and nq
            else f"Equity futures {avg:+.2f}% average"
        )
        rep.confidence = 0.85

        # Futures are the single most direct pre-open read on the cash open,
        # but their information about the *close* is far weaker than their
        # information about the open — hence a modest slope.
        if abs(avg) > 0.05:
            p = 0.5 + max(-0.15, min(0.15, avg * 0.10))
            rep.signal("futures_direction", p,
                       weight=1.1 * ctx.weight_for(self.name),
                       note=f"equity futures {avg:+.2f}% pre-open")

        tnx = rates.get("^TNX")
        if tnx and abs(tnx.change_pct) > 1.5:
            # Sharp yield moves pressure long-duration equity risk.
            p = 0.5 - max(-0.08, min(0.08, tnx.change_pct * 0.01))
            rep.signal("rates_pressure", p, weight=0.6 * ctx.weight_for(self.name),
                       note=f"10Y yield {tnx.change_pct:+.2f}% overnight")
        return rep


class VolatilityRegimeAgent(BaseAgent):
    name = "volatility_regime"
    description = "VIX complex, term structure, and the prevailing regime"

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Volatility regime")
        vol = await ctx.market.volatility_board()
        spy_hist = await ctx.market.history("SPY", range_="1y", interval="1d")

        vix = vol.get("^VIX").price if vol.get("^VIX") else None
        vix3m = vol.get("^VIX3M").price if vol.get("^VIX3M") else None
        vix_chg = vol.get("^VIX").change_pct if vol.get("^VIX") else None

        if spy_hist is None:
            rep.status = "degraded"
            rep.error = "SPY history unavailable — regime inferred from VIX only"
            regime = rg.Regime("unknown", float("nan"), 0, 0, 0,
                               "insufficient price history", ["no SPY history"])
        else:
            regime = rg.classify_regime(spy_hist.closes, vix)

        term = rg.vix_term_structure_signal(vix, vix3m)
        garch = rg.garch_forecast(spy_hist.returns()) if spy_hist else {}

        if vix is not None:
            rep.add(f"VIX {vix:.2f} ({vix_chg:+.1f}% overnight)")
            rep.cite(f"VIX at {vix:.2f}", source="cboe",
                     url="https://finance.yahoo.com/quote/%5EVIX", reliability=0.92, tags=["vix"])
        rep.add(f"Regime: {regime.label} — {regime.playbook}")
        for n in regime.notes:
            rep.add(n)
        if term.get("state") != "unknown":
            rep.add(f"Term structure: {term['state']} (VIX/VIX3M {term['ratio']}) — {term['read']}")

        sigma_daily = garch.get("sigma_daily")
        if sigma_daily and not math.isnan(sigma_daily):
            rep.add(
                f"GARCH(1,1) one-day vol forecast {sigma_daily*100:.2f}% "
                f"({garch.get('sigma_annual', 0)*100:.0f}% annualised), "
                f"{'above' if garch.get('mean_reverting') else 'below'} the long-run level"
            )

        rep.data = {
            "vix": vix,
            "vix3m": vix3m,
            "vix_change_pct": vix_chg,
            "regime": regime.label,
            "regime_playbook": regime.playbook,
            "term_structure": term,
            "garch": garch,
            "sigma_daily": sigma_daily,
            "spy_history": spy_hist,
        }
        rep.headline = (
            f"{regime.label.replace('_', ' ')} regime"
            + (f", VIX {vix:.1f} ({term.get('state', 'n/a')})" if vix else "")
        )
        rep.confidence = 0.8 if vix is not None and spy_hist is not None else 0.5

        # A VIX spike overnight is a genuine risk-off tell for the open.
        if vix_chg is not None and abs(vix_chg) > 4:
            p = 0.5 - max(-0.12, min(0.12, vix_chg * 0.008))
            rep.signal("vix_impulse", p, weight=0.8 * ctx.weight_for(self.name),
                       note=f"VIX {vix_chg:+.1f}% overnight")
        if term.get("state") == "backwardation":
            rep.signal("term_structure_stress", 0.44, weight=0.6 * ctx.weight_for(self.name),
                       note="VIX above VIX3M — near-term risk bid")
        return rep


class TechnicalAgent(BaseAgent):
    name = "technicals"
    description = "Trend, levels, and volatility bands on the index and leaders"
    depends_on = ("overnight_scan", "volatility_regime")

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Technical structure")
        symbols = ctx.index_symbols + [s for s in ctx.universe if s not in ctx.index_symbols][:8]
        histories = await ctx.market.client.gather(
            [ctx.market.history(s, range_="6mo", interval="1d") for s in symbols], label="history"
        )
        profiles = ctx.data_of("overnight_scan", "profiles", {}) or {}

        setups: dict[str, dict] = {}
        for sym, h in zip(symbols, histories):
            if h is None or len(h) < 30:
                continue
            price = float(profiles.get(sym, {}).get("last") or h.last)
            trend = ta.trend_state(h.closes)
            levels = ta.cluster_levels(h.highs, h.lows, h.closes, price)
            support, resistance = ta.nearest_levels(levels, price)
            atr14 = ta.atr(h.highs, h.lows, h.closes)
            rsi14 = ta.rsi(h.closes)
            bb = ta.bollinger(h.closes)
            pivots = ta.pivot_levels(float(h.highs[-1]), float(h.lows[-1]), float(h.closes[-1]))
            rvol = ta.relative_volume(float(h.volumes[-1]), h.volumes[-20:])
            squeeze = ta.keltner_squeeze(h.highs, h.lows, h.closes)
            sigma_d = dist.realized_volatility(h.closes, 20)

            setups[sym] = {
                "price": price,
                "trend": trend.direction,
                "trend_strength": round(trend.strength, 2),
                "ema9": trend.ema9,
                "ema21": trend.ema21,
                "ema50": trend.ema50,
                "atr": atr14,
                "atr_pct": atr14 / price * 100 if price else None,
                "rsi": rsi14,
                "bollinger": bb,
                "pivots": pivots,
                "support": {"price": support.price, "strength": round(support.strength, 2)} if support else None,
                "resistance": {"price": resistance.price, "strength": round(resistance.strength, 2)} if resistance else None,
                "rvol": rvol,
                "squeeze": squeeze,
                "sigma_daily": sigma_d,
                "prev_high": float(h.highs[-1]),
                "prev_low": float(h.lows[-1]),
                "prev_close": float(h.closes[-1]),
            }

            if sym in ctx.index_symbols:
                bits = [f"{sym} {trend.direction} trend (strength {trend.strength:.2f})",
                        f"RSI {rsi14:.0f}", f"ATR {atr14:.2f} ({atr14/price*100:.2f}%)"]
                if support:
                    bits.append(f"support {support.price:.2f}")
                if resistance:
                    bits.append(f"resistance {resistance.price:.2f}")
                if squeeze:
                    bits.append("volatility squeeze — expansion likely")
                rep.add(" | ".join(bits))
                rep.cite(
                    f"{sym} {trend.direction} trend, RSI {rsi14:.0f}, ATR {atr14:.2f}",
                    source="exchange_data", reliability=0.90,
                    url=f"https://finance.yahoo.com/quote/{sym}",
                    value=setups[sym], tags=["technicals", sym],
                )

        if not setups:
            rep.status = "failed"
            rep.error = "no usable price history"
            rep.headline = "Technicals: price history unavailable"
            return rep

        rep.data = {"setups": setups}
        spy = setups.get("SPY")
        if spy:
            rep.headline = (
                f"SPY {spy['trend']} trend, RSI {spy['rsi']:.0f}, "
                f"ATR {spy['atr_pct']:.2f}% of price"
            )
            # Trend and RSI both inform the session, but each is weak alone.
            p_trend = 0.5 + (0.10 * spy["trend_strength"] * (1 if spy["trend"] == "up" else -1 if spy["trend"] == "down" else 0))
            rep.signal("index_trend", p_trend, weight=0.9 * ctx.weight_for(self.name),
                       note=f"SPY daily trend {spy['trend']} (strength {spy['trend_strength']})")
            if spy["rsi"] > 72:
                rep.signal("index_overbought", 0.45, weight=0.5 * ctx.weight_for(self.name),
                           note=f"SPY RSI {spy['rsi']:.0f} — stretched")
            elif spy["rsi"] < 28:
                rep.signal("index_oversold", 0.56, weight=0.5 * ctx.weight_for(self.name),
                           note=f"SPY RSI {spy['rsi']:.0f} — washed out")
        else:
            rep.headline = f"Technical structure mapped for {len(setups)} symbols"
        rep.confidence = 0.75
        return rep
