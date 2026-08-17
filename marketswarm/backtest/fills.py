"""Execution modelling.

A backtest that fills at the mid, instantly, in unlimited size, is a
description of a market that does not exist. Most retail strategies that look
profitable on paper are paying for their edge in the spread and never see the
bill until they trade live.

This model charges for: crossing the spread, slippage that grows with size
relative to average volume, and — for options — the gap between the mid and
the price you can actually get filled at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Fill:
    intended: float
    filled: float
    slippage: float
    cost_bps: float
    note: str

    @property
    def adverse(self) -> bool:
        return abs(self.slippage) > 1e-9


@dataclass
class FillModel:
    """Costs are per-side unless stated. Defaults are deliberately pessimistic.

    spread_bps        : half-spread paid on entry and exit, in basis points
    impact_coef       : square-root market impact coefficient
    commission_per_share / _per_contract : broker fees
    option_edge_pct   : fraction of the bid-ask width surrendered on an option
                        fill. 0.35 means you get filled 35% of the way from the
                        mid toward the far side — optimistic for 0DTE, roughly
                        right for liquid weeklies.
    gap_through_stop  : stops are not guarantees. A fraction of stop exits fill
                        beyond the stop price, and this models that tail.
    """

    spread_bps: float = 2.0
    impact_coef: float = 0.35
    commission_per_share: float = 0.005
    commission_per_contract: float = 0.65
    option_edge_pct: float = 0.35
    slippage_bps_floor: float = 0.5
    gap_through_stop_pct: float = 0.25

    # ---------- equities ----------

    def fill_equity(self, price: float, shares: float, adv: float, side: str,
                    urgent: bool = False, daily_vol: float = 0.015) -> Fill:
        """side: 'buy' | 'sell'. adv: average daily volume in shares.

        Impact follows the standard square-root law:

            impact ≈ coef × σ_daily × sqrt(participation)

        The σ term is not optional. Without it the coefficient has no units and
        the model charges the same impact to a sleepy utility as to a biotech,
        while wildly overcharging small orders — trading 0.02% of ADV should
        cost a fraction of a basis point, not fifty.
        """
        if price <= 0 or shares <= 0:
            return Fill(price, price, 0.0, 0.0, "no size")

        half_spread_bps = self.spread_bps * (1.5 if urgent else 1.0)

        participation = shares / max(adv, 1.0)
        impact_bps = (self.impact_coef * max(daily_vol, 1e-4)
                      * math.sqrt(max(participation, 0.0)) * 10_000)

        total_bps = max(half_spread_bps + impact_bps, self.slippage_bps_floor)
        direction = 1 if side == "buy" else -1
        filled = price * (1 + direction * total_bps / 10_000)

        return Fill(
            intended=price,
            filled=filled,
            slippage=filled - price,
            cost_bps=total_bps,
            note=f"{total_bps:.1f} bps ({half_spread_bps:.1f} spread + {impact_bps:.1f} impact "
                 f"at {participation:.2%} of ADV)",
        )

    def stop_exit(self, stop_price: float, next_open: float | None, side: str) -> Fill:
        """Stops slip. If the session gapped through the level, the fill is at
        the open, not at the stop — the single most under-modelled cost in
        retail backtesting."""
        if next_open is None:
            return Fill(stop_price, stop_price, 0.0, 0.0, "stop filled at level")

        long_side = side == "sell"
        gapped = (next_open < stop_price) if long_side else (next_open > stop_price)
        if gapped:
            return Fill(
                intended=stop_price, filled=next_open, slippage=next_open - stop_price,
                cost_bps=abs(next_open - stop_price) / stop_price * 10_000,
                note="gapped through the stop; filled at the open",
            )
        return Fill(stop_price, stop_price, 0.0, self.spread_bps, "stop filled at level")

    # ---------- options ----------

    def fill_option(self, bid: float, ask: float, contracts: int, side: str) -> Fill:
        """Fill somewhere between the mid and the far touch."""
        if ask <= 0 or bid < 0 or ask < bid:
            return Fill(0.0, 0.0, 0.0, 0.0, "unusable quote")

        mid = (bid + ask) / 2
        width = ask - bid
        if mid <= 0:
            return Fill(0.0, 0.0, 0.0, 0.0, "worthless contract")

        direction = 1 if side == "buy" else -1
        filled = mid + direction * width * self.option_edge_pct
        filled = max(0.01, min(filled, ask) if side == "buy" else max(filled, bid))

        commission = self.commission_per_contract * max(contracts, 1)
        cost_per_contract = abs(filled - mid) * 100 + commission
        cost_bps = cost_per_contract / (mid * 100) * 10_000 if mid > 0 else 0.0

        return Fill(
            intended=mid, filled=filled, slippage=filled - mid, cost_bps=cost_bps,
            note=f"{width / mid:.1%} wide spread; paid {self.option_edge_pct:.0%} of it "
                 f"plus ${commission:.2f} commission",
        )

    def round_trip_cost_r(self, entry: float, stop: float, shares: float, adv: float,
                          daily_vol: float = 0.015) -> float:
        """Total round-trip friction expressed as a fraction of R.

        This is the number the live agent currently hard-codes as 0.06. Running
        it through the real model per-symbol is strictly better, and for a wide
        stop on a liquid name it is much cheaper than the placeholder.
        """
        risk = abs(entry - stop)
        if risk <= 0 or entry <= 0:
            return 1.0
        buy = self.fill_equity(entry, shares, adv, "buy", daily_vol=daily_vol)
        sell = self.fill_equity(entry, shares, adv, "sell", daily_vol=daily_vol)
        cash = abs(buy.slippage) + abs(sell.slippage) + 2 * self.commission_per_share
        return cash / risk

    def option_round_trip_cost_r(self, premium: float, bid: float, ask: float,
                                 target_premium: float, contracts: int = 1) -> float:
        """Round-trip option friction as a fraction of the premium risked.

        Options are where friction actually bites: a 5%-wide spread on a
        contract you hold for two hours is a far larger drag than any equity
        commission, and it is charged twice.
        """
        if premium <= 0:
            return 1.0
        buy = self.fill_option(bid, ask, contracts, "buy")
        sell = self.fill_option(bid, ask, contracts, "sell")
        cash = abs(buy.slippage) + abs(sell.slippage) + 2 * self.commission_per_contract / 100
        return cash / premium


LIQUID_PRESET = FillModel(spread_bps=1.0, impact_coef=0.20, option_edge_pct=0.25)
"""SPY/QQQ and the largest single names: penny-wide equities, tight chains."""

STANDARD_PRESET = FillModel()
"""Large-cap defaults."""

ILLIQUID_PRESET = FillModel(spread_bps=8.0, impact_coef=0.80, option_edge_pct=0.50,
                            gap_through_stop_pct=0.40)
"""Anything outside the mega-caps. If your idea needs this preset, reconsider."""
