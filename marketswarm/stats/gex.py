"""Dealer gamma exposure.

Open-interest walls tell you where contracts sit. Gamma exposure tells you what
the people who sold them have to *do*, which is the part that actually moves
price.

The mechanism: market makers are typically short calls to retail and long puts
against them. To stay delta-neutral they hedge, and the sign of that hedging
flips with aggregate gamma.

  Positive dealer gamma — hedging is mean-reverting. Dealers sell into strength
  and buy into weakness, compressing the range. Fading extremes works; breakout
  entries fail.

  Negative dealer gamma — hedging is momentum-amplifying. Dealers buy strength
  and sell weakness, stretching the range. Trends persist and reversals are
  violent. This is the regime where stops get run.

The level where aggregate gamma crosses zero (the "flip") is the boundary
between those two worlds, and is often a more meaningful intraday level than
any moving average.

Caveat carried into every report: the sign convention below assumes the
standard dealer-short-calls / dealer-long-puts positioning. That assumption is
usually right for index products and frequently wrong for single names in a
retail-call-buying frenzy. It is an inference from public open interest, not an
observation of dealer books.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .distributions import bs_price_and_greeks

CONTRACT_MULTIPLIER = 100


@dataclass
class GammaProfile:
    spot: float
    total_gex: float                      # $ per 1% move
    call_gex: float
    put_gex: float
    zero_gamma: float | None              # the flip level
    regime: str                           # positive | negative | neutral
    by_strike: dict[float, float] = field(default_factory=dict)
    largest_positive: tuple[float, float] | None = None
    largest_negative: tuple[float, float] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def playbook(self) -> str:
        if self.regime == "positive":
            return ("Dealers are long gamma and hedge against the move: expect compression, "
                    "range-bound trade and failed breakouts. Fade extremes toward the largest "
                    "gamma strike; do not chase.")
        if self.regime == "negative":
            return ("Dealers are short gamma and hedge with the move: expect expansion, trend "
                    "persistence and violent reversals. Momentum entries work, mean-reversion "
                    "is dangerous, and stops must be wider than usual.")
        return "Gamma is close to neutral — no strong dealer-hedging bias in either direction."

    def distance_to_flip_pct(self) -> float | None:
        if self.zero_gamma is None or self.spot <= 0:
            return None
        return (self.zero_gamma / self.spot - 1) * 100


def _contract_gamma(spot: float, strike: float, dte: float, iv: float, kind: str) -> float:
    if iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    g = bs_price_and_greeks(spot, strike, max(dte, 0.05), iv, kind)
    return g["gamma"]


def gamma_exposure(
    spot: float,
    calls: list,
    puts: list,
    days_to_expiry: float,
    dealer_short_calls: bool = True,
) -> GammaProfile:
    """Aggregate gamma exposure in dollars per 1% move.

    calls/puts are contract objects exposing .strike, .open_interest and
    .implied_volatility (the `providers.options.Contract` shape).

    GEX per contract = gamma x OI x 100 x spot^2 x 0.01, which converts a
    per-share second-order sensitivity into the dollar delta a dealer must
    trade for a 1% move — the quantity that actually hits the tape.
    """
    if spot <= 0:
        return GammaProfile(spot, 0, 0, 0, None, "unknown", notes=["invalid spot"])

    scale = CONTRACT_MULTIPLIER * spot * spot * 0.01
    by_strike: dict[float, float] = {}
    call_gex = put_gex = 0.0
    sign = 1.0 if dealer_short_calls else -1.0

    for c in calls:
        oi = getattr(c, "open_interest", 0) or 0
        if oi <= 0:
            continue
        g = _contract_gamma(spot, c.strike, days_to_expiry, c.implied_volatility, "call")
        # Dealers short calls -> long gamma from the customer's perspective is
        # negative for the dealer; the convention here reports dealer gamma.
        val = sign * g * oi * scale
        call_gex += val
        by_strike[c.strike] = by_strike.get(c.strike, 0.0) + val

    for p in puts:
        oi = getattr(p, "open_interest", 0) or 0
        if oi <= 0:
            continue
        g = _contract_gamma(spot, p.strike, days_to_expiry, p.implied_volatility, "put")
        val = -sign * g * oi * scale
        put_gex += val
        by_strike[p.strike] = by_strike.get(p.strike, 0.0) + val

    total = call_gex + put_gex
    zero_g = _find_zero_gamma(spot, calls, puts, days_to_expiry, sign)

    if abs(total) < 1e6:
        regime = "neutral"
    else:
        regime = "positive" if total > 0 else "negative"

    pos = max(by_strike.items(), key=lambda kv: kv[1], default=None)
    neg = min(by_strike.items(), key=lambda kv: kv[1], default=None)

    notes = [
        f"Aggregate dealer gamma {total / 1e6:+.1f}M$ per 1% move",
        "Sign convention assumes dealers are short calls and long puts — the usual index "
        "positioning, and an inference from public open interest rather than an observation.",
    ]
    if zero_g:
        notes.append(f"Zero-gamma flip near {zero_g:.2f} "
                     f"({(zero_g / spot - 1) * 100:+.2f}% from spot)")

    return GammaProfile(
        spot=spot, total_gex=total, call_gex=call_gex, put_gex=put_gex,
        zero_gamma=zero_g, regime=regime, by_strike=by_strike,
        largest_positive=pos, largest_negative=neg, notes=notes,
    )


def _find_zero_gamma(spot: float, calls: list, puts: list, dte: float,
                     sign: float, span_pct: float = 0.08, steps: int = 33) -> float | None:
    """Scan spot levels for the price at which aggregate gamma changes sign.

    Recomputing gamma at each hypothetical spot is the honest way to do this —
    gamma is not constant in spot, so interpolating the current profile finds
    the wrong level.
    """
    lo, hi = spot * (1 - span_pct), spot * (1 + span_pct)
    xs = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
    scale_base = CONTRACT_MULTIPLIER * 0.01

    values = []
    for s in xs:
        total = 0.0
        scale = scale_base * s * s
        for c in calls:
            oi = getattr(c, "open_interest", 0) or 0
            if oi > 0:
                total += sign * _contract_gamma(s, c.strike, dte, c.implied_volatility, "call") * oi * scale
        for p in puts:
            oi = getattr(p, "open_interest", 0) or 0
            if oi > 0:
                total += -sign * _contract_gamma(s, p.strike, dte, p.implied_volatility, "put") * oi * scale
        values.append(total)

    for i in range(1, len(values)):
        a, b = values[i - 1], values[i]
        if a == 0:
            return xs[i - 1]
        if (a < 0) != (b < 0):
            # Linear interpolation between the bracketing grid points.
            t = abs(a) / (abs(a) + abs(b)) if (abs(a) + abs(b)) > 0 else 0.5
            return xs[i - 1] + t * (xs[i] - xs[i - 1])
    return None


def gex_signal(profile: GammaProfile) -> dict:
    """Turn a gamma profile into something the fusion layer can consume."""
    if profile.regime == "unknown":
        return {"probability": 0.5, "weight": 0.0, "note": "gamma profile unavailable"}

    flip_dist = profile.distance_to_flip_pct()

    if profile.regime == "negative":
        note = (f"dealers short gamma ({profile.total_gex / 1e6:+.0f}M per 1%) — moves amplify, "
                f"expect range expansion")
        # Not directional, but it raises the odds of a trending session, which
        # the playbook uses to widen targets rather than to pick a side.
        return {"probability": 0.5, "weight": 0.0, "note": note, "expansion_bias": True}

    note = (f"dealers long gamma ({profile.total_gex / 1e6:+.0f}M per 1%) — hedging compresses "
            f"the range, breakouts likely to fail")
    if flip_dist is not None and abs(flip_dist) < 0.5:
        note += f"; spot is within {abs(flip_dist):.2f}% of the flip, so the regime is unstable"
    return {"probability": 0.5, "weight": 0.0, "note": note, "expansion_bias": False}
