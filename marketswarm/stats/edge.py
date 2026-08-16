"""Expected value, sizing, and the go/no-go gate.

A probability is not a trade. This module converts a calibrated probability
plus a bracket (entry/target/stop) into expected value in R-multiples, applies
realistic frictions, and refuses ideas that do not clear the bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TradeMath:
    p_win: float
    reward_risk: float
    expected_r: float
    breakeven_p: float
    kelly_fraction: float
    suggested_risk_pct: float
    edge_after_costs: float
    verdict: str

    @property
    def acceptable(self) -> bool:
        return self.expected_r > 0 and self.p_win > self.breakeven_p


def evaluate_trade(
    p_win: float,
    entry: float,
    target: float,
    stop: float,
    cost_r: float = 0.06,
    kelly_cap: float = 0.25,
    max_risk_pct: float = 1.0,
) -> TradeMath:
    """cost_r: round-trip friction (spread + slippage + commission) expressed as
    a fraction of the risked amount. 6% of R is a realistic default for liquid
    weekly options; tighten it for SPY/QQQ, widen it for anything else.
    """
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return TradeMath(p_win, 0, -1, 1, 0, 0, -1, "invalid bracket: zero risk")

    b = reward / risk
    p = max(0.0, min(1.0, p_win))
    exp_r = p * b - (1 - p) * 1.0
    exp_r_net = exp_r - cost_r
    breakeven = 1 / (1 + b)

    # Kelly for a binary payoff: f* = (p*b - q) / b
    kelly = (p * b - (1 - p)) / b if b > 0 else 0.0
    kelly = max(0.0, kelly)
    # Fractional Kelly. Full Kelly is optimal only if p is exactly right; it
    # never is, and the drawdown penalty for overestimating p is brutally
    # asymmetric. Quarter-Kelly keeps ~94% of the growth at ~44% of the variance.
    frac = min(kelly * kelly_cap, max_risk_pct / 100)

    if exp_r_net <= 0:
        verdict = "negative expectancy after costs — skip"
    elif p <= breakeven:
        verdict = f"win rate below the {breakeven:.0%} breakeven for this R:R — skip"
    elif exp_r_net < 0.10:
        verdict = "marginal edge — size down or pass"
    elif exp_r_net < 0.30:
        verdict = "acceptable edge"
    else:
        verdict = "strong edge"

    return TradeMath(
        p_win=p,
        reward_risk=b,
        expected_r=exp_r,
        breakeven_p=breakeven,
        kelly_fraction=kelly,
        suggested_risk_pct=frac * 100,
        edge_after_costs=exp_r_net,
        verdict=verdict,
    )


def kelly_position_size(
    account_value: float, risk_pct: float, entry: float, stop: float, contract_multiplier: float = 1.0
) -> dict:
    risk_dollars = account_value * risk_pct / 100
    per_unit = abs(entry - stop) * contract_multiplier
    if per_unit <= 0:
        return {"units": 0, "risk_dollars": 0.0}
    units = math.floor(risk_dollars / per_unit)
    return {
        "units": max(0, units),
        "risk_dollars": round(units * per_unit, 2),
        "risk_pct_actual": round(units * per_unit / account_value * 100, 3) if account_value else 0.0,
    }


def option_delta_equivalent(underlying_move_pct: float, delta: float, gamma: float, underlying_price: float,
                            option_price: float) -> float:
    """Approximate option P&L% from an underlying move, second order.

    dV ≈ delta*dS + 0.5*gamma*dS^2. Reported as a percentage of premium, which
    is what actually determines whether a target is reachable before theta and
    the post-open IV crush eat the position.
    """
    if option_price <= 0:
        return 0.0
    ds = underlying_price * underlying_move_pct / 100
    dv = delta * ds + 0.5 * gamma * ds * ds
    return float(dv / option_price * 100)


def theta_burn_pct(option_price: float, theta_per_day: float, hours_held: float = 3.0) -> float:
    """Fraction of premium lost to time decay over the intended hold.

    For 0DTE this is the dominant term after ~11:00 ET and the reason a
    directionally correct trade still loses.
    """
    if option_price <= 0:
        return 0.0
    return float(abs(theta_per_day) * (hours_held / 6.5) / option_price * 100)


def rank_opportunities(candidates: list[dict], top_n: int = 3) -> list[dict]:
    """Rank by a composite of expected value, confidence, and liquidity.

    Expected R dominates; confidence and liquidity are multiplicative penalties
    rather than additive bonuses, so a high-EV idea in an illiquid name cannot
    outrank a solid idea in SPY.
    """
    scored = []
    for c in candidates:
        er = float(c.get("expected_r", 0.0))
        conf = float(c.get("confidence", 50)) / 100
        liq = float(c.get("liquidity_score", 0.5))
        if er <= 0:
            continue
        score = er * (0.5 + 0.5 * conf) * (0.4 + 0.6 * liq)
        scored.append({**c, "composite_score": round(score, 4)})
    scored.sort(key=lambda x: -x["composite_score"])
    return scored[:top_n]


def portfolio_heat(open_risks_pct: list[float], correlation: float = 0.6) -> dict:
    """Total risk when positions are correlated.

    Three 1% SPY-beta longs are not 3% of independent risk; at rho=0.6 the
    effective single-shock exposure is far closer to the naive sum than
    diversification intuition suggests.
    """
    if not open_risks_pct:
        return {"naive_sum": 0.0, "effective": 0.0, "warning": None}
    n = len(open_risks_pct)
    s = sum(open_risks_pct)
    ssq = sum(r * r for r in open_risks_pct)
    var = ssq + correlation * (s * s - ssq)
    eff = math.sqrt(max(var, 0.0))
    warn = None
    if s > 3:
        warn = f"naive risk {s:.1f}% across {n} positions exceeds a 3% daily-loss guideline"
    elif eff > 2.5:
        warn = f"correlated exposure {eff:.1f}% behaves like a single concentrated bet"
    return {"naive_sum": round(s, 2), "effective": round(eff, 2), "warning": warn}
