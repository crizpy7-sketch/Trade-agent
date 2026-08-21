"""Synthesis agents: cross-verification, risk assessment, and the Day Trading
Playbook.

These run last. They consume every other agent's output, fuse the directional
signals into one calibrated probability, and turn that into concrete bracketed
ideas whose probabilities come from a barrier Monte Carlo rather than from
narrative confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..stats import bayes
from ..stats import distributions as dist
from ..stats import edge as edgemod
from .base import AgentReport, BaseAgent, SwarmContext


class CrossVerificationAgent(BaseAgent):
    name = "cross_verify"
    description = "Agreement, contradiction, and single-source risk across the swarm"
    depends_on = ("overnight_scan", "global_markets", "futures", "volatility_regime",
                  "technicals", "breaking_news", "econ_calendar", "earnings",
                  "sec_filings", "options_flow", "institutional", "sentiment")

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Cross-verification")

        usable = [r for r in ctx.reports.values() if r.usable and r.agent != self.name]
        failed = [r for r in ctx.reports.values() if not r.usable]

        all_signals: list[bayes.Signal] = []
        for r in usable:
            all_signals.extend([s for s in r.signals if ":" not in s.name])  # index-level only

        if not all_signals:
            rep.status = "degraded"
            rep.error = "no index-level directional signals produced"
            rep.headline = "Cross-verification: no directional signals to reconcile"
            rep.data = {"probability": 0.5, "signals": [], "failed_agents": [r.agent for r in failed]}
            return rep

        bullish = [s for s in all_signals if s.probability > 0.52]
        bearish = [s for s in all_signals if s.probability < 0.48]
        neutral = [s for s in all_signals if 0.48 <= s.probability <= 0.52]

        # Correlation is the crux: futures, global markets and breadth are all
        # reading the same overnight risk appetite, so they must not be counted
        # as independent confirmations of each other.
        correlation = ctx.config.signal_correlation
        fused = bayes.fuse(all_signals, prior=0.5, correlation=correlation)

        dispersion = float(np.std([s.probability for s in all_signals])) * 2
        label, score = bayes.probability_to_confidence_label(
            fused.probability, fused.effective_n, dispersion
        )

        rep.add(
            f"{len(all_signals)} index-level signals: {len(bullish)} bullish, "
            f"{len(bearish)} bearish, {len(neutral)} neutral"
        )
        rep.add(
            f"Logit-pooled posterior P(SPY closes above the open) = {fused.probability:.1%} "
            f"[{fused.interval[0]:.0%}–{fused.interval[1]:.0%}], "
            f"effective independent signals {fused.effective_n:.1f} of {len(all_signals)} raw "
            f"(assumed pairwise correlation {correlation:.2f})"
        )

        ranked = sorted(fused.contributions.items(), key=lambda kv: -abs(kv[1]))
        for nm, c in ranked[:6]:
            src = next((s for s in all_signals if s.name == nm), None)
            rep.add(f"  {nm}: {c:+.3f} log-odds — {src.note if src else ''}")

        contradictions = []
        if bullish and bearish:
            strongest_up = max(bullish, key=lambda s: s.probability * s.weight)
            strongest_down = min(bearish, key=lambda s: s.probability * s.weight)
            contradictions.append(
                f"{strongest_up.name} ({strongest_up.note}) opposes "
                f"{strongest_down.name} ({strongest_down.note})"
            )
            rep.add(f"Contradiction: {contradictions[-1]}")

        # Evidence corroboration audit
        single_source = []
        for r in usable:
            for e in r.evidence:
                if e.tags and "high" in e.tags and (e.value or {}).get("corroborations", 2) == 1:
                    single_source.append(e.claim[:90])
        if single_source:
            rep.add(f"{len(single_source)} material claims rest on a single source and are flagged unverified")

        if failed:
            rep.add(f"Degraded coverage: {', '.join(r.agent for r in failed)} produced no usable output")

        rep.data = {
            "probability": fused.probability,
            "interval": fused.interval,
            "effective_n": fused.effective_n,
            "raw_signal_count": len(all_signals),
            "contributions": fused.contributions,
            "dispersion": dispersion,
            "confidence_label": label,
            "confidence_score": score,
            "contradictions": contradictions,
            "single_source_claims": single_source,
            "failed_agents": [r.agent for r in failed],
        }
        rep.headline = (
            f"P(up session) {fused.probability:.0%} "
            f"[{fused.interval[0]:.0%}–{fused.interval[1]:.0%}], confidence {label} ({score}/100)"
        )
        rep.confidence = score / 100
        return rep


class RiskAgent(BaseAgent):
    name = "risk"
    description = "What would make today's read wrong, and how much to risk"
    depends_on = ("cross_verify", "econ_calendar", "volatility_regime", "options_flow")

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Risk assessment")

        conf = ctx.data_of("cross_verify", "confidence_score", 40)
        regime = ctx.data_of("volatility_regime", "regime", "unknown")
        very_high = ctx.data_of("econ_calendar", "very_high_impact", []) or []
        vix = ctx.data_of("volatility_regime", "vix")
        flows = ctx.data_of("options_flow", "flows", {}) or {}
        contradictions = ctx.data_of("cross_verify", "contradictions", []) or []

        risks: list[dict] = []

        if very_high:
            risks.append({
                "risk": f"Scheduled high-impact release ({', '.join(very_high)})",
                "severity": "high",
                "mitigation": "No new directional exposure in the 15 minutes either side of the print; "
                              "the first post-release impulse reverses often enough that fading it blind is "
                              "also not a strategy. Wait for the second move.",
            })
        if regime == "stress":
            risks.append({
                "risk": "Stress regime — ranges are wide and levels fail",
                "severity": "high",
                "mitigation": "Half size, wider stops, faster profit-taking. Long premium suffers "
                              "violent IV crush once the panic bid fades.",
            })
        elif regime == "choppy":
            risks.append({
                "risk": "Choppy, mean-reverting tape — breakouts fail",
                "severity": "medium",
                "mitigation": "Trade toward levels rather than through them; skip momentum continuation entries.",
            })
        if vix is not None and vix < 13:
            risks.append({
                "risk": f"VIX at {vix:.1f} — complacency and thin premium",
                "severity": "medium",
                "mitigation": "Long options are cheap but the underlying may not move enough to overcome theta; "
                              "prefer larger deltas over cheap OTM strikes.",
            })
        if contradictions:
            risks.append({
                "risk": "Signals disagree — " + contradictions[0],
                "severity": "medium",
                "mitigation": "Reduce size proportionally to the disagreement; the fused probability already "
                              "reflects it but position sizing should too.",
            })
        if conf < 40:
            risks.append({
                "risk": f"Low aggregate confidence ({conf}/100) — thin or conflicting evidence",
                "severity": "high",
                "mitigation": "Treat every idea today as a smaller-than-normal probe, or stand aside.",
            })

        spy = flows.get("SPY")
        if spy and spy.get("implied_move_pct"):
            im = spy["implied_move_pct"]
            risks.append({
                "risk": f"Option market prices a ±{im:.2f}% SPY session move",
                "severity": "info",
                "mitigation": f"Any intraday target beyond ±{im:.2f}% is betting against the option market's "
                              f"own distribution — it needs a specific catalyst, not just a trend read.",
            })

        # Sizing: start from a base risk budget, then cut for every red flag.
        base = ctx.config.base_risk_pct
        multiplier = 1.0
        if conf < 40:
            multiplier *= 0.5
        elif conf < 60:
            multiplier *= 0.75
        if regime == "stress":
            multiplier *= 0.5
        if very_high:
            multiplier *= 0.6
        if contradictions:
            multiplier *= 0.8
        suggested = round(base * multiplier, 2)

        heat = edgemod.portfolio_heat([suggested] * 3, correlation=ctx.config.signal_correlation + 0.25)

        for r in risks:
            rep.add(f"[{r['severity']}] {r['risk']} → {r['mitigation']}")
        rep.add(
            f"Suggested per-idea risk: {suggested:.2f}% of account "
            f"(base {base:.2f}% × {multiplier:.2f} for today's conditions)"
        )
        rep.add(
            f"Three concurrent correlated ideas at that size behave like a single "
            f"{heat['effective']:.2f}% bet, not {heat['naive_sum']:.2f}% of diversified risk"
        )
        if heat.get("warning"):
            rep.add(f"[high] {heat['warning']}")

        rep.data = {
            "risks": risks,
            "suggested_risk_pct": suggested,
            "size_multiplier": multiplier,
            "portfolio_heat": heat,
            "max_concurrent": ctx.config.max_concurrent_ideas,
        }
        rep.headline = (
            f"{len([r for r in risks if r['severity'] == 'high'])} high-severity risks; "
            f"per-idea risk budget {suggested:.2f}%"
        )
        rep.confidence = 0.8
        return rep


@dataclass
class Idea:
    kind: str                  # call | put | stock
    symbol: str
    direction: str
    entry: float
    target: float
    stop: float
    probability: float
    raw_probability: float
    expected_r: float
    confidence: int
    rationale: str
    invalidation: str
    strike: float | None = None
    expiration: str | None = None
    option_entry: str | None = None
    option_target: str | None = None
    option_stop: str | None = None
    option_note: str | None = None
    liquidity: float = 0.5
    evidence: list[str] = None
    math_note: str = ""
    ev_verdict: str = ""
    clears_bar: bool = False

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["evidence"] = self.evidence or []
        return d


class PlaybookAgent(BaseAgent):
    name = "playbook"
    description = "Concrete bracketed ideas with Monte-Carlo probabilities"
    depends_on = ("cross_verify", "risk", "technicals", "options_flow", "earnings",
                  "sec_filings", "overnight_scan")
    timeout = 90.0

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Day Trading Playbook")

        p_up = ctx.data_of("cross_verify", "probability", 0.5)
        conf_score = ctx.data_of("cross_verify", "confidence_score", 40)
        setups = ctx.data_of("technicals", "setups", {}) or {}
        flows = ctx.data_of("options_flow", "flows", {}) or {}
        profiles = ctx.data_of("overnight_scan", "profiles", {}) or {}
        earnings_reacting = {a["ticker"]: a for a in (ctx.data_of("earnings", "reacting", []) or [])}
        filings_high = set(ctx.data_of("sec_filings", "high_impact", []) or [])
        community_reads = ctx.data_of("sentiment", "symbol_reads", {}) or {}
        regime = ctx.data_of("volatility_regime", "regime", "unknown")
        calibrator = ctx.config.calibrator

        if not setups:
            rep.status = "failed"
            rep.error = "no technical setups available — cannot construct ideas"
            rep.headline = "Playbook: insufficient data to construct ideas"
            return rep

        call_ideas: list[Idea] = []
        put_ideas: list[Idea] = []
        stock_ideas: list[Idea] = []

        for sym, s in setups.items():
            price = s.get("price")
            sigma = s.get("sigma_daily")
            if not price or not sigma or math.isnan(sigma):
                continue

            atr = s.get("atr") or price * sigma
            flow = flows.get(sym)
            liquidity = flow.get("liquidity_score", 0.4) if flow else 0.35
            gap = (profiles.get(sym) or {}).get("gap_pct")

            # Per-symbol evidence assembled into a symbol-level probability.
            sym_signals = [bayes.Signal("index_fusion", p_up, weight=0.9, note="swarm index read")]

            if s["trend"] == "up":
                sym_signals.append(bayes.Signal(
                    "symbol_trend", 0.5 + 0.09 * s["trend_strength"], weight=0.8,
                    note=f"{sym} daily uptrend, 9EMA {s['ema9']:.2f} > 21EMA {s['ema21']:.2f}"))
            elif s["trend"] == "down":
                sym_signals.append(bayes.Signal(
                    "symbol_trend", 0.5 - 0.09 * s["trend_strength"], weight=0.8,
                    note=f"{sym} daily downtrend, 9EMA {s['ema9']:.2f} < 21EMA {s['ema21']:.2f}"))

            if gap is not None and abs(gap) > 0.5:
                had_e = sym in earnings_reacting
                from ..providers.earnings import gap_persistence_prior
                prior = gap_persistence_prior(gap, had_e)
                p_gap = prior if gap > 0 else 1 - prior
                sym_signals.append(bayes.Signal(
                    "gap_persistence", p_gap, weight=0.7,
                    note=f"{gap:+.2f}% gap{' on earnings' if had_e else ''}, "
                         f"base rate of continuation {prior:.0%}"))

            if sym in earnings_reacting:
                a = earnings_reacting[sym]
                bullish = "bullish" in a["bias"]
                sym_signals.append(bayes.Signal(
                    "earnings_reaction", 0.5 + (0.10 if bullish else -0.10),
                    weight=0.9 * a["confidence"] * 2, note=a["read"]))

            if sym in filings_high:
                sym_signals.append(bayes.Signal(
                    "filing_catalyst", 0.5, weight=0.3,
                    note="high-impact 8-K overnight — direction unclear but range will widen"))

            rsi = s.get("rsi")
            if rsi and rsi > 75:
                sym_signals.append(bayes.Signal("stretched", 0.45, weight=0.4, note=f"RSI {rsi:.0f}"))
            elif rsi and rsi < 25:
                sym_signals.append(bayes.Signal("washed_out", 0.55, weight=0.4, note=f"RSI {rsi:.0f}"))

            social = community_reads.get(sym) or {}
            if social.get("qualifies"):
                sym_signals.append(bayes.Signal(
                    "community_consensus",
                    float(social.get("probability_up", 0.5)),
                    # Social evidence is gameable and reflexive. Even after the
                    # independent-author rule it is capped below every measured
                    # market input and can only nudge a symbol read.
                    weight=0.25,
                    note=(f"permissioned community: {social.get('long_sources', 0)} long / "
                          f"{social.get('short_sources', 0)} short across "
                          f"{social.get('independent_sources', 0)} independent sources"),
                ))

            fused = bayes.fuse(sym_signals, prior=0.5, correlation=ctx.config.signal_correlation)
            raw_p = fused.probability
            p_sym = calibrator.transform(raw_p) if calibrator else raw_p

            implied_pct = (flow or {}).get("implied_move_pct")
            # The implied move is the ceiling on any target: beyond it, the idea
            # is betting against the option market's own distribution.
            max_reach = price * (implied_pct / 100) if implied_pct else atr * 0.8

            _, sym_conf = bayes.probability_to_confidence_label(
                p_sym, fused.effective_n, float(np.std([x.probability for x in sym_signals])) * 2
            )
            sym_conf = round(sym_conf * 0.6 + conf_score * 0.4)
            primary_direction = "long" if p_sym >= 0.5 else "short"

            # Screen both sides of every liquid chain. This creates three call
            # and three put *candidate slots* without manufacturing an edge:
            # the counter-thesis normally has worse probability/EV and is
            # labelled WATCH ONLY or REJECTED downstream. Only the primary
            # direction is eligible for the stock-setup list.
            for direction in ("long", "short"):
                long_side = direction == "long"
                bracket = _build_bracket(
                    price=price,
                    atr=atr,
                    max_reach=max_reach,
                    long_side=long_side,
                    support=(s.get("support") or {}).get("price"),
                    resistance=(s.get("resistance") or {}).get("price"),
                )
                if bracket is None:
                    continue                 # no sane structure for this side
                entry, target, stop = bracket

                # Both sides see the same underlying drift; the barrier geometry
                # determines whether the call or put reaches its target first.
                drift = (p_sym - 0.5) * sigma * 2
                barrier = dist.barrier_probabilities(
                    entry=entry, target=target, stop=stop, sigma_daily=sigma,
                    horizon_days=1.0, drift_daily=drift, n_paths=8000,
                    seed=0,   # deterministic: identical inputs produce identical reports
                )
                trade = edgemod.evaluate_bracket(
                    barrier, entry, target, stop,
                    cost_r=(ctx.config.friction_r if sym in ("SPY", "QQQ")
                            else ctx.config.friction_r * 1.6),
                )

                math_note = (
                    f"Monte Carlo (Student-t, df=4, {8000:,} paths, σ={sigma*100:.2f}%/day): "
                    f"P(target first) {barrier.p_target_first:.0%}, P(stop first) {barrier.p_stop_first:.0%}, "
                    f"P(neither barrier) {barrier.p_neither:.0%} — of those, {barrier.p_neither_positive:.0%} "
                    f"close green, averaging {barrier.mean_r_neither:+.2f}R. "
                    f"Overall P(profit) {trade.p_win:.0%}. R:R {trade.reward_risk:.2f}, "
                    f"breakeven {trade.breakeven_p:.0%}, expected {trade.expected_r:+.2f}R "
                    f"({trade.edge_after_costs:+.2f}R after costs). "
                    f"Quarter-Kelly sizing {trade.suggested_risk_pct:.2f}% of account. {trade.verdict}."
                )

                evidence = [sig.note for sig in sym_signals if sig.note]
                if direction != primary_direction:
                    evidence.append(
                        f"Counter-thesis screen: the fused directional read favours "
                        f"{primary_direction}; this side must earn its place on path probability "
                        f"and expected value, not symmetry."
                    )
                invalidation = _invalidation_text(sym, direction, entry, stop, s, regime, gap)
                rationale = _rationale_text(
                    sym, direction, s, gap, p_sym, barrier, earnings_reacting.get(sym)
                )

                base = {
                    "symbol": sym, "direction": direction, "entry": round(entry, 2),
                    "target": round(target, 2), "stop": round(stop, 2),
                    "probability": round(barrier.p_target_first, 4),
                    "raw_probability": round(raw_p if long_side else 1 - raw_p, 4),
                    "expected_r": round(trade.edge_after_costs, 3),
                    "confidence": (sym_conf if direction == primary_direction
                                   else max(0, sym_conf - 10)),
                    "rationale": rationale, "invalidation": invalidation,
                    "liquidity": liquidity, "evidence": evidence, "math_note": math_note,
                    "ev_verdict": trade.verdict,
                    "clears_bar": trade.edge_after_costs > ctx.config.min_expected_r,
                }

                if direction == primary_direction:
                    stock_ideas.append(Idea(kind="stock", **base))
                option = _build_option_leg(flow, price, direction, target, stop, sigma)
                if option:
                    idea = Idea(
                        kind="call" if long_side else "put",
                        **{**base, **option},
                    )
                    (call_ideas if long_side else put_ideas).append(idea)

        def rank(ideas: list[Idea], n: int) -> list[Idea]:
            """Positive-expectancy ideas first, then the least-bad remainder.

            Every candidate is surfaced with its verdict rather than silently
            dropped: on a day when nothing clears the bar, "here are the three
            closest and none of them are worth the cost" is the useful answer,
            and hiding them would just invite trading on a hunch instead.
            """
            tradeable = edgemod.rank_opportunities(
                [{"expected_r": i.expected_r, "confidence": i.confidence,
                  "liquidity_score": i.liquidity, "_idea": i} for i in ideas], top_n=n
            )
            out = [t["_idea"] for t in tradeable]
            if len(out) < n:
                rest = sorted(
                    (i for i in ideas if i not in out),
                    key=lambda i: (-i.expected_r, -i.confidence),
                )
                out.extend(rest[: n - len(out)])
            return out

        top_calls = rank(call_ideas, 3)
        top_puts = rank(put_ideas, 3)
        top_stocks = rank(stock_ideas, 5)

        cleared = sum(1 for i in stock_ideas if i.clears_bar)
        rep.data = {
            "calls": [i.to_dict() for i in top_calls],
            "puts": [i.to_dict() for i in top_puts],
            "stocks": [i.to_dict() for i in top_stocks],
            "index_probability": p_up,
            "regime": regime,
            "candidates_evaluated": len(setups),
            "candidates_bracketed": len(stock_ideas),
            "candidates_clearing_bar": cleared,
        }
        rep.headline = (
            f"{len(top_calls)} call ideas, {len(top_puts)} put ideas, {len(top_stocks)} stock setups "
            f"from {len(setups)} candidates — {cleared} clear the positive-expectancy bar after costs"
        )
        rep.confidence = conf_score / 100

        for i in top_calls + top_puts + top_stocks:
            flag = "" if i.clears_bar else "  [does not clear the cost bar]"
            rep.add(f"{i.kind.upper()} {i.symbol} {i.direction}: entry {i.entry}, target {i.target}, "
                    f"stop {i.stop}, P {i.probability:.0%}, EV {i.expected_r:+.2f}R{flag}")
        if not cleared:
            rep.add(
                "No candidate clears expected value after costs today. The ideas below are ranked "
                "by how close they came, not endorsed — on a day like this, standing aside is the "
                "position with the highest expected value."
            )
        return rep


MIN_REWARD_RISK = 1.15


def _build_bracket(
    price: float,
    atr: float,
    max_reach: float,
    long_side: bool,
    support: float | None,
    resistance: float | None,
    min_risk_atr: float = 0.35,
    max_risk_atr: float = 1.10,
    buffer_atr: float = 0.15,
) -> tuple[float, float, float] | None:
    """Place entry, stop and target off structure and volatility.

    Three rules, each learned the expensive way by every discretionary trader:
      1. The stop goes *beyond* the level, not on it. A stop resting inside
         support is a donation to whoever is defending that level.
      2. The stop is never tighter than a fraction of ATR, or normal noise
         takes it out before the thesis has a chance to be right or wrong.
      3. The target is capped by the option market's implied move and must
         still clear a minimum reward:risk. When structure leaves no room
         between the entry and the next level, there is no trade — returning
         None here is a feature.
    """
    if atr <= 0 or price <= 0:
        return None

    buffer = buffer_atr * atr
    min_risk = min_risk_atr * atr
    max_risk = max(min_risk, max_risk_atr * atr)

    if long_side:
        struct_stop = (support - buffer) if (support and support < price) else (price - 0.6 * atr)
        stop = min(struct_stop, price - min_risk)
        stop = max(stop, price - max_risk)
        risk = price - stop
        if risk <= 0:
            return None

        ceiling = price + max_reach
        struct_target = (resistance - buffer) if (resistance and resistance > price) else None
        target = min(struct_target, ceiling) if struct_target else ceiling
        if target < price + MIN_REWARD_RISK * risk:
            # Structure is too close; reach for the ceiling instead.
            target = min(price + 1.6 * risk, ceiling)
        if target <= price + MIN_REWARD_RISK * risk:
            return None
    else:
        struct_stop = (resistance + buffer) if (resistance and resistance > price) else (price + 0.6 * atr)
        stop = max(struct_stop, price + min_risk)
        stop = min(stop, price + max_risk)
        risk = stop - price
        if risk <= 0:
            return None

        floor_ = price - max_reach
        struct_target = (support + buffer) if (support and support < price) else None
        target = max(struct_target, floor_) if struct_target else floor_
        if target > price - MIN_REWARD_RISK * risk:
            target = max(price - 1.6 * risk, floor_)
        if target >= price - MIN_REWARD_RISK * risk:
            return None

    return price, target, stop


def _build_option_leg(flow, price, direction, target, stop, sigma) -> dict | None:
    """Pick a strike and reprice it at the underlying's target and stop.

    A ~40-delta strike is the deliberate default: enough delta that a normal
    intraday move actually pays, without the pure-lottery decay profile of a
    far-OTM 0DTE contract.
    """
    if not flow or not flow.get("chain"):
        return None
    chain = flow["chain"]
    kind = "call" if direction == "long" else "put"

    contract = chain.by_delta(0.40, kind)
    if not contract or contract.mid <= 0:
        return None

    entry_greeks = chain.greeks_for(contract)
    prem = contract.mid
    iv = contract.implied_volatility
    dte = chain.days_to_expiry

    # Reprice with Black-Scholes at the bracket levels, holding IV fixed and
    # advancing the clock: a target reached mid-session has burned roughly half
    # the remaining life, which is exactly the cost the linear approximation
    # misses on short-dated contracts.
    remaining = max(dte * 0.5, 0.05)
    prem_target = dist.bs_price_and_greeks(target, contract.strike, remaining, iv, kind)["price"]
    prem_stop = dist.bs_price_and_greeks(stop, contract.strike, remaining, iv, kind)["price"]
    prem_target = max(prem_target, 0.01)
    prem_stop = max(prem_stop, 0.01)

    theta_pct = edgemod.theta_burn_pct(prem, entry_greeks["theta"], hours_held=3.0)

    return {
        "strike": contract.strike,
        "expiration": flow.get("expiration"),
        "option_entry": f"${prem * 0.97:.2f}–${prem * 1.05:.2f} (mid ${prem:.2f}, "
                        f"{abs(entry_greeks['delta']):.2f} delta, IV {iv * 100:.0f}%)",
        "option_target": f"${prem_target:.2f} ({(prem_target / prem - 1) * 100:+.0f}% on premium)",
        "option_stop": f"${prem_stop:.2f} ({(prem_stop / prem - 1) * 100:+.0f}% on premium)",
        "option_note": (
            f"Repriced with Black-Scholes at IV {iv * 100:.0f}% with half the contract's remaining "
            f"life burned. Theta alone costs about {theta_pct:.0f}% of premium over a three-hour "
            f"hold, so the move has to arrive early — being right late is still a loss."
        ),
    }


def _rationale_text(sym, direction, s, gap, p, barrier, earnings) -> str:
    parts = []
    if earnings:
        parts.append(earnings["read"])
    if gap is not None and abs(gap) > 0.3:
        parts.append(f"{sym} is gapping {gap:+.2f}% into the open")
    parts.append(
        f"daily structure is {s['trend']} with price {s['price']:.2f} versus 9EMA {s['ema9']:.2f} "
        f"and 21EMA {s['ema21']:.2f}, RSI {s['rsi']:.0f}"
    )
    if s.get("support"):
        parts.append(f"nearest support {s['support']['price']:.2f}")
    if s.get("resistance"):
        parts.append(f"nearest resistance {s['resistance']['price']:.2f}")
    parts.append(
        f"the fused evidence puts P({'up' if direction == 'long' else 'down'}) at {p if direction == 'long' else 1-p:.0%} "
        f"and the path simulation puts P(target before stop) at {barrier.p_target_first:.0%}"
    )
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + "."


def _invalidation_text(sym, direction, entry, stop, s, regime, gap) -> str:
    conds = [f"{sym} trading through {stop:.2f} on a 5-minute close (the structural stop)"]
    if direction == "long":
        if s.get("support"):
            conds.append(f"loss of {s['support']['price']:.2f} support")
        conds.append("failure to hold above the pre-market high in the first 30 minutes")
        if gap is not None and gap > 0:
            conds.append(f"a full fill of the {gap:+.2f}% gap back to {s['prev_close']:.2f}")
    else:
        if s.get("resistance"):
            conds.append(f"reclaim of {s['resistance']['price']:.2f} resistance")
        conds.append("failure to break the pre-market low in the first 30 minutes")
        if gap is not None and gap < 0:
            conds.append(f"a full fill of the {gap:+.2f}% gap back to {s['prev_close']:.2f}")
    conds.append("VIX spiking more than 10% intraday (regime change invalidates the setup's base rates)")
    if regime == "choppy":
        conds.append("two consecutive failed breaks of the entry level — the chop thesis wins")
    return "; ".join(conds)
