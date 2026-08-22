"""Positioning agents: options flow, institutional footprints, and sentiment."""

from __future__ import annotations

import math

from ..providers.community import CommunityData, aggregate_symbol_reads
from .base import AgentReport, BaseAgent, SwarmContext


class OptionsFlowAgent(BaseAgent):
    name = "options_flow"
    description = "Implied moves, skew, open-interest walls and unusual activity"

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Options flow")
        targets = ctx.index_symbols + [s for s in ctx.universe if s not in ctx.index_symbols][:6]
        snaps = await ctx.options.client.gather(
            [ctx.options.flow_snapshot(s) for s in targets], label="flow"
        )
        flows = {s["symbol"]: s for s in snaps if s}

        if not flows:
            rep.status = "failed"
            rep.error = "no option chain data"
            rep.headline = "Options flow: chains unavailable"
            return rep

        for sym, f in flows.items():
            bits = [f"{sym} implied move ±{f['implied_move_pct']:.2f}%" if f.get("implied_move_pct") else f"{sym}"]
            if f.get("put_call_volume_ratio") is not None:
                bits.append(f"P/C volume {f['put_call_volume_ratio']:.2f}")
            if f.get("skew", {}).get("skew") is not None:
                bits.append(f"25d skew {f['skew']['skew']:+.1f} — {f['skew']['read']}")
            rep.add(" | ".join(bits))

            walls = f.get("walls", {})
            if walls.get("call_walls"):
                cw = ", ".join(f"{w['strike']:g} ({w['oi']:,} OI)" for w in walls["call_walls"][:2])
                pw = ", ".join(f"{w['strike']:g} ({w['oi']:,} OI)" for w in walls.get("put_walls", [])[:2])
                rep.add(f"{sym} call walls {cw}; put walls {pw}; max-pain proxy {walls.get('max_pain_proxy')}")

            for u in f.get("unusual", [])[:3]:
                rep.add(
                    f"{sym} unusual: {u['volume']:,} {u['kind']}s at {u['strike']:g} "
                    f"vs {u['open_interest']:,} OI (turnover {u['turnover']}x, ~${u['notional_estimate']:,} notional)"
                )

            rep.cite(
                f"{sym} option chain: implied move ±{f.get('implied_move_pct')}%, "
                f"P/C volume {f.get('put_call_volume_ratio')}, liquidity {f.get('liquidity_score')}",
                source="exchange_data", url=f"https://finance.yahoo.com/quote/{sym}/options",
                reliability=0.90, value={k: v for k, v in f.items() if k != "chain"},
                tags=["options", sym],
            )

        rep.data = {"flows": flows}
        spy = flows.get("SPY")
        if spy:
            rep.headline = (
                f"SPY implied move ±{spy['implied_move_pct']:.2f}% "
                f"({spy['expiration']}), P/C volume {spy.get('put_call_volume_ratio')}"
            )
            pc = spy.get("put_call_volume_ratio")
            if pc and not math.isnan(pc):
                # Put/call is a contrarian gauge at extremes and a confirming
                # one in the middle. Only the extremes are traded on.
                if pc > 1.35:
                    rep.signal("put_call_extreme", 0.56, weight=0.6 * ctx.weight_for(self.name),
                               note=f"P/C {pc:.2f} — hedging demand elevated, contrarian bullish")
                elif pc < 0.65:
                    rep.signal("put_call_extreme", 0.45, weight=0.6 * ctx.weight_for(self.name),
                               note=f"P/C {pc:.2f} — call-heavy complacency, contrarian bearish")
            sk = (spy.get("skew") or {}).get("skew")
            if sk is not None and sk > 8:
                rep.signal("skew_defensive", 0.46, weight=0.5 * ctx.weight_for(self.name),
                           note=f"25d skew {sk:+.1f} — protection aggressively bid")
        else:
            rep.headline = f"Option structure mapped for {len(flows)} underlyings"
        rep.confidence = 0.7
        rep.add(
            "Note: retail chain data shows volume and open interest, not the trade tape. "
            "'Unusual activity' here is inferred from volume/OI turnover and cannot distinguish "
            "an institutional buy from a dealer hedge or a closing trade."
        )
        return rep


class InstitutionalAgent(BaseAgent):
    name = "institutional"
    description = "Insider Form 4 activity and >5% ownership filings"

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Institutional and insider activity")
        focus = ctx.universe[:10]
        insider = await ctx.edgar.client.gather(
            [ctx.edgar.insider_transactions(t, 30) for t in focus], label="form4"
        )
        stakes = await ctx.edgar.client.gather(
            [ctx.edgar.institutional_13f_hint(t) for t in focus], label="13d"
        )

        clusters, moves = [], []
        for row in insider:
            if row and row.get("count", 0) >= 4:
                clusters.append(row)
                rep.add(f"{row['ticker']}: {row['signal']}")
                rep.cite(row["signal"], source="sec_edgar",
                         url=row.get("urls", [None])[0], reliability=0.98,
                         tags=["insider", row["ticker"]])
        for row in stakes:
            if row and row.get("recent_13d_13g"):
                moves.append(row)
                rep.add(f"{row['ticker']}: {row['signal']}")
                rep.cite(row["signal"], source="sec_edgar",
                         url=row["recent_13d_13g"][0]["url"], reliability=0.98,
                         tags=["ownership", row["ticker"]])

        if not clusters and not moves:
            rep.add("No insider clusters or new >5% ownership filings in the watchlist")

        rep.add(
            "Caveat carried into the report: 13F holdings lag by up to 45 days and say nothing "
            "about today. Only 13D/G filings and Form 4s are timely, and Form 4 sales under a "
            "10b5-1 plan are pre-scheduled and carry no directional information."
        )

        rep.data = {
            "insider_clusters": [{"ticker": c["ticker"], "count": c["count"]} for c in clusters],
            "ownership_moves": [{"ticker": m["ticker"], "filings": m["recent_13d_13g"]} for m in moves],
        }
        rep.headline = (
            f"{len(clusters)} insider clusters, {len(moves)} ownership filings"
            if clusters or moves else "No timely institutional footprints in the watchlist"
        )
        rep.confidence = 0.6
        return rep


class SentimentAgent(BaseAgent):
    name = "sentiment"
    description = "Measured positioning plus permissioned, low-weight community research"
    depends_on = ("volatility_regime", "options_flow", "breaking_news")

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Sentiment")

        vix = ctx.data_of("volatility_regime", "vix")
        vix_chg = ctx.data_of("volatility_regime", "vix_change_pct")
        term = ctx.data_of("volatility_regime", "term_structure", {}) or {}
        flows = ctx.data_of("options_flow", "flows", {}) or {}
        breadth = ctx.data_of("overnight_scan", "breadth_pct_up")
        news_high = ctx.data_of("breaking_news", "high_materiality", []) or []

        # Social material is evidence, not authority. TradingView reaches this
        # path only through a permitted Discord intake channel; X is queried
        # only for explicitly allowlisted handles through the official API.
        community_posts = []
        community_configured = bool(
            (ctx.config.community_discord_channel_ids and ctx.config.discord_bot_token)
            or (ctx.config.x_handles and ctx.config.x_bearer_token)
        )
        if community_configured:
            community = CommunityData(
                ctx.market.client,
                ctx.universe,
                lookback_hours=ctx.config.community_lookback_hours,
            )
            batches = await ctx.market.client.gather([
                community.discord(
                    ctx.config.community_discord_channel_ids,
                    ctx.config.discord_bot_token or "",
                ),
                community.x(ctx.config.x_handles, ctx.config.x_bearer_token or ""),
            ], label="community")
            community_posts = [post for batch in batches if batch for post in batch]

        symbol_reads = aggregate_symbol_reads(
            community_posts,
            min_sources=ctx.config.community_min_sources,
        )
        qualifying_reads = [r for r in symbol_reads.values() if r["qualifies"]]
        platform_counts: dict[str, int] = {}
        for post in community_posts:
            platform_counts[post.platform] = platform_counts.get(post.platform, 0) + 1
            rep.cite(
                f"{post.author} expressed an explicit {post.direction or 'non-directional'} "
                f"view on {', '.join(post.symbols)}",
                source="social_sentiment",
                url=post.url,
                reliability=0.30,
                value={
                    "platform": post.platform,
                    "author": post.author,
                    "direction": post.direction,
                    "engagement": post.engagement,
                },
                tags=["social", *post.symbols],
            )

        if community_configured:
            independent_accounts = len({p.source_key for p in community_posts})
            rep.add(
                f"Permissioned community intake: {len(community_posts)} relevant post(s) "
                f"from {independent_accounts} "
                f"independent account(s); {len(qualifying_reads)} symbol read(s) met the "
                f"{ctx.config.community_min_sources}-source corroboration rule"
            )
            for read in sorted(qualifying_reads,
                               key=lambda r: (-r["independent_sources"], r["symbol"]))[:8]:
                rep.add(
                    f"{read['symbol']} community {read['direction']}: "
                    f"{read['long_sources']} long / {read['short_sources']} short "
                    f"across {read['independent_sources']} independent sources"
                )
                rep.signal(
                    f"community:{read['symbol']}",
                    read["probability_up"],
                    weight=0.25 * ctx.weight_for(self.name),
                    note=(f"permissioned community consensus: {read['long_sources']} long / "
                          f"{read['short_sources']} short; cold-start social prior capped at 58/42"),
                )

        components: dict[str, float] = {}   # each in [0,1], 1 = greedy/bullish

        if vix is not None:
            # VIX 12 → greedy, VIX 35 → fearful. Linear in between is crude but
            # monotone and transparent, which beats a black box here.
            components["vix_level"] = max(0.0, min(1.0, (30 - vix) / 20))
            rep.add(f"VIX {vix:.1f} → {'complacent' if vix < 15 else 'fearful' if vix > 25 else 'neutral'}")
        if vix_chg is not None:
            components["vix_change"] = max(0.0, min(1.0, 0.5 - vix_chg / 40))
        if term.get("state") == "backwardation":
            components["term_structure"] = 0.25
        elif term.get("state") == "steep contango":
            components["term_structure"] = 0.72

        spy_flow = flows.get("SPY")
        if spy_flow and spy_flow.get("put_call_volume_ratio"):
            pc = spy_flow["put_call_volume_ratio"]
            components["put_call"] = max(0.0, min(1.0, (1.4 - pc) / 0.9))
            rep.add(f"SPY put/call volume {pc:.2f} → {'hedged/fearful' if pc > 1.1 else 'call-heavy/greedy' if pc < 0.8 else 'balanced'}")
        if breadth is not None:
            components["breadth"] = float(breadth)
            rep.add(f"Pre-market breadth {breadth:.0%} of watchlist green")

        if not components and not community_posts:
            rep.status = "degraded"
            rep.error = "no sentiment inputs available"
            rep.headline = "Sentiment: inputs unavailable"
            rep.confidence = 0.2
            return rep

        if not components:
            rep.data = {
                "score": None,
                "label": "community evidence only",
                "components": {},
                "symbol_reads": symbol_reads,
                "community_posts": len(community_posts),
                "platform_counts": platform_counts,
            }
            rep.headline = (
                f"Community research: {len(community_posts)} relevant post(s), "
                f"{len(qualifying_reads)} corroborated symbol read(s)"
            )
            rep.confidence = 0.30
            rep.add(
                "Community evidence is never used alone to authorise a trade; without "
                "measurable positioning inputs this report remains context only."
            )
            return rep

        score = sum(components.values()) / len(components)
        label = ("extreme fear" if score < 0.25 else "fear" if score < 0.42 else
                 "neutral" if score < 0.58 else "greed" if score < 0.75 else "extreme greed")

        rep.add(f"Composite sentiment {score:.2f} → {label} (from {len(components)} measurable proxies)")
        rep.cite(f"Composite sentiment {score:.2f} ({label})", source="exchange_data",
                 reliability=0.65, value=components, tags=["sentiment"])
        rep.add(
            "Community posts are excluded from the index sentiment composite. Corroborated "
            "symbol reads may only nudge a ticker probability at a capped low weight; "
            "popularity and follower counts do not count as predictive evidence."
        )
        if news_high:
            rep.add(f"{len(news_high)} high-materiality headlines are shaping the mood into the open")

        rep.data = {
            "score": score,
            "label": label,
            "components": components,
            "symbol_reads": symbol_reads,
            "community_posts": len(community_posts),
            "platform_counts": platform_counts,
        }
        rep.headline = f"Sentiment {label} ({score:.2f})"
        rep.confidence = 0.55 + 0.05 * len(components)

        # Sentiment is contrarian at the extremes only.
        if score < 0.25:
            rep.signal("sentiment_extreme", 0.57, weight=0.6 * ctx.weight_for(self.name),
                       note=f"extreme fear ({score:.2f}) — historically a poor moment to add shorts")
        elif score > 0.75:
            rep.signal("sentiment_extreme", 0.44, weight=0.6 * ctx.weight_for(self.name),
                       note=f"extreme greed ({score:.2f}) — chasing upside is late here")
        return rep
