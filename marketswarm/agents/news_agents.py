"""Information agents: breaking news, the economic calendar, earnings
reactions, and SEC filings."""

from __future__ import annotations

from ..providers.earnings import gap_persistence_prior, reaction_quality
from .base import AgentReport, BaseAgent, SwarmContext


class BreakingNewsAgent(BaseAgent):
    name = "breaking_news"
    description = "Overnight headlines, deduplicated and corroboration-weighted"

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Breaking news")
        headlines = await ctx.news.overnight_headlines(max_age_hours=18)
        if not headlines:
            rep.status = "degraded"
            rep.error = "no headlines retrieved"
            rep.headline = "Breaking news: no feeds returned data"
            rep.confidence = 0.2
            return rep

        high = [h for h in headlines if h.materiality == "high"]
        medium = [h for h in headlines if h.materiality == "medium"]

        for h in (high + medium)[:12]:
            corr = f", corroborated by {h.corroborations} outlets" if h.corroborations > 1 else " (single source)"
            rep.add(f"[{h.materiality}] {h.title} — {h.source}, {h.age_minutes:.0f}m ago{corr}")
            rep.evidence.append(h.to_evidence())

        single_source_high = [h for h in high if h.corroborations == 1]
        if single_source_high:
            rep.add(
                f"{len(single_source_high)} high-materiality headlines are single-source and "
                f"unconfirmed — treated as unverified until a second outlet carries them"
            )

        rep.data = {
            "count": len(headlines),
            "high_materiality": [
                {"title": h.title, "source": h.source, "url": h.url,
                 "corroborations": h.corroborations, "tickers": h.tickers,
                 "age_minutes": round(h.age_minutes)}
                for h in high[:15]
            ],
            "ticker_mentions": _mention_counts(headlines),
        }
        rep.headline = f"{len(high)} high-materiality and {len(medium)} medium headlines overnight"
        rep.confidence = 0.7 if high else 0.5
        return rep


def _mention_counts(headlines) -> dict[str, int]:
    counts: dict[str, int] = {}
    for h in headlines:
        for t in h.tickers:
            counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:15])


class EconCalendarAgent(BaseAgent):
    name = "econ_calendar"
    description = "Scheduled macro releases and their timing relative to the open"

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Economic calendar")
        events = await ctx.econ.todays_calendar(ctx.run_date)
        macro = await ctx.econ.macro_dashboard()
        curve = await ctx.econ.treasury_curve()

        if not events:
            rep.add("No scheduled high-impact U.S. releases today")
        for e in events:
            rep.add(e.describe() + ("".join(f" — {n}" for n in e.notes)))
            rep.cite(e.describe(), source=e.source,
                     url="https://www.bls.gov/schedule/news_release/" if "CPI" in e.name else None,
                     reliability=0.9, tags=["econ", e.impact])

        if curve and curve.get("tenors"):
            tenors = curve["tenors"]
            rep.add("Treasury par curve: " + ", ".join(f"{k} {v:.2f}%" for k, v in list(tenors.items())[:6]))
            rep.cite("U.S. Treasury par yield curve retrieved",
                     source="treasury", url="https://home.treasury.gov", reliability=0.97, tags=["rates"])

        if macro.get("available"):
            for sid, s in macro.get("series", {}).items():
                if s.get("value") is not None:
                    rep.add(f"{s['label']}: {s['value']} (as of {s['date']})")
        else:
            rep.add(macro.get("note", "FRED unavailable"))

        pre_open = [e for e in events if e.before_open]
        very_high = [e for e in events if e.impact == "very high"]

        rep.data = {
            "events": [{"name": e.name, "time_et": e.time_et, "impact": e.impact,
                        "before_open": e.before_open} for e in events],
            "pre_open_count": len(pre_open),
            "very_high_impact": [e.name for e in very_high],
            "macro": macro,
            "curve": curve,
            "event_risk": "high" if very_high else "medium" if events else "low",
        }
        rep.headline = (
            f"{len(events)} scheduled releases, {len(pre_open)} before the open"
            + (f" — including {very_high[0].name}" if very_high else "")
        )
        rep.confidence = 0.85

        if very_high:
            rep.add(
                "Event risk is the dominant factor today: pre-release ranges compress and the "
                "post-release move frequently reverses the first impulse. Directional conviction "
                "before the print should be low regardless of what other signals say."
            )
            # Not a direction signal — a confidence damper the risk agent reads.
            rep.data["conviction_damper"] = 0.5
        return rep


class EarningsAgent(BaseAgent):
    name = "earnings"
    description = "Names reacting to results and names reporting tonight"
    depends_on = ("overnight_scan",)

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Earnings")
        window = await ctx.earnings.relevant_window(ctx.run_date, ctx.prev_session)
        profiles = ctx.data_of("overnight_scan", "profiles", {}) or {}

        reacting = window.get("reacting_today", [])
        tonight = window.get("reporting_tonight", [])

        if not reacting and not tonight:
            rep.status = "degraded"
            rep.headline = "Earnings: calendar unavailable or empty"
            rep.add("No large-cap earnings events identified for this session")
            rep.confidence = 0.3
            return rep

        analyses = []
        for e in reacting[:10]:
            gap = (profiles.get(e.ticker) or {}).get("gap_pct")
            q = reaction_quality(e.surprise_pct, gap)
            analyses.append({
                "ticker": e.ticker, "company": e.company, "timing": e.timing,
                "surprise_pct": e.surprise_pct, "gap_pct": gap,
                "read": q["read"], "bias": q["bias"], "confidence": q["confidence"],
                "prior_persistence": gap_persistence_prior(gap, True) if gap is not None else None,
            })
            rep.add(f"{e.ticker}: {q['read']} → {q['bias']}")
            rep.cite(
                f"{e.ticker} earnings reaction: surprise {e.surprise_pct if e.surprise_pct is not None else 'n/a'}, "
                f"gap {gap if gap is not None else 'n/a'}",
                source="company_ir", reliability=0.88,
                url=f"https://finance.yahoo.com/quote/{e.ticker}",
                value=analyses[-1], tags=["earnings", e.ticker],
            )

        for e in tonight[:8]:
            rep.add(f"{e.ticker} reports after today's close — IV inflated, expect compression into the bell")

        rep.data = {
            "reacting": analyses,
            "reporting_tonight": [{"ticker": e.ticker, "company": e.company} for e in tonight[:10]],
            "counts": window.get("counts", {}),
        }
        rep.headline = (
            f"{len(reacting)} large caps trading on results, {len(tonight)} report tonight"
        )
        rep.confidence = 0.7

        # Earnings reactions inform single names, not the index — signals here
        # are attached per symbol and consumed by the ranking agent.
        for a in analyses:
            if a["prior_persistence"] is None:
                continue
            bullish = "bullish" in a["bias"]
            p = a["prior_persistence"] if bullish else 1 - a["prior_persistence"]
            rep.signal(f"earnings_reaction:{a['ticker']}", p,
                       weight=0.9 * ctx.weight_for(self.name) * a["confidence"],
                       note=a["read"])
        return rep


class SECFilingsAgent(BaseAgent):
    name = "sec_filings"
    description = "Overnight EDGAR filings — primary-source catalysts"

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="SEC filings")
        filings = await ctx.edgar.scan_universe(ctx.universe, lookback_hours=20)

        if not filings:
            rep.add("No material EDGAR filings from watchlist names in the last 20 hours")
            rep.headline = "SEC filings: nothing material overnight"
            rep.confidence = 0.6
            rep.data = {"filings": [], "high_impact": []}
            return rep

        high = [f for f in filings if f.impact == "high"]
        for f in filings[:12]:
            rep.add(
                f"{f.ticker} {f.plain_english()} filed {f.age_hours:.1f}h ago"
                + (" [HIGH IMPACT]" if f.impact == "high" else "")
            )
            rep.cite(
                f"{f.ticker} filed {f.form}"
                + (f" (items {', '.join(f.items)})" if f.items else ""),
                source="sec_edgar", url=f.url, reliability=0.98,
                value={"form": f.form, "items": f.items, "filed_at": f.filed_at.isoformat()},
                tags=["filing", f.ticker, f.impact],
            )

        rep.data = {
            "filings": [
                {"ticker": f.ticker, "form": f.form, "items": f.items, "url": f.url,
                 "impact": f.impact, "age_hours": round(f.age_hours, 1),
                 "meaning": f.plain_english()}
                for f in filings[:20]
            ],
            "high_impact": [f.ticker for f in high],
        }
        rep.headline = (
            f"{len(filings)} overnight filings, {len(high)} high-impact"
            + (f" ({', '.join(sorted(set(f.ticker for f in high))[:4])})" if high else "")
        )
        rep.confidence = 0.9  # primary source — the most trustworthy input available
        return rep
