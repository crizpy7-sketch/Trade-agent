"""Report rendering — Markdown and standalone HTML.

Structure mirrors the research workflow: what happened, what it means, what
could break it, and only then the playbook. The disclaimer is not decoration;
every idea section carries it.
"""

from __future__ import annotations

import html
from pathlib import Path

from . import clock
from .orchestrator import SwarmResult
from .recommend.speculative import build_board

DISCLAIMER = (
    "**Research and educational analysis only — not financial advice, not a recommendation "
    "to buy or sell any security, and not a solicitation. Options carry substantial risk of "
    "total loss and 0DTE contracts can lose their entire value within minutes. Every "
    "probability below is a model estimate from incomplete public data, not a guarantee. "
    "Nobody has reviewed these ideas for suitability to your circumstances. Do your own "
    "research and consult a licensed professional before risking capital.**"
)

SECTION_ORDER = [
    ("overnight_scan", "1. Overnight Scan"),
    ("global_markets", "2. Global Markets"),
    ("futures", "3. U.S. Futures, Rates & Commodities"),
    ("breaking_news", "4. Breaking News"),
    ("econ_calendar", "5. Economic Calendar"),
    ("earnings", "6. Earnings"),
    ("sec_filings", "7. SEC Filings"),
    ("options_flow", "8. Options Flow & Positioning"),
    ("institutional", "9. Institutional & Insider Activity"),
    ("sentiment", "10. Sentiment"),
    ("volatility_regime", "11. Volatility Regime"),
    ("technicals", "12. Technical Analysis"),
    ("cross_verify", "13. Cross-Verification"),
    ("risk", "14. Risk Assessment"),
]


DEMO_BANNER = (
    "> # ⚠ SAMPLE REPORT — SYNTHETIC DATA\n"
    "> **Every price, ticker, headline and filing below is fabricated test data.** The\n"
    "> fixtures generate each symbol from a random walk starting at $100, so a symbol\n"
    "> labelled NVDA may show $118 when the real stock trades at $225. Strikes are chosen\n"
    "> from those fake prices.\n"
    ">\n"
    "> **Do not trade anything on this page.** It exists to show the layout and the maths.\n"
    "> On a live install every number is replaced by the real quote."
)


def render_markdown(result: SwarmResult, narrative: str | None = None,
                    demo: bool = False) -> str:
    d = result.run_date
    status = clock.day_status(d)
    out: list[str] = []

    out.append(f"# Pre-Market Intelligence — {d:%A, %d %B %Y}")
    out.append("")
    if demo:
        out.append(DEMO_BANNER)
        out.append("")
    out.append(f"*Generated {result.started_at.astimezone(clock.ET):%H:%M:%S ET} · "
               f"{result.duration_seconds:.1f}s · "
               f"{result.agents_ok} agents reporting, {result.agents_failed} degraded*")
    out.append("")

    if not result.market_open:
        out.append(f"## Market closed — {result.closed_reason}")
        out.append("")
        out.append(f"No pre-market session required. Next trading day: "
                   f"{clock.next_trading_day(d):%A, %d %B %Y}.")
        return "\n".join(out)

    out.append(f"**Session**: {status.summary}")
    out.append("")

    # ---- executive summary ----
    out.append("## Executive Summary")
    out.append("")
    out.append(f"- **Directional read**: P(SPY closes above the open) = **{result.probability:.0%}**")
    cv = result.reports.get("cross_verify")
    if cv and cv.usable:
        lo, hi = cv.data.get("interval", (0, 0))
        out.append(f"- **Uncertainty band**: {lo:.0%}–{hi:.0%} "
                   f"({cv.data.get('effective_n', 0):.1f} effective independent signals from "
                   f"{cv.data.get('raw_signal_count', 0)} raw)")
    out.append(f"- **Confidence**: {result.confidence}/100")
    vr = result.reports.get("volatility_regime")
    if vr and vr.usable:
        out.append(f"- **Regime**: {vr.data.get('regime', 'unknown').replace('_', ' ')} — "
                   f"{vr.data.get('regime_playbook', '')}")
    rk = result.reports.get("risk")
    if rk and rk.usable:
        out.append(f"- **Risk budget**: {rk.data.get('suggested_risk_pct')}% per idea, "
                   f"max {rk.data.get('max_concurrent')} concurrent")
    out.append("")

    if narrative:
        out.append("### Narrative")
        out.append("")
        out.append(narrative)
        out.append("")

    # ---- agent sections ----
    for key, title in SECTION_ORDER:
        rep = result.reports.get(key)
        if not rep:
            continue
        out.append(f"## {title}")
        out.append("")
        badge = {"ok": "", "degraded": " *(degraded)*", "failed": " *(failed)*",
                 "skipped": " *(skipped)*"}[rep.status]
        out.append(f"**{rep.headline}**{badge}")
        out.append("")
        if rep.error:
            out.append(f"> Error: {rep.error}")
            out.append("")
        for f in rep.findings:
            out.append(f"- {f}")
        out.append("")
        if rep.evidence:
            out.append("<details><summary>Sources</summary>")
            out.append("")
            for e in rep.evidence[:20]:
                link = f"[{e.source}]({e.url})" if e.url else e.source
                out.append(f"- {e.claim} — {link} (reliability {e.reliability:.2f})")
            out.append("")
            out.append("</details>")
            out.append("")

    # ---- playbook ----
    out.append("---")
    out.append("")
    out.append("# Day Trading Playbook")
    out.append("")
    if demo:
        out.append(DEMO_BANNER)
        out.append("")
    out.append(DISCLAIMER)
    out.append("")

    pub = result.publication
    if pub.suppressed:
        out.append("## Recommendations withheld")
        out.append("")
        out.append(f"> **No recommendation is published for this session.** "
                   f"{pub.suppression_reason}")
        out.append("")
        out.append("The analysis below is still shown, because the observations are "
                   "real and may be useful. What is missing is the part that turns "
                   "observations into a position, and that part is missing on purpose "
                   "rather than by omission.")
        out.append("")
        screened = result.screened_ideas
        out.extend(_render_screened_option_block(
            "## 1. Best Call Options — Three Screening Slots",
            screened.get("calls", []), "call", demo))
        out.extend(_render_screened_option_block(
            "## 2. Best Put Options — Three Screening Slots",
            screened.get("puts", []), "put", demo))
        out.extend(_render_review_summary(pub))
        out.extend(_render_supporting_sections(result))
        return "\n".join(out)

    pb = result.reports.get("playbook")
    if (not pb or not pb.usable) and not pub.active():
        out.append("The playbook could not be constructed — insufficient data this morning.")
        if pb and pb.error:
            out.append(f"> {pb.error}")
        out.append("")
        # The board still renders. A failed playbook is exactly the day the
        # owner asked about, and this exit used to take the board down with it.
        out.extend(_render_supporting_sections(result))
        return "\n".join(out)

    # The publication set is the source of the ideas below; the playbook
    # agent only supplies the headline it happened to write.
    if pb and pb.usable:
        out.append(f"*{pb.headline}*")
        out.append("")

    # Six fixed screening slots, not six forced recommendations. Rejected and
    # weak candidates are visible only with their status attached; the active
    # ideas view remains approved-only and is still the sole input to tracking.
    screened = result.screened_ideas
    ideas = result.ideas
    out.extend(_render_screened_option_block(
        "## 1. Best Call Options — Three Screening Slots",
        screened.get("calls", []), "call", demo))
    out.extend(_render_screened_option_block(
        "## 2. Best Put Options — Three Screening Slots",
        screened.get("puts", []), "put", demo))
    out.extend(_render_stock_block(ideas.get("stocks", []), demo))
    out.extend(_render_non_actionable(pub))
    out.extend(_render_review_summary(pub))

    out.extend(_render_supporting_sections(result))
    return "\n".join(out)


def _render_daily_contracts(result) -> list[str]:
    """Six contracts, every session, whatever else the run decided.

    Rendered from _render_supporting_sections precisely because that is called
    on both the normal and the suppressed path — a board that disappears on the
    days the pipeline gives up is not a daily board.

    A filled slot here is deliberately NOT an endorsement. The tier on each row
    says how much belief is behind it, and on most days the honest answer is
    none. That separation is the only thing that makes always-emitting safe:
    without it, a fixed board plus a duty to fill it is a machine for
    manufacturing conviction.
    """
    flows = None
    of = result.reports.get("options_flow")
    if of is not None and getattr(of, "data", None):
        flows = of.data.get("flows")

    board = build_board(flows)

    out = ["## 3d. Daily Contract Board — six slots, filled every session", ""]
    out.append(
        "*Not recommendations. Each row states the tier behind it: QUALIFIED "
        "cleared the swarm's gates, SPECULATIVE means no edge was established, "
        "LOTTERY means the contract is real but the data is thin, NO CHAIN "
        "means the option data was missing. Nothing here is scored or added to "
        "the track record.*"
    )
    out.append("")

    for title, key, right in (("Calls", "calls", "call"), ("Puts", "puts", "put")):
        out.append(f"### {title}")
        out.append("")
        for n, row in enumerate(board[key], 1):
            if row["strike"] is None:
                out.append(f"**Slot {n} — {row['tier']}**")
                out.append("")
                out.append(f"> {row['reason']}")
                out.append("")
                continue
            out.append(
                f"**Slot {n} — {row['tier']}: {row['symbol']} {row['strike']:g} "
                f"{right.upper()} · exp {row['expiration']}**"
            )
            out.append("")
            out.append(f"> {row['reason']}")
            out.append("")
            out.append("| | |")
            out.append("|---|---|")
            out.append(f"| Premium (mid) | ${row['premium']:.2f} |")
            out.append(f"| Cost, 1 contract | ${row['cost']:,.2f} |")
            out.append(f"| Max loss | ${row['max_loss']:,.2f} — the whole premium |")
            out.append(f"| Breakeven at expiry | ${row['breakeven']:.2f} |")
            if row.get("delta") is not None:
                out.append(f"| Delta | {row['delta']:.2f} |")
            if row.get("spread_pct") is not None:
                out.append(f"| Bid/ask spread | {row['spread_pct']:.1f}% of mid |")
            out.append(f"| Open interest / volume | {row['open_interest']:,} / {row['volume']:,} |")
            out.append("")

    return out


def _render_supporting_sections(result) -> list[str]:
    """Red team and the reading guide — shown whether or not anything was
    published, because the objections are informative either way."""
    out: list[str] = _render_daily_contracts(result)

    rt = result.reports.get("red_team")
    if rt and rt.usable and rt.data.get("objections"):
        out.append("## 4. Red Team — the case against everything above")
        out.append("")
        out.append(f"*{rt.headline}*")
        out.append("")
        for o in rt.data["objections"]:
            out.append(f"- **[{o['severity']}]** {o['objection']}")
            out.append(f"  - *Test:* {o['test']}")
        out.append("")
        if rt.data.get("recommend_stand_down"):
            out.append("> **Two or more high-severity objections stand.** Today's conviction is not "
                       "supported by independent evidence. Trade below the risk budget, or not at all.")
            out.append("")

    out.append("## How to read these numbers")
    out.append("")
    out.append(
        "- **Probability** is P(target touched before stop) from a Student-t Monte Carlo on the "
        "symbol's own realized volatility — a path simulation, not a subjective confidence. It is "
        "usually lower than intuition suggests, because a one-session horizon frequently resolves "
        "to neither barrier.\n"
        "- **Expected R** is net of assumed round-trip friction. Ideas that fail to clear it are "
        "still listed, but labelled *watchlist only* — hiding them would leave the impression that "
        "nothing was considered.\n"
        "- **Confidence** blends signal strength, breadth of independent evidence, and agreement. "
        "A 60% probability at 30/100 confidence means the evidence is thin, not that the setup is safe.\n"
        "- **Option premium levels** are Black-Scholes repricings at the bracket levels using each "
        "contract's own implied volatility, with half the remaining life burned. They are estimates: "
        "verify against the live chain, and expect the real fill to be worse."
    )
    out.append("")
    out.append("---")
    out.append("")
    out.append(DISCLAIMER)
    out.append("")
    return out


def _render_non_actionable(pub) -> list[str]:
    """Recommendations that are real outputs but are not trades.

    A WATCH, a NO_EDGE and an INSUFFICIENT_EVIDENCE are conclusions. 1.x had
    no way to express them, so they were silently absent; showing them is how
    a reader can tell the difference between "we looked and found nothing" and
    "we did not look".
    """
    others = [r for r in pub.active() if not r.actionable]
    if not others:
        return []

    out = ["## 3b. Considered and not recommended", ""]
    out.append("| Symbol | Verdict | Conviction | Why |")
    out.append("|---|---|---|---|")
    for r in sorted(others, key=lambda x: x.subject):
        why = (r.uncertainty_notes[0] if r.uncertainty_notes else r.rationale)[:160]
        out.append(f"| {r.subject} | {r.rec_type.value} | {r.conviction.value} | {why} |")
    out.append("")
    return out


def _render_review_summary(pub) -> list[str]:
    """What the review gate did. Rejected candidates appear here — as review
    history, which is the only place they are ever allowed to appear."""
    if not pub.rejected and not pub.suppressed:
        return []

    out = ["## 3c. Review gate", ""]
    s = pub.summary()
    out.append(f"*{s['approved']} approved, {s['modified']} modified, "
               f"{s['rejected']} rejected across {pub.review_iterations} "
               f"review iteration(s).*")
    out.append("")
    if pub.rejected:
        out.append("**Rejected before publication.** These were built, attacked, and did "
                   "not survive. They are listed so the reasoning is auditable — they are "
                   "not recommendations and are not tracked as ideas.")
        out.append("")
        for r in pub.rejected:
            out.append(f"- **{r.subject}** — {r.reason}")
            for f in r.findings[:3]:
                out.append(f"  - {f}")
        out.append("")
    return out


def _render_idea_block(title: str, ideas: list[dict], kind: str, demo: bool = False) -> list[str]:
    out = [title, ""]
    if not ideas:
        out.append(f"No {kind} candidate could be constructed today — either no symbol carried a "
                   f"{'bullish' if kind == 'call' else 'bearish'} bias, or structure left no room "
                   f"between a sane stop and the next level. Standing aside is a position.")
        out.append("")
        return out

    for n, i in enumerate(ideas, 1):
        tag = "SAMPLE / FAKE PRICE — " if demo else ""
        out.append(f"### {n}. {tag}{i['symbol']} {i.get('strike', '')} {kind.upper()} "
                   f"{'exp ' + i['expiration'] if i.get('expiration') else ''}")
        out.append("")
        out.append("| | |")
        out.append("|---|---|")
        out.append(f"| Underlying | {i['symbol']} at {i['entry']:.2f} |")
        if i.get("strike"):
            out.append(f"| Strike / expiration | {i['strike']:g} {kind} · {i.get('expiration', 'nearest weekly')} |")
        if i.get("option_entry"):
            out.append(f"| Premium entry zone | {i['option_entry']} |")
            out.append(f"| Premium target | {i['option_target']} |")
            out.append(f"| Premium stop | {i['option_stop']} |")
        out.append(f"| Underlying target | {i['target']:.2f} |")
        out.append(f"| Underlying stop | {i['stop']:.2f} |")
        out.append(f"| P(target before stop) | **{i['probability']:.0%}** |")
        out.append(f"| Expected value | **{i['expected_r']:+.2f}R** after costs — {i.get('ev_verdict', '')} |")
        out.append(f"| Confidence | {i['confidence']}/100 |")
        out.append("")
        if not i.get("clears_bar", True):
            out.append("> **This idea does not clear expected value after trading costs.** It is listed "
                       "because it was the closest candidate in its category, not because the math "
                       "supports taking it. Treat it as a watchlist item.")
            out.append("")
        out.append(f"**Rationale.** {i['rationale']}")
        out.append("")
        if i.get("evidence"):
            out.append("**Supporting evidence.**")
            for e in i["evidence"][:6]:
                out.append(f"- {e}")
            out.append("")
        if i.get("math_note"):
            out.append(f"**Math.** {i['math_note']}")
            out.append("")
        if i.get("option_note"):
            out.append(f"**Option pricing.** {i['option_note']}")
            out.append("")
        out.append(f"**Invalidated if:** {i['invalidation']}")
        out.append("")
    return out


def _render_screened_option_block(title: str, rows: list[dict], kind: str,
                                  demo: bool = False) -> list[str]:
    """Render exactly three slots without turning symmetry into endorsement."""
    out = [title, ""]
    out.append(
        "*QUALIFIED cleared both gates. WATCH ONLY, REJECTED, and WITHHELD are "
        "research/audit context, not trade recommendations. DATA UNAVAILABLE is "
        "intentionally left blank.*"
    )
    out.append("")
    for n, row in enumerate(rows, 1):
        status = str(row.get("screen_status") or "WITHHELD")
        symbol = row.get("symbol")
        if not symbol or status in {"REJECTED", "WITHHELD", "DATA UNAVAILABLE"}:
            subject = f" {symbol}" if symbol else ""
            strike = (f" {row['strike']:g}" if symbol and row.get("strike")
                      else "")
            out.append(f"**Slot {n} — {status}:{subject}{strike} {kind.upper()}**")
            out.append("")
            out.append(f"> **{status}.** {row.get('screen_reason', 'No candidate available.')}")
            out.append("")
            continue

        tag = "SAMPLE / FAKE PRICE — " if demo else ""
        strike = f" {row['strike']:g}" if row.get("strike") else ""
        expiry = f" · exp {row['expiration']}" if row.get("expiration") else ""
        out.append(f"### {n}. {tag}{symbol}{strike} {kind.upper()}{expiry}")
        out.append("")
        out.append(f"> **{status}.** {row.get('screen_reason', '')}")
        out.append("")
        out.append("| | |")
        out.append("|---|---|")
        if row.get("entry") is not None:
            out.append(f"| Underlying entry | {float(row['entry']):.2f} |")
        if row.get("target") is not None:
            out.append(f"| Underlying target | {float(row['target']):.2f} |")
        if row.get("stop") is not None:
            out.append(f"| Underlying stop | {float(row['stop']):.2f} |")
        if row.get("option_entry"):
            out.append(f"| Premium entry zone | {row['option_entry']} |")
            out.append(f"| Premium target | {row.get('option_target', 'unavailable')} |")
            out.append(f"| Premium stop | {row.get('option_stop', 'unavailable')} |")
        if row.get("probability") is not None:
            out.append(f"| P(target before stop) | {float(row['probability']):.0%} |")
        if row.get("expected_r") is not None:
            out.append(f"| Expected value | {float(row['expected_r']):+.2f}R after costs |")
        if row.get("confidence") is not None:
            out.append(f"| Confidence | {int(row['confidence'])}/100 |")
        out.append("")
        if row.get("rationale"):
            out.append(f"**Screening rationale.** {row['rationale']}")
            out.append("")
        if row.get("invalidation"):
            invalidation = row["invalidation"]
            if isinstance(invalidation, list):
                invalidation = "; ".join(str(x) for x in invalidation)
            out.append(f"**Invalidated if:** {invalidation}")
            out.append("")
    return out


def _render_stock_block(ideas: list[dict], demo: bool = False) -> list[str]:
    out = ["## 3. Best Stocks for Day Trading", ""]
    if demo:
        out.append("*Every price in this table is synthetic test data, not a real quote.*")
        out.append("")
    if not ideas:
        out.append("No stock setup cleared the expectancy bar today.")
        out.append("")
        return out

    out.append("| # | Symbol | Bias | Entry | Target | Stop | P(hit) | EV | Confidence | Clears cost bar |")
    out.append("|---|--------|------|-------|--------|------|--------|-----|------------|-----------------|")
    for n, i in enumerate(ideas, 1):
        out.append(
            f"| {n} | **{i['symbol']}** | {i['direction']} | {i['entry']:.2f} | {i['target']:.2f} | "
            f"{i['stop']:.2f} | {i['probability']:.0%} | {i['expected_r']:+.2f}R | {i['confidence']}/100 | "
            f"{'yes' if i.get('clears_bar', True) else 'no — watchlist only'} |"
        )
    out.append("")
    for n, i in enumerate(ideas, 1):
        out.append(f"**{n}. {i['symbol']} — {i['direction']}**"
                   + ("" if i.get("clears_bar", True) else " *(negative expectancy after costs)*"))
        out.append("")
        out.append(f"{i['rationale']}")
        out.append("")
        out.append(f"*Key levels:* entry {i['entry']:.2f} · target {i['target']:.2f} · stop {i['stop']:.2f}")
        out.append("")
        out.append(f"*Invalidated if:* {i['invalidation']}")
        out.append("")
    return out


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pre-Market Intelligence — {date}</title>
<style>
:root {{ --bg:#ffffff; --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --accent:#0b6bcb;
         --up:#0a7d3f; --down:#b3261e; --card:#f7f8fa; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#14161a; --fg:#e8e8e8; --muted:#9aa0a6; --line:#2a2e35; --accent:#5aa9f7;
           --up:#4ade80; --down:#f87171; --card:#1c1f26; }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:2rem 1rem; background:var(--bg); color:var(--fg);
        font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif; }}
main {{ max-width:52rem; margin:0 auto; }}
h1 {{ font-size:1.9rem; margin:0 0 .3rem; }}
h2 {{ font-size:1.3rem; margin:2.2rem 0 .6rem; padding-bottom:.3rem; border-bottom:1px solid var(--line); }}
h3 {{ font-size:1.08rem; margin:1.6rem 0 .5rem; color:var(--accent); }}
table {{ border-collapse:collapse; width:100%; margin:.8rem 0; font-size:.93rem; }}
th,td {{ text-align:left; padding:.45rem .6rem; border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-weight:600; }}
.wrap {{ overflow-x:auto; }}
blockquote {{ margin:.8rem 0; padding:.7rem 1rem; background:var(--card);
              border-left:3px solid var(--accent); border-radius:0 6px 6px 0; }}
.disclaimer {{ background:var(--card); border:1px solid var(--line); border-radius:8px;
               padding:1rem; font-size:.9rem; color:var(--muted); margin:1.2rem 0; }}
details {{ margin:.6rem 0; }} summary {{ cursor:pointer; color:var(--muted); font-size:.9rem; }}
code {{ background:var(--card); padding:.1rem .3rem; border-radius:4px; font-size:.9em; }}
ul {{ padding-left:1.2rem; }} li {{ margin:.25rem 0; }}
.meta {{ color:var(--muted); font-size:.88rem; }}
.demo-bar {{ position:sticky; top:0; z-index:99; background:#b3261e; color:#fff;
             padding:.6rem 1rem; margin:-2rem -1rem 1.5rem; font-weight:700;
             text-align:center; font-size:.95rem; }}
hr {{ border:0; border-top:1px solid var(--line); margin:2rem 0; }}
</style></head><body><main>
{body}
</main></body></html>
"""


def render_html(result: SwarmResult, narrative: str | None = None, demo: bool = False) -> str:
    """Minimal Markdown→HTML conversion, dependency-free."""
    md = render_markdown(result, narrative, demo)
    body: list[str] = []
    in_table = in_list = False

    for line in md.split("\n"):
        stripped = line.strip()

        if stripped.startswith("<details") or stripped == "</details>":
            if in_list:
                body.append("</ul>"); in_list = False
            body.append(stripped)
            continue

        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells if c):
                continue
            if not in_table:
                body.append('<div class="wrap"><table>')
                in_table = True
            tag = "td"
            body.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table></div>")
            in_table = False

        if stripped.startswith("- "):
            if not in_list:
                body.append("<ul>"); in_list = True
            body.append(f"<li>{_inline(stripped[2:])}</li>")
            continue
        if in_list:
            body.append("</ul>"); in_list = False

        if not stripped:
            continue
        if stripped == "---":
            body.append("<hr>")
        elif stripped.startswith("### "):
            body.append(f"<h3>{_inline(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            body.append(f"<h2>{_inline(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            body.append(f"<h1>{_inline(stripped[2:])}</h1>")
        elif stripped.startswith("> "):
            body.append(f"<blockquote>{_inline(stripped[2:])}</blockquote>")
        elif stripped.startswith("**Research and educational"):
            body.append(f'<div class="disclaimer">{_inline(stripped)}</div>')
        elif stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
            body.append(f'<p class="meta">{_inline(stripped)}</p>')
        else:
            body.append(f"<p>{_inline(stripped)}</p>")

    if in_table:
        body.append("</table></div>")
    if in_list:
        body.append("</ul>")

    page = "\n".join(body)
    if demo:
        page = ('<div class="demo-bar">⚠ SAMPLE REPORT — ALL PRICES ARE SYNTHETIC TEST DATA. '
                'DO NOT TRADE.</div>') + page
    return HTML_TEMPLATE.format(date=f"{result.run_date:%d %B %Y}", body=page)


def _inline(text: str) -> str:
    import re
    t = html.escape(text)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" rel="noopener">\1</a>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t


def write_reports(result: SwarmResult, report_dir: Path, narrative: str | None = None,
                  demo: bool = False) -> dict[str, Path]:
    report_dir = Path(report_dir).expanduser()
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.run_date.isoformat()

    md_path = report_dir / f"premarket-{stamp}.md"
    html_path = report_dir / f"premarket-{stamp}.html"
    md_path.write_text(render_markdown(result, narrative, demo), encoding="utf-8")
    html_path.write_text(render_html(result, narrative, demo), encoding="utf-8")

    latest = report_dir / "latest.html"
    try:
        latest.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError:
        pass

    return {"markdown": md_path, "html": html_path, "latest": latest}
