"""Delivery: webhook push for Discord/Slack/Telegram-compatible endpoints."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from . import __version__

log = logging.getLogger("marketswarm.notify")

MAX_LEN = 1800

# Discord's edge rejects urllib's default `Python-urllib/x.y` User-Agent with a
# bare 403 before the payload is ever looked at — which is why a webhook that
# answers curl can still refuse this client. Identifying ourselves is the fix.
# Slack and Telegram accept the same header, so it is set unconditionally.
USER_AGENT = f"MarketSwarm/{__version__} (+https://github.com/crizpy7-sketch/marketswarm)"


from .recommend.speculative import build_board, filled_count


def _contract_lines(result) -> list[str]:
    """The daily board, condensed for a 1800-character message.

    Placed before the stocks block on purpose: summarize truncates by a blind
    slice, so anything appended last is what gets cut. This is the part the
    owner asked to receive daily, so it does not go last.

    When nothing was reachable the board is six identical NO CHAIN rows. Six
    copies of the same sentence is not more informative than one, and it would
    crowd out the rest of the message, so the empty case collapses to a single
    line that still says why.
    """
    of = result.reports.get("options_flow")
    flows = of.data.get("flows") if (of is not None and getattr(of, "data", None)) else None
    pub = getattr(result, "publication", None)
    rejected = pub.rejected_subjects() if pub is not None else set()
    board = build_board(flows, excluded_symbols=rejected)

    if filled_count(board) == 0:
        return ["\n__Contract board__",
                "No live option chains reached this run — every slot is empty for "
                "want of data, not for want of conviction."]

    out: list[str] = []
    # Owner's decision, 2026-08-24: the board still goes out on a day the red
    # team rejects everything, but that disagreement leads the message rather
    # than sitting quietly under it. The alternative considered was withholding
    # the board entirely on such days; it was rejected because a board that
    # disappears on the worst days is not a daily board. What is NOT negotiable
    # is that the rejected names themselves are barred from it — see
    # build_board(excluded_symbols=...).
    if pub is not None and pub.rejected and not pub.approved:
        out.append("\n⚠️ __RED TEAM REJECTED EVERY IDEA TODAY__")
        out.append("The swarm's adversary found every candidate unsound. What "
                   "follows is below even the usual speculative bar, and the "
                   "rejected names are excluded from it.")

    out.append("\n__Contract board__ *(tier, not endorsement — never scored)*")
    for key, right in (("calls", "C"), ("puts", "P")):
        for row in board[key]:
            if row["strike"] is None:
                continue
            out.append(
                f"• {row['symbol']} {row['strike']:g}{right} {row['expiration']} "
                f"**{row['tier']}** — ${row['premium']:.2f}, max loss "
                f"${row['max_loss']:,.0f}, breakeven ${row['breakeven']:.2f}"
            )
    return out


def summarize(result) -> str:
    lines = [
        f"**Pre-Market — {result.run_date:%a %d %b %Y}**",
        f"P(up session) {result.probability:.0%} · confidence {result.confidence}/100 · "
        f"{result.agents_ok} agents ok, {result.agents_failed} degraded",
    ]
    vr = result.reports.get("volatility_regime")
    if vr and vr.usable:
        lines.append(f"Regime: {vr.data.get('regime', 'unknown').replace('_', ' ')}")

    pub = getattr(result, "publication", None)
    if pub is not None and pub.suppressed:
        lines.append(f"\n__No recommendations published__\n{pub.suppression_reason}")

    # Always show three slots on each side. A fixed board is not a fixed number
    # of recommendations: every row carries QUALIFIED, WATCH ONLY, REJECTED,
    # WITHHELD, or DATA UNAVAILABLE, and only QUALIFIED rows enter tracking.
    screened = result.screened_ideas
    for label, key in (("3 Call candidates", "calls"),
                       ("3 Put candidates", "puts")):
        lines.append(f"\n__{label}__")
        for n, row in enumerate(screened.get(key, []), 1):
            status = row.get("screen_status", "WITHHELD")
            symbol = row.get("symbol")
            if not symbol:
                lines.append(f"{n}. **{status}** — {row.get('screen_reason', '')[:95]}")
                continue
            strike = f" {row['strike']:g}" if row.get("strike") else ""
            metrics = []
            if row.get("probability") is not None:
                metrics.append(f"P {float(row['probability']):.0%}")
            if row.get("expected_r") is not None:
                metrics.append(f"EV {float(row['expected_r']):+.2f}R")
            suffix = f" · {', '.join(metrics)}" if metrics else ""
            lines.append(f"{n}. {symbol}{strike} — **{status}**{suffix}")

    lines.extend(_contract_lines(result))

    # Preserve the separately screened stock ideas. The fixed six-slot board
    # applies only to options; omitting stocks here would silently remove an
    # existing downstream view for otherwise publishable recommendations.
    stocks = (result.ideas or {}).get("stocks", [])
    if stocks:
        lines.append("\n__Stocks__")
        for idea in stocks:
            lines.append(
                f"• {idea['symbol']} {idea['direction']} — entry {idea['entry']:.2f}, "
                f"target {idea['target']:.2f}, stop {idea['stop']:.2f} "
                f"(P {idea['probability']:.0%}, EV {idea['expected_r']:+.2f}R)"
            )
    lines.append("\n_Research/educational analysis only — not financial advice._")
    text = "\n".join(lines)
    return text[:MAX_LEN] + ("…" if len(text) > MAX_LEN else "")


def send_webhook(url: str, text: str, timeout: float = 15.0) -> bool:
    """Post to a webhook. `content` suits Discord, `text` suits Slack — both
    keys are sent so one endpoint config works for either."""
    payload = json.dumps({"content": text, "text": text}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = 200 <= resp.status < 300
            if not ok:
                log.warning("webhook returned %s", resp.status)
            return ok
    except urllib.error.HTTPError as exc:
        # The status alone does not say why. Discord and Slack both explain the
        # refusal in the response body, and without it a 403 is unactionable.
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace").strip()[:300]
        except Exception:  # noqa: BLE001 — the body is a nicety, never required
            pass
        log.warning("webhook delivery failed: %s%s", exc, f" — {detail}" if detail else "")
        return False
    except Exception as exc:  # noqa: BLE001 — delivery failure must not fail the run
        log.warning("webhook delivery failed: %s", exc)
        return False


def should_notify(result, policy: str) -> bool:
    if policy == "never":
        return False
    pub = getattr(result, "publication", None)
    if pub is not None and pub.suppressed and policy == "high_confidence":
        # Nothing was published, so there is no high-confidence idea to send.
        return False
    if policy == "high_confidence":
        return result.confidence >= 60 and bool(
            (result.ideas or {}).get("calls") or (result.ideas or {}).get("puts")
        )
    return True
