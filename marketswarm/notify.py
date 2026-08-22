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
