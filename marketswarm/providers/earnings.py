"""Earnings calendar and post-report reaction analysis.

Two questions matter pre-market:
  1. Who reported after yesterday's close or before today's open? Those names
     have a live catalyst and the widest ranges of the session.
  2. Who reports after today's close? Those names see IV inflate and price
     compress into the event — a very different setup from a fresh reaction.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

from .base import DataClient, ProviderError

log = logging.getLogger("marketswarm.earnings")

NASDAQ_CAL = "https://api.nasdaq.com/api/calendar/earnings"
NASDAQ_HEADERS = {
    "Accept": "application/json",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}


@dataclass
class EarningsEvent:
    ticker: str
    company: str
    date: dt.date
    timing: str                    # "bmo" | "amc" | "unknown"
    eps_estimate: float | None = None
    eps_actual: float | None = None
    surprise_pct: float | None = None
    market_cap: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_fresh_catalyst(self) -> bool:
        """Already reported and trading on the reaction."""
        return self.eps_actual is not None

    def describe(self) -> str:
        when = {"bmo": "before open", "amc": "after close", "unknown": "timing unconfirmed"}[self.timing]
        base = f"{self.ticker} ({self.company}) reports {when} {self.date:%d %b}"
        if self.surprise_pct is not None:
            base += f" — EPS surprise {self.surprise_pct:+.1f}%"
        return base


def _f(val) -> float | None:
    if val in (None, "", "N/A"):
        return None
    try:
        return float(str(val).replace("$", "").replace(",", "").replace("(", "-").replace(")", ""))
    except ValueError:
        return None


class EarningsData:
    def __init__(self, client: DataClient):
        self.client = client

    async def calendar(self, day: dt.date) -> list[EarningsEvent]:
        try:
            payload = await self.client.get_json(
                NASDAQ_CAL, params={"date": day.isoformat()}, headers=NASDAQ_HEADERS
            )
            rows = (payload.get("data") or {}).get("rows") or []
        except (ProviderError, AttributeError, TypeError) as exc:
            log.warning("earnings calendar %s unavailable: %s", day, exc)
            return []

        out: list[EarningsEvent] = []
        for r in rows:
            timing_raw = (r.get("time") or "").lower()
            timing = "bmo" if "pre-market" in timing_raw or "before" in timing_raw else \
                     "amc" if "after" in timing_raw else "unknown"
            est, act = _f(r.get("epsForecast")), _f(r.get("eps"))
            surprise = None
            if est not in (None, 0) and act is not None:
                surprise = (act - est) / abs(est) * 100
            out.append(
                EarningsEvent(
                    ticker=(r.get("symbol") or "").upper(),
                    company=r.get("name") or "",
                    date=day,
                    timing=timing,
                    eps_estimate=est,
                    eps_actual=act,
                    surprise_pct=surprise,
                    market_cap=_f(r.get("marketCap")),
                )
            )
        return out

    async def relevant_window(self, today: dt.date, prev_session: dt.date, min_market_cap: float = 5e9) -> dict:
        """Everything with a live earnings catalyst for today's session."""
        results = await self.client.gather(
            [self.calendar(prev_session), self.calendar(today)], label="earnings"
        )
        yesterday_amc = [
            e for e in (results[0] or []) if e.timing == "amc" and (e.market_cap or 0) >= min_market_cap
        ]
        today_bmo = [
            e for e in (results[1] or []) if e.timing == "bmo" and (e.market_cap or 0) >= min_market_cap
        ]
        today_amc = [
            e for e in (results[1] or []) if e.timing == "amc" and (e.market_cap or 0) >= min_market_cap
        ]

        for group in (yesterday_amc, today_bmo, today_amc):
            group.sort(key=lambda e: -(e.market_cap or 0))

        return {
            "reacting_today": yesterday_amc + today_bmo,   # gapping on results right now
            "reporting_tonight": today_amc,                # IV inflating into the event
            "counts": {
                "prior_amc": len(yesterday_amc),
                "today_bmo": len(today_bmo),
                "today_amc": len(today_amc),
            },
        }


def reaction_quality(surprise_pct: float | None, gap_pct: float | None) -> dict:
    """Compare the gap to the surprise.

    The informative case is divergence: a beat that gaps down means guidance or
    positioning dominated the headline number, and fading the initial move has
    historically been the higher-probability read. Agreement between surprise
    and gap favours continuation.
    """
    if surprise_pct is None or gap_pct is None:
        return {"read": "insufficient data", "bias": "neutral", "confidence": 0.3}

    beat, gapped_up = surprise_pct > 0, gap_pct > 0
    magnitude = abs(gap_pct)

    if beat and gapped_up:
        return {
            "read": f"beat by {surprise_pct:+.1f}% and gapping {gap_pct:+.1f}% — reaction confirms the print",
            "bias": "bullish continuation" if magnitude > 2 else "mild bullish",
            "confidence": 0.62 if magnitude > 2 else 0.55,
        }
    if beat and not gapped_up:
        return {
            "read": f"beat by {surprise_pct:+.1f}% yet gapping {gap_pct:+.1f}% — guidance or positioning "
                    f"outweighed the headline",
            "bias": "bearish — sell-the-news",
            "confidence": 0.60,
        }
    if not beat and gapped_up:
        return {
            "read": f"missed by {surprise_pct:+.1f}% yet gapping {gap_pct:+.1f}% — expectations were already "
                    f"washed out or guidance was raised",
            "bias": "bullish — capitulation reversal",
            "confidence": 0.55,
        }
    return {
        "read": f"missed by {surprise_pct:+.1f}% and gapping {gap_pct:+.1f}% — reaction confirms the print",
        "bias": "bearish continuation" if magnitude > 2 else "mild bearish",
        "confidence": 0.62 if magnitude > 2 else 0.55,
    }


def gap_persistence_prior(gap_pct: float, had_earnings: bool) -> float:
    """Prior P(close beyond the open, in the gap's direction).

    Anchored on the well-documented behaviour of U.S. equity gaps: small gaps
    fill more often than not, large earnings-driven gaps trend, and the
    transition sits around 2-3%. Deliberately conservative — these are priors
    the evidence then updates, not conclusions.
    """
    a = abs(gap_pct)
    if had_earnings:
        if a >= 5:
            return 0.63       # large earnings gaps persist
        if a >= 2:
            return 0.57
        return 0.52
    if a >= 3:
        return 0.55
    if a >= 1:
        return 0.50
    return 0.47               # small non-event gaps mildly favour the fill
