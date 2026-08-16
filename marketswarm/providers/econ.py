"""Economic calendar, rates and macro series.

Sources, in order of preference:
  1. FRED (free API key, optional) — authoritative series values
  2. U.S. Treasury par-yield XML — no key, no rate limit worth worrying about
  3. A built-in release schedule — BLS/BEA/Census/Fed release dates follow
     stable rules (e.g. CPI ~mid-month 08:30 ET, FOMC on published dates), so
     the agent knows what is due today even with zero connectivity.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

from .base import DataClient, ProviderError

log = logging.getLogger("marketswarm.econ")

FRED = "https://api.stlouisfed.org/fred/series/observations"
TREASURY_YIELDS = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
)

# Recurring U.S. releases that reliably move index futures at 08:30/10:00 ET.
# (name, ET time, impact, rule) — rule is evaluated against the date.
RECURRING_RELEASES: list[tuple[str, str, str, str]] = [
    ("CPI (Consumer Price Index)", "08:30", "very high", "monthly_mid"),
    ("PPI (Producer Price Index)", "08:30", "high", "monthly_mid"),
    ("Nonfarm Payrolls / Unemployment", "08:30", "very high", "first_friday"),
    ("Initial Jobless Claims", "08:30", "medium", "thursday"),
    ("Retail Sales", "08:30", "high", "monthly_mid"),
    ("PCE Price Index", "08:30", "very high", "monthly_end"),
    ("ISM Manufacturing PMI", "10:00", "high", "first_business_day"),
    ("ISM Services PMI", "10:00", "high", "third_business_day"),
    ("Consumer Confidence", "10:00", "medium", "last_tuesday"),
    ("Univ. of Michigan Sentiment", "10:00", "medium", "second_friday"),
    ("EIA Crude Inventories", "10:30", "medium", "wednesday"),
    ("GDP (advance/second/third)", "08:30", "high", "monthly_end"),
]

# FOMC decision dates must be maintained manually — they are set by the Fed and
# not derivable from a rule. Update annually from federalreserve.gov.
FOMC_DATES: set[dt.date] = {
    dt.date(2026, 1, 28), dt.date(2026, 3, 18), dt.date(2026, 4, 29),
    dt.date(2026, 6, 17), dt.date(2026, 7, 29), dt.date(2026, 9, 16),
    dt.date(2026, 11, 4), dt.date(2026, 12, 16),
}


@dataclass
class EconEvent:
    name: str
    date: dt.date
    time_et: str
    impact: str                    # very high | high | medium | low
    actual: str | None = None
    forecast: str | None = None
    previous: str | None = None
    source: str = "release_schedule"
    notes: list[str] = field(default_factory=list)

    @property
    def before_open(self) -> bool:
        try:
            h, m = (int(x) for x in self.time_et.split(":"))
        except ValueError:
            return False
        return (h, m) < (9, 30)

    def describe(self) -> str:
        when = "pre-open" if self.before_open else "intraday"
        return f"{self.name} — {self.time_et} ET ({when}), impact {self.impact}"


def _nth_business_day(year: int, month: int, n: int) -> dt.date:
    d, count = dt.date(year, month, 1), 0
    while True:
        if d.weekday() < 5:
            count += 1
            if count == n:
                return d
        d += dt.timedelta(days=1)


def scheduled_events(day: dt.date) -> list[EconEvent]:
    """Which recurring releases land on this date."""
    out: list[EconEvent] = []

    if day in FOMC_DATES:
        out.append(EconEvent("FOMC rate decision + statement", day, "14:00", "very high",
                             notes=["Chair press conference 14:30 ET",
                                    "Expect compressed pre-14:00 ranges and an expansion after"]))

    for name, time_et, impact, rule in RECURRING_RELEASES:
        hit = False
        if rule == "monthly_mid":
            hit = 10 <= day.day <= 15 and day.weekday() < 5
        elif rule == "monthly_end":
            hit = day.day >= 26 and day.weekday() < 5
        elif rule == "first_friday":
            hit = day.weekday() == 4 and day.day <= 7
        elif rule == "second_friday":
            hit = day.weekday() == 4 and 8 <= day.day <= 14
        elif rule == "thursday":
            hit = day.weekday() == 3
        elif rule == "wednesday":
            hit = day.weekday() == 2
        elif rule == "first_business_day":
            hit = day == _nth_business_day(day.year, day.month, 1)
        elif rule == "third_business_day":
            hit = day == _nth_business_day(day.year, day.month, 3)
        elif rule == "last_tuesday":
            hit = day.weekday() == 1 and (day + dt.timedelta(days=7)).month != day.month
        if hit:
            out.append(EconEvent(name, day, time_et, impact))

    out.sort(key=lambda e: (e.time_et, {"very high": 0, "high": 1, "medium": 2, "low": 3}[e.impact]))
    return out


class EconData:
    def __init__(self, client: DataClient, fred_api_key: str | None = None):
        self.client = client
        self.fred_key = fred_api_key

    async def fred_series(self, series_id: str, limit: int = 12) -> list[dict] | None:
        if not self.fred_key:
            return None
        try:
            payload = await self.client.get_json(
                FRED,
                params={
                    "series_id": series_id,
                    "api_key": self.fred_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": limit,
                },
            )
            return payload.get("observations", [])
        except ProviderError as exc:
            log.warning("FRED %s unavailable: %s", series_id, exc)
            return None

    async def macro_dashboard(self) -> dict:
        """Headline macro levels the report cites when framing the tape."""
        series = {
            "DFF": "Fed funds effective rate",
            "DGS10": "10-year Treasury",
            "DGS2": "2-year Treasury",
            "T10Y2Y": "10y-2y spread",
            "CPIAUCSL": "CPI (index)",
            "UNRATE": "Unemployment rate",
        }
        if not self.fred_key:
            return {"available": False,
                    "note": "FRED_API_KEY not set — macro levels sourced from market quotes only"}

        results = await self.client.gather(
            [self.fred_series(sid, 2) for sid in series], label="fred"
        )
        out: dict = {"available": True, "series": {}}
        for (sid, label), obs in zip(series.items(), results):
            if not obs:
                continue
            try:
                latest = obs[0]
                prev = obs[1] if len(obs) > 1 else None
                out["series"][sid] = {
                    "label": label,
                    "value": float(latest["value"]) if latest["value"] != "." else None,
                    "date": latest["date"],
                    "previous": float(prev["value"]) if prev and prev["value"] != "." else None,
                }
            except (KeyError, ValueError, TypeError):
                continue
        return out

    async def todays_calendar(self, day: dt.date) -> list[EconEvent]:
        """Scheduled releases, enriched with FRED actuals where available."""
        events = scheduled_events(day)
        if not self.fred_key:
            return events

        fred_map = {"CPI (Consumer Price Index)": "CPIAUCSL",
                    "Nonfarm Payrolls / Unemployment": "PAYEMS",
                    "PPI (Producer Price Index)": "PPIACO",
                    "Retail Sales": "RSAFS"}
        for ev in events:
            sid = fred_map.get(ev.name)
            if not sid:
                continue
            obs = await self.fred_series(sid, 2)
            if obs and len(obs) >= 2:
                try:
                    ev.previous = obs[1]["value"]
                    ev.notes.append(f"prior print {obs[1]['value']} ({obs[1]['date']}) via FRED")
                    ev.source = "fred"
                except (KeyError, TypeError):
                    pass
        return events

    async def treasury_curve(self) -> dict | None:
        """Par yield curve — the cleanest read on the rates backdrop."""
        year = dt.date.today().year
        try:
            raw = await self.client.get_text(
                TREASURY_YIELDS,
                params={"data": "daily_treasury_yield_curve", "field_tdr_date_value": str(year)},
            )
        except ProviderError as exc:
            log.warning("treasury curve unavailable: %s", exc)
            return None

        import re

        entries = re.findall(r"<d:BC_(\w+)>([-\d.]+)</d:BC_\w+>", raw)
        if not entries:
            return None
        tail = entries[-12:]
        return {"tenors": {k: float(v) for k, v in tail}, "source": "U.S. Treasury par yield curve"}
