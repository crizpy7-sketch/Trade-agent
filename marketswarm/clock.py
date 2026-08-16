"""US equity market calendar and session clock.

Self-contained: NYSE/Nasdaq holiday rules are computed from first principles so
the agent can decide "is the market open today?" with no network and no
third-party calendar package. Rules implemented per NYSE Rule 7.2.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

REGULAR_OPEN = dt.time(9, 30)
REGULAR_CLOSE = dt.time(16, 0)
EARLY_CLOSE = dt.time(13, 0)
PREMARKET_OPEN = dt.time(4, 0)
AFTERHOURS_CLOSE = dt.time(20, 0)


class Session(str, Enum):
    CLOSED_HOLIDAY = "closed_holiday"
    CLOSED_WEEKEND = "closed_weekend"
    OVERNIGHT = "overnight"          # 20:00 -> 04:00 ET
    PREMARKET = "premarket"          # 04:00 -> 09:30 ET
    REGULAR = "regular"
    AFTERHOURS = "afterhours"
    POST_CLOSE = "post_close"        # early-close days after 13:00


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """n-th `weekday` (Mon=0) of a month. n=-1 means last."""
    if n > 0:
        d = dt.date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + dt.timedelta(days=offset + 7 * (n - 1))
    nxt = dt.date(year + (month == 12), (month % 12) + 1, 1)
    d = nxt - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year: int) -> dt.date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def _observed(d: dt.date) -> dt.date:
    """Saturday holidays observe Friday, Sunday holidays observe Monday."""
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def holidays(year: int) -> dict[dt.date, str]:
    """Full-day NYSE closures for a calendar year."""
    good_friday = _easter(year) - dt.timedelta(days=2)
    out: dict[dt.date, str] = {
        _observed(dt.date(year, 1, 1)): "New Year's Day",
        _nth_weekday(year, 1, 0, 3): "Martin Luther King Jr. Day",
        _nth_weekday(year, 2, 0, 3): "Washington's Birthday",
        good_friday: "Good Friday",
        _nth_weekday(year, 5, 0, -1): "Memorial Day",
        _observed(dt.date(year, 7, 4)): "Independence Day",
        _nth_weekday(year, 9, 0, 1): "Labor Day",
        _nth_weekday(year, 11, 3, 4): "Thanksgiving Day",
        _observed(dt.date(year, 12, 25)): "Christmas Day",
    }
    if year >= 2022:  # Juneteenth became an NYSE holiday in 2022
        out[_observed(dt.date(year, 6, 19))] = "Juneteenth National Independence Day"
    # A New Year's Day falling on Saturday is NOT observed on Dec 31 by the NYSE.
    if dt.date(year, 1, 1).weekday() == 5:
        out.pop(dt.date(year - 1, 12, 31), None)
        out.pop(dt.date(year, 1, 1) - dt.timedelta(days=1), None)
    # Ad-hoc closures (national days of mourning etc.) that rules cannot derive.
    out.update({d: n for d, n in AD_HOC_CLOSURES.items() if d.year == year})
    return out


# Closures proclaimed outside the standing rules. Append as they are announced.
AD_HOC_CLOSURES: dict[dt.date, str] = {
    dt.date(2025, 1, 9): "National Day of Mourning (President Carter)",
    dt.date(2018, 12, 5): "National Day of Mourning (President G.H.W. Bush)",
}


def early_closes(year: int) -> dict[dt.date, str]:
    """1:00 p.m. ET closes: July 3rd, day after Thanksgiving, Christmas Eve."""
    out: dict[dt.date, str] = {}
    hol = holidays(year)

    july3 = dt.date(year, 7, 3)
    if july3.weekday() < 5 and july3 not in hol:
        out[july3] = "Day before Independence Day"

    black_friday = _nth_weekday(year, 11, 3, 4) + dt.timedelta(days=1)
    out[black_friday] = "Day after Thanksgiving"

    xmas_eve = dt.date(year, 12, 24)
    if xmas_eve.weekday() < 5 and xmas_eve not in hol:
        out[xmas_eve] = "Christmas Eve"
    return out


@dataclass(frozen=True)
class DayStatus:
    date: dt.date
    is_trading_day: bool
    reason: str | None            # holiday/weekend name when closed
    close_time: dt.time
    early_close: bool

    @property
    def summary(self) -> str:
        if not self.is_trading_day:
            return f"{self.date:%A %d %B %Y} — U.S. equity markets CLOSED ({self.reason})"
        tail = " (early close 13:00 ET)" if self.early_close else ""
        return f"{self.date:%A %d %B %Y} — regular session 09:30–{self.close_time:%H:%M} ET{tail}"


def day_status(d: dt.date) -> DayStatus:
    if d.weekday() >= 5:
        return DayStatus(d, False, "weekend", REGULAR_CLOSE, False)
    hol = holidays(d.year)
    if d in hol:
        return DayStatus(d, False, hol[d], REGULAR_CLOSE, False)
    ec = early_closes(d.year)
    if d in ec:
        return DayStatus(d, True, None, EARLY_CLOSE, True)
    return DayStatus(d, True, None, REGULAR_CLOSE, False)


def is_trading_day(d: dt.date) -> bool:
    return day_status(d).is_trading_day


def next_trading_day(d: dt.date) -> dt.date:
    nxt = d + dt.timedelta(days=1)
    while not is_trading_day(nxt):
        nxt += dt.timedelta(days=1)
    return nxt


def previous_trading_day(d: dt.date) -> dt.date:
    prev = d - dt.timedelta(days=1)
    while not is_trading_day(prev):
        prev -= dt.timedelta(days=1)
    return prev


def trading_days_between(start: dt.date, end: dt.date) -> list[dt.date]:
    out, cur = [], start
    while cur <= end:
        if is_trading_day(cur):
            out.append(cur)
        cur += dt.timedelta(days=1)
    return out


def now_et() -> dt.datetime:
    return dt.datetime.now(tz=ET)


def session_at(ts: dt.datetime | None = None) -> Session:
    ts = (ts or now_et()).astimezone(ET)
    status = day_status(ts.date())
    if not status.is_trading_day:
        return Session.CLOSED_WEEKEND if status.reason == "weekend" else Session.CLOSED_HOLIDAY
    t = ts.time()
    if t < PREMARKET_OPEN:
        return Session.OVERNIGHT
    if t < REGULAR_OPEN:
        return Session.PREMARKET
    if t < status.close_time:
        return Session.REGULAR
    if status.early_close and t >= status.close_time:
        return Session.POST_CLOSE if t >= AFTERHOURS_CLOSE else Session.AFTERHOURS
    if t < AFTERHOURS_CLOSE:
        return Session.AFTERHOURS
    return Session.OVERNIGHT


def minutes_to_open(ts: dt.datetime | None = None) -> float:
    """Minutes until the next regular open (negative once the session is live)."""
    ts = (ts or now_et()).astimezone(ET)
    today = day_status(ts.date())
    if today.is_trading_day:
        open_dt = dt.datetime.combine(ts.date(), REGULAR_OPEN, tzinfo=ET)
        if ts < open_dt:
            return (open_dt - ts).total_seconds() / 60
        if ts < dt.datetime.combine(ts.date(), today.close_time, tzinfo=ET):
            return (open_dt - ts).total_seconds() / 60
    nxt = next_trading_day(ts.date())
    return (dt.datetime.combine(nxt, REGULAR_OPEN, tzinfo=ET) - ts).total_seconds() / 60
