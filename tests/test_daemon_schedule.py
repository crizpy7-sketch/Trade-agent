"""Daemon scheduling: when a slot fires, and when it is too late to fire.

Written after a live incident. The daemon keeps its "already ran today" set in
memory, and fired any slot whose time had merely passed. So restarting the
service at 09:52 ET re-ran the 08:15 pre-market pass and published a report
headed "Pre-Market" twenty-two minutes after the open — with the whole
architecture behaving correctly underneath a header that was a lie.

Being late is not a smaller version of being on time. A pre-market pass that
runs mid-session is wrong, not delayed, so a missed slot is now recorded as
missed and waits for tomorrow.
"""

from __future__ import annotations

import datetime as dt

import pytest

from marketswarm.cli import GRACE_MINUTES, _is_due, _is_missed, _parse_hhmm

RUN_AT = dt.time(8, 15)


def at(h: int, m: int) -> dt.time:
    return dt.time(h, m)


# --------------------------------------------------------------------------
# due / missed are exclusive and total: every moment is exactly one of
# not-yet, due, or missed.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("now,due,missed", [
    (at(7, 0),  False, False),   # long before
    (at(8, 14), False, False),   # a minute early is not due
    (at(8, 15), True,  False),   # exactly on time
    (at(8, 16), True,  False),   # just after
    (at(8, 59), True,  False),   # inside the grace window
    (at(9, 0),  True,  False),   # last minute of the window
    (at(9, 1),  False, True),    # one minute past — missed
    (at(9, 52), False, True),    # the restart that caused the incident
    (at(15, 0), False, True),    # hours later
])
def test_slot_state_by_clock(now, due, missed):
    assert _is_due(now, RUN_AT) is due
    assert _is_missed(now, RUN_AT) is missed


def test_a_slot_is_never_both_due_and_missed():
    for hour in range(24):
        for minute in (0, 14, 15, 16, 30, 45, 59):
            now = at(hour, minute)
            assert not (_is_due(now, RUN_AT) and _is_missed(now, RUN_AT)), (
                f"{now} is both due and missed")


def _offset(base: dt.time, minutes: int) -> dt.time:
    """Wall-clock arithmetic — 08:15 + 45 is 09:00, not 08:60."""
    return (dt.datetime.combine(dt.date(2026, 1, 1), base)
            + dt.timedelta(minutes=minutes)).time()


def test_the_grace_window_is_the_documented_width():
    assert _is_due(_offset(RUN_AT, GRACE_MINUTES), RUN_AT), \
        "the last minute of the window must still fire"
    assert _is_missed(_offset(RUN_AT, GRACE_MINUTES + 1), RUN_AT), \
        "one minute past the window must not fire"


def test_the_window_closes_before_the_market_opens():
    """The point of the whole change: a pre-market pass cannot run in-session."""
    open_et = dt.time(9, 30)
    assert _is_missed(open_et, RUN_AT), (
        f"a {GRACE_MINUTES}-minute grace window from {RUN_AT} still allows a "
        f"'pre-market' run at {open_et} ET, which is after the open")


# --------------------------------------------------------------------------
# the incident, restated as a test
# --------------------------------------------------------------------------

def test_a_restart_after_the_window_does_not_refire_the_run():
    """Restarting the daemon mid-morning must not trigger a catch-up run.

    `ran` is in-memory, so a fresh process has no record of the morning. The
    only thing standing between a restart and a spurious run is the window.
    """
    restart = at(9, 52)   # exactly when the real restart happened
    ran: set = set()      # fresh process — remembers nothing

    due = _is_due(restart, RUN_AT) and "run" not in ran
    assert not due, "a restart at 09:52 ET would re-run the 08:15 pre-market pass"
    assert _is_missed(restart, RUN_AT), "the slot should be recorded as missed"


def test_a_restart_inside_the_window_still_runs_the_day():
    """The window must not be so tight that a normal restart loses the day."""
    restart = at(8, 20)
    assert _is_due(restart, RUN_AT), (
        "a restart five minutes after the scheduled time should still run — "
        "otherwise a slow boot silently costs a trading day")


# --------------------------------------------------------------------------
# time parsing, which decides everything above
# --------------------------------------------------------------------------

def test_configured_times_parse():
    assert _parse_hhmm("08:15") == dt.time(8, 15)
    assert _parse_hhmm("16:45") == dt.time(16, 45)
    assert _parse_hhmm("00:00") == dt.time(0, 0)


def test_a_bad_time_falls_back_rather_than_crashing_the_daemon():
    # A typo in config must not take the service down; it degrades to the
    # documented default and logs.
    for bad in ("", "not a time", "25:00", None, "8"):
        assert _parse_hhmm(bad) == dt.time(8, 15)
