import datetime as dt

from marketswarm import clock


def test_known_2025_holidays():
    h = clock.holidays(2025)
    assert h[dt.date(2025, 1, 1)] == "New Year's Day"
    assert h[dt.date(2025, 1, 20)] == "Martin Luther King Jr. Day"
    assert h[dt.date(2025, 4, 18)] == "Good Friday"
    assert h[dt.date(2025, 5, 26)] == "Memorial Day"
    assert h[dt.date(2025, 6, 19)] == "Juneteenth National Independence Day"
    assert h[dt.date(2025, 7, 4)] == "Independence Day"
    assert h[dt.date(2025, 9, 1)] == "Labor Day"
    assert h[dt.date(2025, 11, 27)] == "Thanksgiving Day"
    assert h[dt.date(2025, 12, 25)] == "Christmas Day"
    assert dt.date(2025, 1, 9) in h  # Carter day of mourning


def test_known_2026_holidays():
    h = clock.holidays(2026)
    # July 4 2026 is a Saturday, so the NYSE observes it on Friday July 3.
    assert dt.date(2026, 7, 3) in h
    assert dt.date(2026, 7, 4) not in h
    assert h[dt.date(2026, 4, 3)] == "Good Friday"
    assert h[dt.date(2026, 11, 26)] == "Thanksgiving Day"


def test_easter_dates():
    assert clock._easter(2024) == dt.date(2024, 3, 31)
    assert clock._easter(2025) == dt.date(2025, 4, 20)
    assert clock._easter(2026) == dt.date(2026, 4, 5)
    assert clock._easter(2027) == dt.date(2027, 3, 28)


def test_juneteenth_only_from_2022():
    assert dt.date(2021, 6, 18) not in clock.holidays(2021)
    assert dt.date(2022, 6, 20) in clock.holidays(2022)  # 19th was a Sunday


def test_early_closes():
    ec = clock.early_closes(2025)
    assert dt.date(2025, 7, 3) in ec
    assert dt.date(2025, 11, 28) in ec
    assert dt.date(2025, 12, 24) in ec
    # July 3 2026 is a full holiday, so it must not also be an early close.
    assert dt.date(2026, 7, 3) not in clock.early_closes(2026)


def test_trading_day_helpers():
    assert not clock.is_trading_day(dt.date(2025, 12, 25))
    assert not clock.is_trading_day(dt.date(2025, 12, 27))  # Saturday
    assert clock.is_trading_day(dt.date(2025, 12, 26))

    # Friday before Christmas 2025 (Thu) -> previous trading day of the 26th
    assert clock.previous_trading_day(dt.date(2025, 12, 26)) == dt.date(2025, 12, 24)
    assert clock.next_trading_day(dt.date(2025, 12, 24)) == dt.date(2025, 12, 26)


def test_day_status_early_close():
    s = clock.day_status(dt.date(2025, 11, 28))
    assert s.is_trading_day and s.early_close
    assert s.close_time == clock.EARLY_CLOSE
    assert "early close" in s.summary


def test_sessions():
    et = clock.ET
    day = dt.date(2025, 6, 10)  # a normal Tuesday
    assert clock.session_at(dt.datetime.combine(day, dt.time(3, 0), tzinfo=et)) == clock.Session.OVERNIGHT
    assert clock.session_at(dt.datetime.combine(day, dt.time(7, 0), tzinfo=et)) == clock.Session.PREMARKET
    assert clock.session_at(dt.datetime.combine(day, dt.time(11, 0), tzinfo=et)) == clock.Session.REGULAR
    assert clock.session_at(dt.datetime.combine(day, dt.time(17, 0), tzinfo=et)) == clock.Session.AFTERHOURS
    holiday = dt.datetime.combine(dt.date(2025, 12, 25), dt.time(11, 0), tzinfo=et)
    assert clock.session_at(holiday) == clock.Session.CLOSED_HOLIDAY


def test_minutes_to_open():
    et = clock.ET
    pre = dt.datetime.combine(dt.date(2025, 6, 10), dt.time(8, 30), tzinfo=et)
    assert 59 < clock.minutes_to_open(pre) < 61


def test_trading_days_between_excludes_closures():
    days = clock.trading_days_between(dt.date(2025, 12, 22), dt.date(2025, 12, 28))
    assert dt.date(2025, 12, 25) not in days
    assert dt.date(2025, 12, 26) in days
    assert len(days) == 4
