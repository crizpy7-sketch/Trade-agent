"""Option chains from Polygon, and the seam that makes the provider swappable.

The swarm lost its options data twice in a fortnight to one undocumented
endpoint. What is protected here is not Polygon specifically — it is that the
provider can be changed, that a broken one degrades instead of crashing the
run, and that whatever comes back is checked before it is believed.

These do NOT prove the wire format is right. Polygon's shapes are written from
its published API and have never met the live service; the fake below encodes
my reading of the documentation, so a test passing against it proves the parser
matches my reading and nothing more. The first live run is the real test, which
is why diagnose() prints what actually arrived.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from marketswarm.providers.chains import PolygonChains, select_source

TODAY = dt.date.today()
NEAR = TODAY + dt.timedelta(days=3)


class FakePolygon:
    """Answers the two endpoints, and records what it was asked."""

    def __init__(self, *, expirations=True, snapshot=True, pages=1, fail=None):
        self.expirations, self.snapshot, self.pages = expirations, snapshot, pages
        self.fail = fail
        self.calls: list[tuple[str, dict]] = []

    async def get_json(self, url, params=None, use_cache=True, headers=None):
        self.calls.append((url, dict(params or {})))
        if self.fail:
            raise self.fail
        if "reference/options/contracts" in url:
            if not self.expirations:
                return {"results": []}
            return {"results": [
                {"expiration_date": NEAR.isoformat(), "strike_price": 100},
                {"expiration_date": NEAR.isoformat(), "strike_price": 105},
                {"expiration_date": (TODAY + dt.timedelta(days=10)).isoformat()},
                {"expiration_date": (TODAY - dt.timedelta(days=5)).isoformat()},  # past
                {"expiration_date": "not-a-date"},                                # junk
            ]}
        if not self.snapshot:
            return {"results": []}
        page = len([c for c in self.calls if "snapshot" in c[0]])
        results = [
            _contract("call", 100.0, bid=2.00, ask=2.10, volume=900, oi=4000),
            _contract("put", 100.0, bid=1.80, ask=1.90, volume=700, oi=3500),
            {"details": {"contract_type": "junk", "strike_price": 1}},   # skipped
            {"details": {"contract_type": "call", "strike_price": 0}},   # skipped
        ]
        out = {"results": results}
        if page < self.pages:
            out["next_url"] = f"https://api.polygon.io/next?cursor=p{page}"
        return out


def _contract(kind, strike, *, bid, ask, volume, oi, underlying=101.5):
    return {
        "details": {"contract_type": kind, "strike_price": strike,
                    "expiration_date": NEAR.isoformat()},
        "last_quote": {"bid": bid, "ask": ask, "midpoint": (bid + ask) / 2},
        "day": {"close": (bid + ask) / 2, "volume": volume},
        "open_interest": oi,
        "implied_volatility": 0.24,
        "underlying_asset": {"price": underlying, "ticker": "TEST"},
    }


def src(fake, key="k"):
    return PolygonChains(fake, key)


# ------------------------------------------------------------- expirations

def test_expirations_are_upcoming_sorted_and_deduplicated():
    got = asyncio.run(src(FakePolygon()).expirations("SPY"))
    assert got == sorted(got)
    assert len(got) == len(set(got)), "duplicate expiries would fetch twice"
    assert all(d >= TODAY for d in got), "a past expiry is not tradable"


def test_an_unparseable_date_is_skipped_not_fatal():
    """One malformed row must not cost the whole chain."""
    assert asyncio.run(src(FakePolygon()).expirations("SPY"))


def test_a_provider_error_degrades_to_empty_rather_than_raising():
    """One dead provider must never take the morning run down."""
    fake = FakePolygon(fail=RuntimeError("401 Unauthorized"))
    s = src(fake)
    assert asyncio.run(s.expirations("SPY")) == []
    assert "401" in (s.last_error or ""), "the cause was not kept for diagnosis"


# ------------------------------------------------------------------ chains

def test_a_chain_carries_the_fields_the_board_needs():
    c = asyncio.run(src(FakePolygon()).chain("SPY", NEAR))

    assert c is not None
    assert c.underlying_price == pytest.approx(101.5)
    assert c.expiration == NEAR
    call = c.calls[0]
    assert call.strike == 100.0
    assert call.bid == pytest.approx(2.00) and call.ask == pytest.approx(2.10)
    assert call.mid == pytest.approx(2.05)
    assert call.open_interest == 4000 and call.volume == 900
    assert call.implied_volatility == pytest.approx(0.24)


def test_moneyness_is_computed_per_side():
    """in_the_money is derived, not taken on trust — the sides are opposite."""
    c = asyncio.run(src(FakePolygon()).chain("SPY", NEAR))
    assert c.calls[0].in_the_money is True, "a 100 call under a 101.5 spot is ITM"
    assert c.puts[0].in_the_money is False


def test_contracts_that_cannot_be_parsed_are_dropped_not_faked():
    c = asyncio.run(src(FakePolygon()).chain("SPY", NEAR))
    assert len(c.calls) == 1 and len(c.puts) == 1
    assert all(x.strike > 0 for x in c.calls + c.puts)


def test_an_empty_snapshot_returns_no_chain_rather_than_an_empty_one():
    """An empty Chain would read downstream as a real but quiet market."""
    assert asyncio.run(src(FakePolygon(snapshot=False)).chain("SPY", NEAR)) is None


def test_pagination_is_followed_but_bounded():
    """A paginating loop against a rate-limited tier eats the whole quota."""
    from marketswarm.providers.chains import POLYGON_MAX_PAGES
    fake = FakePolygon(pages=99)
    asyncio.run(src(fake).chain("SPY", NEAR))
    snaps = [c for c in fake.calls if "snapshot" in c[0] or "next" in c[0]]
    assert len(snaps) <= POLYGON_MAX_PAGES


def test_the_api_key_travels_on_every_request_including_paged_ones():
    fake = FakePolygon(pages=2)
    asyncio.run(src(fake, key="secret-key").chain("SPY", NEAR))
    for url, params in fake.calls:
        assert params.get("apiKey") == "secret-key", f"key missing on {url}"


# ------------------------------------------------------------- the seam

def test_auto_prefers_a_configured_key_over_the_endpoint_that_failed():
    assert select_source(None, provider="auto", polygon_key=None) is None
    assert select_source(None, provider="auto", polygon_key="k").name == "polygon"


def test_yahoo_can_be_forced_back_so_a_bad_switch_needs_no_deploy():
    assert select_source(None, provider="yahoo", polygon_key="k") is None


def test_naming_polygon_without_a_key_falls_back_rather_than_failing_closed():
    """A misconfiguration must degrade to the old path, not to no options."""
    assert select_source(None, provider="polygon", polygon_key=None) is None


def test_an_unknown_provider_name_does_not_silently_disable_options():
    assert select_source(None, provider="nonsense", polygon_key="k").name == "polygon"


def test_diagnose_reports_what_came_back():
    assert "expirations" in asyncio.run(src(FakePolygon()).diagnose("SPY"))
    assert "no expirations" in asyncio.run(src(FakePolygon(expirations=False)).diagnose("SPY"))
    assert "no usable chain" in asyncio.run(src(FakePolygon(snapshot=False)).diagnose("SPY"))
