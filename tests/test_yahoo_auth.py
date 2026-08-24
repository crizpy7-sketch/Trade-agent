"""Yahoo's cookie/crumb handshake for the option-chain endpoint.

Yahoo's v7 options endpoint began requiring a session cookie plus a matching
crumb and answers 401 for every symbol without them. On the live VPS that meant
options flow was degraded on every run while the rest of the swarm looked
healthy — thirteen agents reporting, three of them blind.

IMPORTANT: these exercise the handshake against a local fake, not against
Yahoo. The sandbox this was written in has no route to finance.yahoo.com, so
the shape of the flow is verified here and the *live* behaviour has to be
confirmed on a machine that can reach Yahoo. A green suite here is necessary
and not sufficient.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading

import pytest

from marketswarm.providers import base as pb
from marketswarm.providers.base import DataClient, ProviderError, YahooSession

CHAIN_PAYLOAD = {"optionChain": {"result": [{"expirationDates": [1767225600],
                                             "quote": {"regularMarketPrice": 100.0},
                                             "options": [{"calls": [], "puts": []}]}]}}


class _Yahoo(http.server.BaseHTTPRequestHandler):
    """Stand-in for Yahoo. Refuses the chain until a valid crumb is presented."""

    crumb = "abc123CRUMB"
    require_browser_ua = True
    always_refuse_chain = False
    calls: list = []

    def _send(self, status, body: bytes, ctype="text/plain"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        ua = self.headers.get("User-Agent", "")
        type(self).calls.append((self.path.split("?")[0], ua))

        if self.path.startswith("/cookie"):
            self.send_response(404)          # Yahoo answers 404 and sets a cookie
            self.send_header("Set-Cookie", "A1=session-token; Path=/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if self.path.startswith("/getcrumb"):
            if type(self).require_browser_ua and "Mozilla" not in ua:
                return self._send(403, b"denied")
            return self._send(200, type(self).crumb.encode())

        if self.path.startswith("/options"):
            if type(self).always_refuse_chain:
                return self._send(401, b'{"error":"Unauthorized"}', "application/json")
            if f"crumb={type(self).crumb}" not in self.path:
                return self._send(401, b'{"error":"Unauthorized"}', "application/json")
            return self._send(200, json.dumps(CHAIN_PAYLOAD).encode(), "application/json")

        self._send(404, b"nope")

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_yahoo(monkeypatch, tmp_path):
    _Yahoo.calls = []
    _Yahoo.crumb = "abc123CRUMB"
    _Yahoo.require_browser_ua = True
    _Yahoo.always_refuse_chain = False

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Yahoo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    root = f"http://127.0.0.1:{srv.server_address[1]}"

    monkeypatch.setattr(pb, "YAHOO_COOKIE_URL", f"{root}/cookie")
    monkeypatch.setattr(pb, "YAHOO_CRUMB_URL", f"{root}/getcrumb")

    from marketswarm.providers import options as opt
    monkeypatch.setattr(opt, "CHAIN", root + "/options/{symbol}")

    yield root, srv, tmp_path
    srv.shutdown()


def _client(tmp_path):
    return DataClient(cache_dir=tmp_path / "cache", max_retries=2, rate_per_second=100)


# --------------------------------------------------------------------------

def test_the_handshake_acquires_a_crumb(fake_yahoo):
    _, _, tmp_path = fake_yahoo

    async def go():
        async with _client(tmp_path) as c:
            s = YahooSession(c)
            return await s.ensure()

    assert asyncio.run(go()) == "abc123CRUMB"
    paths = [p for p, _ in _Yahoo.calls]
    assert "/cookie" in paths, "the cookie endpoint was never called"
    assert "/getcrumb" in paths


def test_the_crumb_request_uses_a_browser_user_agent(fake_yahoo):
    """Yahoo 403s the agent's own User-Agent on this endpoint."""
    _, _, tmp_path = fake_yahoo

    async def go():
        async with _client(tmp_path) as c:
            return await YahooSession(c).ensure()

    assert asyncio.run(go()) == "abc123CRUMB"
    crumb_ua = [ua for p, ua in _Yahoo.calls if p == "/getcrumb"][0]
    assert "Mozilla" in crumb_ua


def test_an_html_interstitial_is_not_mistaken_for_a_crumb(fake_yahoo, monkeypatch):
    """A consent or block page returns 200 with a body. Treating that as a
    crumb would send garbage on every subsequent call."""
    _, _, tmp_path = fake_yahoo
    _Yahoo.crumb = "<html><body>consent required</body></html>"

    async def go():
        async with _client(tmp_path) as c:
            s = YahooSession(c)
            return await s.ensure(), s.failed

    crumb, failed = asyncio.run(go())
    assert crumb is None
    assert failed is True


def test_the_chain_fetch_succeeds_through_the_handshake(fake_yahoo):
    """End to end: the call that used to 401 on every symbol now returns."""
    from marketswarm.providers.options import OptionsData
    _, _, tmp_path = fake_yahoo

    async def go():
        async with _client(tmp_path) as c:
            return await OptionsData(c).expirations("SPY")

    assert asyncio.run(go()), "the chain fetch returned nothing — still refused"


def test_a_stale_crumb_is_refreshed_once_and_the_call_retried(fake_yahoo):
    """A crumb expires. One 401 means stale token, not outage."""
    from marketswarm.providers.options import OptionsData
    _, _, tmp_path = fake_yahoo

    async def go():
        async with _client(tmp_path) as c:
            data = OptionsData(c)
            await data.session.ensure()
            # Yahoo rotates the crumb behind us; the cached one is now stale.
            data.session._crumb = "STALE"
            return await data.expirations("SPY")

    assert asyncio.run(go()), "a stale crumb was not refreshed and retried"
    assert len([p for p, _ in _Yahoo.calls if p == "/getcrumb"]) >= 2


def test_a_persistent_refusal_degrades_instead_of_raising(fake_yahoo):
    """If Yahoo really is refusing, the agent degrades — the run still finishes."""
    from marketswarm.providers.options import OptionsData
    _, _, tmp_path = fake_yahoo
    # Refuses the chain no matter how good the crumb — a real outage, not a
    # stale token, so the refresh-and-retry must give up rather than loop.
    _Yahoo.always_refuse_chain = True

    async def go():
        async with _client(tmp_path) as c:
            return await OptionsData(c).expirations("SPY")

    assert asyncio.run(go()) == [], "a hard refusal should degrade to empty, not raise"
    chain_calls = len([p for p, _ in _Yahoo.calls if p.startswith("/options")])
    assert chain_calls == 2, (
        f"expected exactly one retry after refresh, saw {chain_calls} attempts")


# --------------------------------------------------------------------------
# the retry-policy change that came with this
# --------------------------------------------------------------------------

def test_a_4xx_is_not_retried(fake_yahoo):
    """A 401 is a statement about the request. Retrying it three times per
    symbol with backoff only made a broken run slower and noisier."""
    _, root_srv, tmp_path = fake_yahoo
    root = [p for p, _ in _Yahoo.calls]  # noqa: F841
    url = None

    async def go():
        nonlocal url
        async with _client(tmp_path) as c:
            url = pb.YAHOO_CRUMB_URL.replace("/getcrumb", "/options/SPY")
            with pytest.raises(ProviderError) as ei:
                await c.get_json(url, use_cache=False)
            return ei.value

    err = asyncio.run(go())
    assert err.status == 401
    attempts = len([p for p, _ in _Yahoo.calls if p == "/options/SPY"])
    assert attempts == 1, f"a 401 was retried {attempts} times"


def test_a_5xx_is_still_retried(tmp_path):
    """Transient server faults must keep their retries."""
    seen = {"n": 0}

    class Flaky(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen["n"] += 1
            body = b"{}" if seen["n"] > 1 else b"boom"
            code = 200 if seen["n"] > 1 else 503
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Flaky)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        async def go():
            async with DataClient(cache_dir=tmp_path / "c", max_retries=3,
                                  rate_per_second=100) as c:
                return await c.get_json(
                    f"http://127.0.0.1:{srv.server_address[1]}/x", use_cache=False)

        assert asyncio.run(go()) == {}
        assert seen["n"] == 2, "the 503 was not retried"
    finally:
        srv.shutdown()


# ===================================================== rate limiting (429)

class _Resp:
    def __init__(self, status, text="", headers=None):
        self.status_code, self.text = status, text
        self.headers = headers or {}


class _CountingHttp:
    """Answers 429 a set number of times, then succeeds."""

    def __init__(self, refusals: int, headers=None):
        self.refusals, self.calls = refusals, 0
        self.headers = headers or {}

    async def get(self, url, headers=None, timeout=None):
        if "getcrumb" not in url:
            return _Resp(404)                       # the cookie call, by design
        self.calls += 1
        if self.calls <= self.refusals:
            return _Resp(429, "Too Many Requests", self.headers)
        return _Resp(200, "aBcD1234")


#: Captured before any patching. base.asyncio IS the asyncio module, so a
#: replacement that calls asyncio.sleep calls itself.
_REAL_SLEEP = asyncio.sleep


def _no_wait(monkeypatch) -> list:
    """Record the delays the code asked for, without actually waiting."""
    from marketswarm.providers import base
    slept: list = []

    async def fake(seconds):
        slept.append(seconds)
        await _REAL_SLEEP(0)

    monkeypatch.setattr(base.asyncio, "sleep", fake)
    return slept


def _session(http):
    from marketswarm.providers.base import YahooSession
    s = YahooSession.__new__(YahooSession)
    s.client = type("C", (), {"http": http})()
    s._crumb = None
    s._lock = asyncio.Lock()
    s.failed = False
    s.rate_limited = False
    return s


def test_a_rate_limited_crumb_is_retried_rather_than_abandoned(monkeypatch):
    """429 means "wait and ask again", not "refused".

    A full swarm run fetches a chain per symbol, so a second run close behind
    trips Yahoo's limit. Giving up loses the options data for the entire
    session over a delay measured in seconds.
    """
    slept = _no_wait(monkeypatch)
    http = _CountingHttp(refusals=2)
    s = _session(http)

    assert asyncio.run(s._acquire()) == "aBcD1234"
    assert http.calls == 3, "it did not retry through the throttling"
    assert slept, "it retried without waiting, which is what got us throttled"
    assert s.rate_limited is False, "recovered, so it is no longer rate-limited"


def test_persistent_throttling_gives_up_but_says_it_was_throttled(monkeypatch):
    """The caller must be able to tell 'wait' from 'change the code'."""
    _no_wait(monkeypatch)
    s = _session(_CountingHttp(refusals=99))

    assert asyncio.run(s._acquire()) is None
    assert s.rate_limited is True, (
        "throttling reported as a refusal — the diagnostic would tell the owner "
        "Yahoo changed its endpoint when the real answer is to wait")


def test_yahoos_own_retry_after_is_honoured_over_our_backoff(monkeypatch):
    slept = _no_wait(monkeypatch)
    s = _session(_CountingHttp(refusals=1, headers={"Retry-After": "7"}))
    asyncio.run(s._acquire())

    assert slept[0] == 7.0, "ignored the server's own stated wait"


def test_an_unparseable_retry_after_falls_back_rather_than_crashing(monkeypatch):
    """Retry-After may be an HTTP date rather than seconds."""
    slept = _no_wait(monkeypatch)
    s = _session(_CountingHttp(refusals=1, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}))
    assert asyncio.run(s._acquire()) == "aBcD1234"
    assert slept and slept[0] > 0


class _NoCookieHttp:
    """Serves pages but issues no session — what the VPS actually observes."""

    def __init__(self):
        self.cookies = {}          # nothing was set

    async def get(self, url, headers=None, timeout=None):
        if "getcrumb" in url:
            # Yahoo answers the crumb endpoint 429 when there is no session.
            # It reads exactly like throttling and is not.
            return _Resp(429, "Too Many Requests")
        return _Resp(200, "<html>…</html>")


def test_no_session_cookie_is_not_reported_as_rate_limiting(monkeypatch):
    """The distinction that cost three wrong diagnoses in production.

    Yahoo serves this host 200 with no Set-Cookie, then 429s the crumb. Reading
    the 429 alone says "throttled, wait a few minutes" — which sends the reader
    to fix the wrong thing, because no wait and no retry can produce a session
    that is not being issued.
    """
    _no_wait(monkeypatch)
    s = _session(_NoCookieHttp())

    assert asyncio.run(s._acquire()) is None
    assert s.cookie_count == 0, "the absent session was not detected"


def test_a_session_that_exists_is_counted(monkeypatch):
    """So the two failures stay distinguishable in the other direction."""
    _no_wait(monkeypatch)
    http = _CountingHttp(refusals=0)
    http.cookies = {"A1": "x", "A3": "y"}
    s = _session(http)

    assert asyncio.run(s._acquire()) == "aBcD1234"
    assert s.cookie_count == 2
