"""HTTP plumbing shared by every provider: retries, caching, rate limiting,
and the Evidence record that carries provenance into the report.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("marketswarm.providers")

DEFAULT_UA = "marketswarm/1.0 (research agent; contact: set MARKETSWARM_CONTACT)"


class ProviderError(RuntimeError):
    """A provider failed in a way the agent should degrade around, not crash on.

    Carries the HTTP status when there was one, so a caller can tell an
    authentication problem it could fix from an outage it can only wait out.
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class Evidence:
    """One verifiable claim, with where it came from and how much to trust it."""

    claim: str
    source: str
    url: str | None = None
    observed_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    reliability: float = 0.6         # prior trust in the source, 0-1
    value: Any = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def cite(self) -> str:
        return f"{self.claim} [{self.source}{': ' + self.url if self.url else ''}]"


# Prior reliability by source class. Updated at runtime from scored history
# (memory.learning), so these are only the cold-start values.
SOURCE_RELIABILITY: dict[str, float] = {
    "sec_edgar": 0.98,          # primary filings — the company's own words
    "federal_reserve": 0.97,
    "treasury": 0.97,
    "bls": 0.96,
    "exchange_data": 0.93,      # quotes, OHLCV, option chains
    "cboe": 0.92,
    "company_ir": 0.90,
    "major_newswire": 0.82,     # Reuters / AP / Bloomberg / WSJ
    "financial_media": 0.70,    # CNBC / MarketWatch / Barron's
    "aggregator": 0.60,
    "analyst_estimate": 0.55,
    "social_sentiment": 0.30,   # deliberately low — reflexive and gameable
    "unknown": 0.40,
}


class ResponseCache:
    """Disk cache keyed by URL+params. Pre-market runs re-request the same
    endpoints across agents; caching keeps the agent well inside rate limits and
    makes re-runs reproducible."""

    def __init__(self, directory: Path, ttl_seconds: int = 300):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds

    def _path(self, key: str) -> Path:
        return self.dir / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            payload = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if time.time() - payload.get("_cached_at", 0) > self.ttl:
            return None
        return payload.get("data")

    def set(self, key: str, data: Any) -> None:
        try:
            self._path(key).write_text(json.dumps({"_cached_at": time.time(), "data": data}))
        except (OSError, TypeError) as exc:
            log.debug("cache write failed: %s", exc)

    def purge(self, older_than_seconds: int = 86400) -> int:
        cutoff, removed = time.time() - older_than_seconds, 0
        for f in self.dir.glob("*.json"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                pass
        return removed


class RateLimiter:
    """Token-bucket per host. SEC EDGAR enforces 10 req/s and will ban a client
    that ignores it; Yahoo throttles silently."""

    def __init__(self, rate_per_second: float = 5.0):
        self.min_interval = 1.0 / max(rate_per_second, 0.1)
        self._last: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, host: str) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last.get(host, 0.0))
            if wait > 0:
                await asyncio.sleep(wait)
            self._last[host] = time.monotonic()


class DataClient:
    """Async HTTP client with retry/backoff, caching and rate limiting."""

    def __init__(
        self,
        cache_dir: Path | str = "~/.marketswarm/cache",
        cache_ttl: int = 300,
        timeout: float = 20.0,
        user_agent: str | None = None,
        max_retries: int = 3,
        rate_per_second: float = 5.0,
    ):
        self.cache = ResponseCache(Path(cache_dir).expanduser(), cache_ttl)
        self.limiter = RateLimiter(rate_per_second)
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent or DEFAULT_UA
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "DataClient":
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": self.user_agent, "Accept": "application/json, text/html, */*"},
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_json(
        self, url: str, params: dict | None = None, use_cache: bool = True, headers: dict | None = None
    ) -> Any:
        raw = await self.get_text(url, params, use_cache, headers)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"non-JSON response from {url}: {exc}") from exc

    async def get_text(
        self, url: str, params: dict | None = None, use_cache: bool = True, headers: dict | None = None
    ) -> str:
        if self._client is None:
            raise ProviderError("DataClient used outside async context manager")

        key = f"{url}?{json.dumps(params or {}, sort_keys=True)}"
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        host = httpx.URL(url).host or "unknown"
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                await self.limiter.acquire(host)
                r = await self._client.get(url, params=params, headers=headers)
                if r.status_code == 429 or r.status_code >= 500:
                    raise ProviderError(f"{r.status_code} from {host}", r.status_code)
                # A 4xx other than 429 is a statement about the request, not a
                # transient fault. Retrying it three times with backoff only
                # makes a broken run slower and noisier — a 401 on every symbol
                # cost three attempts each before this.
                if r.status_code >= 400:
                    raise ProviderError(f"{r.status_code} from {host} for {url}",
                                        r.status_code)
                if use_cache:
                    self.cache.set(key, r.text)
                return r.text
            except ProviderError as exc:
                last = exc
                if exc.status is not None and 400 <= exc.status < 500 and exc.status != 429:
                    raise
                if attempt < self.max_retries - 1:
                    await asyncio.sleep((2**attempt) + random.random())
            except Exception as exc:  # noqa: BLE001 - degrade, never crash the run
                last = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep((2**attempt) + random.random())
        status = getattr(last, "status", None)
        raise ProviderError(
            f"GET {url} failed after {self.max_retries} attempts: {last}", status)

    @property
    def http(self) -> httpx.AsyncClient:
        """The underlying client, for flows that need cookie continuity or must
        tolerate a non-2xx response. Yahoo's cookie endpoint answers 404 and
        sets the cookie anyway, which `get_text` would correctly refuse."""
        if self._client is None:
            raise ProviderError("DataClient used outside async context manager")
        return self._client

    async def gather(self, coros: list, label: str = "batch") -> list:
        """Run provider calls concurrently, returning None where one failed.

        A single dead endpoint must never take down a pre-market run — the
        report degrades and says so instead.
        """
        results = await asyncio.gather(*coros, return_exceptions=True)
        out = []
        for r in results:
            if isinstance(r, Exception):
                log.warning("%s: provider call failed: %s", label, r)
                out.append(None)
            else:
                out.append(r)
        return out


# --------------------------------------------------------------------------
# Yahoo authentication
# --------------------------------------------------------------------------

# Yahoo's edge refuses a non-browser User-Agent on the crumb and options
# endpoints. This is not an attempt to hide what the agent is — the contact
# address still travels on every other request, and the rate limiter is
# unchanged — it is the minimum the endpoint accepts.
YAHOO_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
YAHOO_COOKIE_URL = "https://fc.yahoo.com/"
YAHOO_CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"


class YahooSession:
    """Cookie and crumb for Yahoo's query endpoints.

    Yahoo's v7 option-chain endpoint began requiring a session cookie plus a
    matching crumb, and answers 401 for every symbol without them. The chart
    endpoint still works unauthenticated, so this is applied where it is needed
    rather than globally.

    Acquired once and reused. A crumb does expire, so callers pass the 401 back
    via `refresh=True` and retry exactly once — a crumb that fails twice is a
    real outage, not a stale token, and must degrade like any other.
    """

    def __init__(self, client: "DataClient"):
        self.client = client
        self._crumb: str | None = None
        self._lock = asyncio.Lock()
        self.failed = False

    @property
    def crumb(self) -> str | None:
        return self._crumb

    async def ensure(self, refresh: bool = False) -> str | None:
        async with self._lock:
            if self._crumb and not refresh:
                return self._crumb
            self._crumb = await self._acquire()
            self.failed = self._crumb is None
            return self._crumb

    async def _acquire(self) -> str | None:
        headers = {"User-Agent": YAHOO_BROWSER_UA}
        try:
            http = self.client.http
        except ProviderError:
            return None

        # Sets the session cookie. It answers 404 by design; the cookie is the
        # point, so the status is deliberately not checked.
        try:
            await http.get(YAHOO_COOKIE_URL, headers=headers, timeout=10.0)
        except Exception as exc:  # noqa: BLE001 — no cookie is survivable
            log.debug("yahoo cookie fetch failed: %s", exc)

        try:
            r = await http.get(YAHOO_CRUMB_URL, headers=headers, timeout=10.0)
        except Exception as exc:  # noqa: BLE001 — degrade, never crash the run
            log.warning("yahoo crumb unavailable: %s", exc)
            return None

        crumb = (r.text or "").strip()
        # A crumb is a short opaque token. An HTML page here means Yahoo served
        # a consent or block interstitial, and treating that as a crumb would
        # send garbage on every subsequent call.
        if r.status_code != 200 or not crumb or len(crumb) > 64 or "<" in crumb:
            log.warning("yahoo crumb rejected: status=%s len=%d",
                        r.status_code, len(crumb))
            return None
        log.info("yahoo session established")
        return crumb

    def headers(self) -> dict:
        return {"User-Agent": YAHOO_BROWSER_UA}

    def params(self, params: dict | None = None) -> dict:
        out = dict(params or {})
        if self._crumb:
            out["crumb"] = self._crumb
        return out
