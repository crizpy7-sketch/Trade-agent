"""Where option chains come from, so that it can change without a rewrite.

The swarm lost its options data twice in a fortnight because the chain fetch was
welded to Yahoo's undocumented endpoint — first when Yahoo began requiring a
crumb, then when it stopped issuing session cookies to the host's address at
all. Neither failure was a bug in the analysis; both took the whole options half
of the report down anyway.

So the source is a seam. Everything above it — implied move, skew, open-interest
walls, unusual activity, the contract board — works on a `Chain` and does not
know or care who produced it. Replacing a provider is a config change and a new
class here, not surgery on the pipeline.

WIRE FORMATS ARE UNVERIFIED. The Polygon shapes below are written from its
published API and have never been run against the live service — the machine
they were written on has no route to it. Parsing is deliberately defensive and
`diagnose()` prints what actually came back, so one run on a networked host
either confirms the shape or shows exactly how it differs. Treat the first live
run as the test.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Protocol

from .options import Chain, Contract

log = logging.getLogger("marketswarm.chains")

POLYGON_BASE = "https://api.polygon.io"

#: Contracts fetched per expiry. A chain wider than this is being read for
#: something other than a 40-delta pick, and every extra page costs rate limit.
POLYGON_PAGE = 250

#: Pages followed before giving up. Bounded because a paginating loop against a
#: rate-limited free tier is how one symbol consumes a whole minute's quota.
POLYGON_MAX_PAGES = 4


class ChainSource(Protocol):
    """What the options layer needs from a provider, and nothing more."""

    name: str

    async def expirations(self, symbol: str) -> list[dt.date]: ...

    async def chain(self, symbol: str, expiration: dt.date | None) -> Chain | None: ...

    async def diagnose(self, symbol: str) -> str:
        """A human-readable account of what happened, for check-options.py."""
        ...


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class PolygonChains:
    """Option chains from Polygon.io.

    Polygon sells market data and nothing else — it cannot place an order, so a
    leaked key cannot trade. That is why it is the default recommendation over a
    broker's API: the safety is structural rather than a promise about scope.
    """

    name = "polygon"

    def __init__(self, client, api_key: str):
        self.client = client
        self.api_key = api_key
        self.last_error: str | None = None

    def _params(self, **kw) -> dict:
        # The key travels as a query parameter because that is what Polygon's
        # API expects. It is redacted out of logs by providers.base.redact.
        return {"apiKey": self.api_key, **kw}

    async def expirations(self, symbol: str) -> list[dt.date]:
        """Distinct upcoming expiries, from the contract reference endpoint."""
        today = dt.date.today()
        try:
            payload = await self.client.get_json(
                f"{POLYGON_BASE}/v3/reference/options/contracts",
                params=self._params(underlying_ticker=symbol.upper(),
                                    expiration_date_gte=today.isoformat(),
                                    limit=1000, sort="expiration_date"),
            )
        except Exception as exc:  # noqa: BLE001 — degrade, never kill the run
            self.last_error = f"expirations: {exc}"
            log.warning("polygon expirations %s unavailable: %s", symbol, exc)
            return []

        seen: list[dt.date] = []
        for row in (payload or {}).get("results") or []:
            raw = row.get("expiration_date")
            if not raw:
                continue
            try:
                d = dt.date.fromisoformat(raw)
            except ValueError:
                continue
            if d >= today and d not in seen:
                seen.append(d)
        return sorted(seen)

    async def chain(self, symbol: str, expiration: dt.date | None = None) -> Chain | None:
        rows, underlying = await self._snapshot(symbol, expiration)
        if not rows:
            return None

        calls: list[Contract] = []
        puts: list[Contract] = []
        exp_seen: dt.date | None = None

        for row in rows:
            details = row.get("details") or {}
            kind = (details.get("contract_type") or "").lower()
            if kind not in ("call", "put"):
                continue
            strike = _f(details.get("strike_price"))
            if strike <= 0:
                continue

            raw_exp = details.get("expiration_date")
            if raw_exp and exp_seen is None:
                try:
                    exp_seen = dt.date.fromisoformat(raw_exp)
                except ValueError:
                    pass

            quote = row.get("last_quote") or {}
            day = row.get("day") or {}
            bid, ask = _f(quote.get("bid")), _f(quote.get("ask"))
            last = _f(day.get("close")) or _f(quote.get("midpoint"))

            contract = Contract(
                strike=strike, last=last, bid=bid, ask=ask,
                volume=_i(day.get("volume")),
                open_interest=_i(row.get("open_interest")),
                implied_volatility=_f(row.get("implied_volatility")),
                in_the_money=(strike < underlying if kind == "call"
                              else strike > underlying),
                kind=kind,
            )
            (calls if kind == "call" else puts).append(contract)

        if not calls and not puts:
            self.last_error = "snapshot returned rows but none parsed as contracts"
            return None
        if underlying <= 0:
            self.last_error = "no underlying price in the snapshot"
            return None

        exp = expiration or exp_seen or dt.date.today()
        dte = max((exp - dt.date.today()).days, 0)
        # 0DTE: express remaining life as the fraction of a session still ahead,
        # matching the Yahoo source so downstream pricing is unchanged.
        return Chain(symbol.upper(), underlying, exp, dte if dte > 0 else 0.4,
                     sorted(calls, key=lambda c: c.strike),
                     sorted(puts, key=lambda c: c.strike))

    async def _snapshot(self, symbol: str, expiration: dt.date | None):
        params = self._params(limit=POLYGON_PAGE)
        if expiration:
            params["expiration_date"] = expiration.isoformat()
        url = f"{POLYGON_BASE}/v3/snapshot/options/{symbol.upper()}"

        rows: list[dict] = []
        underlying = 0.0
        for _ in range(POLYGON_MAX_PAGES):
            try:
                payload = await self.client.get_json(url, params=params, use_cache=False)
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"snapshot: {exc}"
                log.warning("polygon snapshot %s unavailable: %s", symbol, exc)
                break

            results = (payload or {}).get("results") or []
            rows.extend(results)
            if not underlying:
                for row in results:
                    underlying = _f((row.get("underlying_asset") or {}).get("price"))
                    if underlying:
                        break

            nxt = (payload or {}).get("next_url")
            if not nxt or not results:
                break
            # next_url carries its own cursor; the key still has to be attached.
            url, params = nxt, self._params()
        return rows, underlying

    async def diagnose(self, symbol: str) -> str:
        exps = await self.expirations(symbol)
        if not exps:
            return (f"polygon: no expirations for {symbol}. "
                    f"{self.last_error or 'the reference endpoint returned nothing'}\n"
                    "  A 401/403 means the key is wrong or lacks options access.\n"
                    "  A 429 means the free tier's rate limit — wait and retry.")
        c = await self.chain(symbol, exps[0])
        if c is None:
            return (f"polygon: {len(exps)} expirations, but no usable chain for "
                    f"{exps[0]}. {self.last_error or 'snapshot returned nothing'}\n"
                    "  Options snapshots are not on every Polygon plan; the "
                    "reference endpoint being reachable does not prove the "
                    "snapshot endpoint is.")
        return (f"polygon: {len(exps)} expirations, nearest {exps[0]}, "
                f"{len(c.calls)} calls / {len(c.puts)} puts, "
                f"underlying {c.underlying_price}")


def select_source(client, *, provider: str | None, polygon_key: str | None):
    """Pick a chain source, or None to keep the built-in Yahoo path.

    "auto" prefers a configured key over the endpoint that has already failed
    twice. Naming a provider explicitly overrides that, including naming yahoo,
    because being able to force the old path is what makes a bad switch
    recoverable without a deploy.
    """
    choice = (provider or "auto").strip().lower()

    if choice == "yahoo":
        return None
    if choice == "polygon":
        if not polygon_key:
            log.warning("options provider is polygon but no API key is set — "
                        "falling back to yahoo, which may not work on this host")
            return None
        return PolygonChains(client, polygon_key)
    if choice != "auto":
        log.warning("unknown options provider %r; using auto", provider)

    return PolygonChains(client, polygon_key) if polygon_key else None
