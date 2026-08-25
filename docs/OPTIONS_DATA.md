# Where option chains come from

The swarm lost its options data twice in a fortnight. Both times the analysis
was fine and the source was not:

1. Yahoo began requiring a session cookie plus a matching crumb on its v7
   option-chain endpoint, and every symbol started answering 401.
2. Yahoo stopped issuing session cookies to the VPS's address altogether. It
   still serves the pages — `finance.yahoo.com` returns 200 with no
   `Set-Cookie` and no redirect — but with no session there is nothing to
   attach a crumb to, and the crumb endpoint then answers **429**.

That 429 is the trap worth remembering: read alone it says *throttled, wait a
few minutes*, and waiting does nothing. Both crumb hosts answer 429 on a cold
single request with no burst behind it, which rules out throttling; a `curl`
carrying a cookie jar collects zero cookies, which identifies it.

```sh
curl -sS -D - -o /dev/null https://finance.yahoo.com/ | grep -i set-cookie
```

No output means no session is being issued, and **no code change fixes that** —
it is the server's address. Shared VPS ranges get blocked because of what else
lives on them.

## The seam

`marketswarm/providers/chains.py` holds the sources. Everything above it —
implied move, skew, open-interest walls, unusual activity, the contract board —
works on a `Chain` and never learns who produced it. Changing provider is a
config change plus a class, not surgery on the pipeline.

```
MARKETSWARM_OPTIONS_PROVIDER=auto     # auto | yahoo | polygon
MARKETSWARM_POLYGON_KEY=...
```

`auto` uses Polygon when a key is set and falls back to Yahoo otherwise.
Naming a provider forces it — including naming `yahoo`, which is what makes a
bad switch recoverable without a deploy.

## Why Polygon rather than a broker's API

Polygon sells market data and nothing else. It **cannot place an order**, so a
leaked key cannot trade. Alpaca and Tradier both offer good options data, but
their keys are brokerage account credentials: putting one on the server means
the machine holds something capable of trading, which is precisely what this
system is built not to have. A paper-only account is a reasonable mitigation,
but a data vendor needs no mitigation — the capability does not exist.

The key is redacted from logs by `security.redact` (the `generic_api_key`
pattern matches `apiKey=`), and travels as a query parameter because that is
what Polygon's API expects.

## Previous-close data is enough

The swarm runs before the open, and open interest is a settlement figure that
updates once a day. A previous-close plan is sufficient; real-time buys
precision the morning report cannot use. Check the cheapest tier first.

## Checking it

```sh
sudo marketswarm-python check-options.py SPY
```

It names the provider in use and, for a keyed provider, reports what actually
came back — whether the key works, whether the plan includes option snapshots,
and how many contracts arrived. A reachable reference endpoint does not prove
the snapshot endpoint is included in the plan, and the check distinguishes them.

## The state of the Polygon code

**The wire format is unverified.** It is written from Polygon's published API
and has never run against the live service — the machine it was written on has
no route to it. Parsing is deliberately defensive and `diagnose()` prints what
arrived, so the first run on a networked host either confirms the shape or shows
exactly how it differs. Treat that run as the test.
