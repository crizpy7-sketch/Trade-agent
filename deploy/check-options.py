"""Check that Yahoo's option-chain endpoint still authenticates.

Run on a host with real network access. Absolute paths on both halves, so it
does not matter which directory you are standing in:

    sudo /opt/marketswarm/venv/bin/python /root/marketswarm/deploy/check-options.py SPY

(The second path is the repo checkout. install.sh copies only the package into
/opt, not deploy/, so this script is not there.)

Why this exists rather than being read off a normal run: OptionsData's
expirations() and chain() catch ProviderError and return empty, so the swarm
degrades instead of failing when Yahoo refuses it. That is right for a morning
report — one dead provider must not take the run down — but it means a broken
handshake and a genuinely empty chain look identical from outside. This tells
them apart and says which happened.

Yahoo has changed this endpoint's authentication once already. When it changes
again the symptom will be options data quietly going missing, and this is the
first thing to run.

Exit status is 0 only if real option data came back, so it is usable from cron
or a health check, not only by eye.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile

from marketswarm.providers.base import DataClient
from marketswarm.providers.options import OptionsData


async def check(symbol: str, cache_dir: str) -> int:
    # A throwaway cache, so the check neither reads a stale success nor leaves
    # anything behind in whichever account happens to run it.
    async with DataClient(cache_dir=cache_dir) as client:
        options = OptionsData(client)

        crumb = await options.session.ensure()
        if not crumb:
            print("FAIL  no Yahoo crumb — the cookie or crumb request was refused.")
            print("      Options data will be missing from every report until this")
            print("      works. Two very different causes look the same here:")
            print()
            print("        Yahoo changed the endpoint again")
            print("        this host cannot reach Yahoo at all")
            print()
            print("      Tell them apart before changing any code:")
            print("        curl -sS -o /dev/null -w '%{http_code}\\n' https://finance.yahoo.com/")
            print("      A code means you reached Yahoo and the problem is theirs.")
            print("      A connection or proxy error means the problem is this host's.")
            return 1
        print(f"ok    crumb acquired ({len(crumb)} chars)")

        expirations = await options.expirations(symbol)
        if not expirations:
            print(f"FAIL  crumb accepted, but {symbol} returned no expirations.")
            print("      Authentication works and the payload still is not usable.")
            return 1
        print(f"ok    {symbol}: {len(expirations)} expirations, nearest {expirations[0]}")

        chain = await options.chain(symbol, expirations[0])
        if chain is None:
            print(f"FAIL  no chain for {symbol} {expirations[0]}")
            return 1
        if not chain.calls and not chain.puts:
            print(f"FAIL  chain for {symbol} {expirations[0]} came back with no contracts")
            return 1
        print(f"ok    chain: {len(chain.calls)} calls / {len(chain.puts)} puts, "
              f"underlying {chain.underlying_price}")

        atm = chain.atm("call")
        if atm:
            print(f"ok    ATM call strike {atm.strike}, mid {atm.mid:.2f}, "
                  f"OI {atm.open_interest}")

        print("\nPASS  options data is live")
        return 0


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    cache_dir = tempfile.mkdtemp(prefix="marketswarm-check-")
    try:
        return asyncio.run(check(symbol, cache_dir))
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
