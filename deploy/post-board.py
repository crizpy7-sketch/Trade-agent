"""Fetch live option chains, build today's contract board, post it to Discord.

    sudo marketswarm-python /root/marketswarm/deploy/post-board.py --dry-run  # print only
    sudo marketswarm-python /root/marketswarm/deploy/post-board.py            # post to Discord
    sudo marketswarm-python /root/marketswarm/deploy/post-board.py NVDA TSLA  # pick the names

Why this exists separately from the daily run: the 08:15 daemon posts a full
report, and you cannot ask it for one on demand without either waiting until
tomorrow or firing a whole swarm run out of session. This does the one thing —
real chains, real strikes, real prices, posted now — so the board can be checked
against a broker screen while the market is open.

Every number it prints comes from a live quote. Nothing here is estimated,
modelled or filled in, because the whole point is that you can look these
contracts up and find them.

Read-only and advisory. It fetches chains and posts text. There is no order
path in MarketSwarm and this adds none.
"""

from __future__ import annotations

import asyncio
import sys

from marketswarm.config import Config
from marketswarm.notify import send_webhook
from marketswarm.providers.base import DataClient
from marketswarm.providers.options import OptionsData
from marketswarm.recommend.speculative import build_board, filled_count

#: Kept small deliberately. Each symbol is a chain fetch, and a board only has
#: six slots — screening thirty names to fill six is spent rate limit.
DEFAULT_SYMBOLS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMD", "TSLA", "META"]

MAX_DISCORD = 1900


async def collect(symbols: list[str]) -> tuple[dict, list[str]]:
    """Live chains, and the names we could not read."""
    flows, failed = {}, []
    async with DataClient() as client:
        options = OptionsData(client)
        if not await options.session.ensure():
            return {}, list(symbols)
        for sym in symbols:
            try:
                snap = await options.flow_snapshot(sym)
            except Exception as exc:  # noqa: BLE001 — one bad name is not fatal
                print(f"  {sym}: {exc}", file=sys.stderr)
                snap = None
            if snap and snap.get("chain"):
                flows[sym] = snap
            else:
                failed.append(sym)
    return flows, failed


def format_for_discord(board: dict, failed: list[str]) -> str:
    lines = ["**Contract Board — live chains, fetched just now**"]
    lines.append("_Not recommendations. The tier on each row says how much is "
                 "behind it. Nothing here is scored or tracked._")

    for title, key in (("Calls", "calls"), ("Puts", "puts")):
        lines.append("")
        lines.append(f"__{title}__")
        for n, r in enumerate(board[key], 1):
            if r["strike"] is None:
                lines.append(f"{n}. **{r['tier']}** — {r['reason'][:90]}")
                continue
            lines.append(
                f"{n}. **{r['symbol']} {r['strike']:g} {r['right'].upper()}** "
                f"exp {r['expiration']} · **{r['tier']}**"
            )
            lines.append(
                f"    ${r['premium']:.2f} mid · costs ${r['cost']:,.0f} · "
                f"max loss ${r['max_loss']:,.0f} · breakeven ${r['breakeven']:.2f}"
            )

    if failed:
        lines.append("")
        lines.append(f"_No chain for: {', '.join(failed)}_")
    lines.append("")
    lines.append("_Research/educational analysis only — not financial advice._")

    text = "\n".join(lines)
    return text if len(text) <= MAX_DISCORD else text[:MAX_DISCORD - 14] + "\n… (truncated)"


def main() -> int:
    dry = "--dry-run" in sys.argv
    symbols = [a.upper() for a in sys.argv[1:] if not a.startswith("-")] or DEFAULT_SYMBOLS

    cfg = Config.load()
    if not dry and not cfg.webhook_url:
        print("MARKETSWARM_WEBHOOK is not set — nothing to post to.\n"
              "Run with --dry-run to print the board instead.", file=sys.stderr)
        return 1

    print(f"fetching chains for {', '.join(symbols)} …", file=sys.stderr)
    flows, failed = asyncio.run(collect(symbols))

    if not flows:
        # Deliberately not posted. An all-empty board in the channel looks like
        # the swarm having an opinion about a quiet market, when the truth is
        # that nothing was reachable — two very different messages.
        print("FAIL  no chains reachable. Either Yahoo refused this host, or the\n"
              "      handshake is broken. Check with:\n"
              "      /opt/marketswarm/venv/bin/python "
              "/root/marketswarm/deploy/check-options.py SPY", file=sys.stderr)
        return 1

    board = build_board(flows)
    text = format_for_discord(board, failed)
    print(text)

    if dry:
        print(f"\n(dry run — {filled_count(board)} of 6 slots carry a live contract)",
              file=sys.stderr)
        return 0

    ok = send_webhook(cfg.webhook_url, text)
    print("\nposted to Discord" if ok else "\nDiscord refused the post", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
