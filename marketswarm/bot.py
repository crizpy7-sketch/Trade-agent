"""Interactive Discord bot: ask the swarm things instead of only being told.

The webhook in `notify.py` pushes one report a day and cannot receive anything.
This is the other direction — a bot that reads a channel, answers questions
about what the swarm found, and looks up a symbol on demand.

**Why polling and not a gateway connection.** A gateway bot needs a persistent
WebSocket and, in practice, a library. Polling `GET /channels/{id}/messages`
needs neither: it reuses the httpx client the providers already depend on, adds
no package, needs no inbound port and no TLS certificate, and survives a network
blip by simply asking again. The cost is a few seconds of latency on a system
whose answers take longer than that to compute anyway. For a single-operator
research tool that is the right trade, and it keeps to the project's rule of
deterministic, boring runtime code.

**Authorisation fails closed.** With no allowlist configured the bot answers
nobody. A research command spends API quota and provider rate limit, so an open
bot in a channel someone else can join is a way to burn both. `final_authority`
is unchanged by anything here: the bot reads, looks up and explains. It cannot
run a pre-market pass, cannot publish, cannot trade, and has no path to any of
those — the commands map onto the read API and the quote provider, and nothing
else is reachable from here.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import time
from contextlib import contextmanager

import httpx

from . import __version__, clock
from .api import ReadOnlyAPI
from .config import Config
from .memory import MemoryStore

log = logging.getLogger("marketswarm.bot")

API = "https://discord.com/api/v10"
POLL_SECONDS = 3.0
MAX_REPLY = 1900          # Discord's limit is 2000; leave room for the wrapper
COOLDOWN_SECONDS = 5.0    # per user, per command


def _env_ids(name: str) -> set[str]:
    raw = os.environ.get(name, "")
    return {p.strip() for p in raw.replace(",", " ").split() if p.strip()}


class Unauthorized(Exception):
    """Raised for a command from someone not on the allowlist."""


class DiscordBot:
    def __init__(self, cfg: Config, token: str | None = None,
                 channel_id: str | None = None, allowed: set[str] | None = None):
        self.cfg = cfg
        self.token = token or os.environ.get("MARKETSWARM_DISCORD_BOT_TOKEN", "")
        self.channel_id = channel_id or os.environ.get("MARKETSWARM_DISCORD_CHANNEL_ID", "")
        self.allowed = allowed if allowed is not None else \
            _env_ids("MARKETSWARM_DISCORD_ALLOWED_USERS")
        self._last_id: str | None = None
        self._cooldown: dict[str, float] = {}
        self._http: httpx.AsyncClient | None = None

    # ---------------------------------------------------------------- wiring

    def configured(self) -> tuple[bool, str]:
        if not self.token:
            return False, "MARKETSWARM_DISCORD_BOT_TOKEN is not set"
        if not self.channel_id:
            return False, "MARKETSWARM_DISCORD_CHANNEL_ID is not set"
        if not self.allowed:
            return False, ("MARKETSWARM_DISCORD_ALLOWED_USERS is empty — the bot "
                           "refuses to answer anyone rather than answer everyone")
        return True, "ready"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bot {self.token}",
            "User-Agent": f"MarketSwarm/{__version__} (+https://github.com/crizpy7-sketch/marketswarm)",
            "Content-Type": "application/json",
        }

    async def _get(self, path: str, params: dict | None = None):
        r = await self._http.get(f"{API}{path}", params=params, headers=self._headers())
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 5))
            log.warning("rate limited by discord; waiting %.1fs", wait)
            await asyncio.sleep(wait)
            return None
        if r.status_code >= 400:
            log.warning("discord GET %s -> %s: %s", path, r.status_code, r.text[:200])
            return None
        return r.json()

    async def send(self, text: str) -> bool:
        """Post a reply. Long answers are truncated with a visible marker rather
        than silently cut, so a clipped answer never reads as a complete one."""
        if len(text) > MAX_REPLY:
            text = text[:MAX_REPLY] + "\n… (truncated)"
        r = await self._http.post(f"{API}/channels/{self.channel_id}/messages",
                                  json={"content": text}, headers=self._headers())
        if r.status_code >= 400:
            log.warning("discord post failed %s: %s", r.status_code, r.text[:200])
            return False
        return True

    # ---------------------------------------------------------------- polling

    async def poll_once(self) -> list[dict]:
        params = {"limit": 20}
        if self._last_id:
            params["after"] = self._last_id
        payload = await self._get(f"/channels/{self.channel_id}/messages", params)
        if not payload:
            return []
        # Discord returns newest first; process oldest first so a burst of
        # commands is answered in the order it was typed.
        messages = list(reversed(payload))
        if messages:
            self._last_id = messages[-1]["id"]
        return messages

    def _rate_limited(self, user_id: str) -> bool:
        now = time.monotonic()
        last = self._cooldown.get(user_id, 0.0)
        if now - last < COOLDOWN_SECONDS:
            return True
        self._cooldown[user_id] = now
        return False

    async def handle(self, message: dict) -> str | None:
        if message.get("author", {}).get("bot"):
            return None                      # never answer ourselves
        content = (message.get("content") or "").strip()
        if not content.startswith("!"):
            return None

        user = str(message.get("author", {}).get("id", ""))
        name = message.get("author", {}).get("username", "someone")

        if user not in self.allowed:
            log.warning("refused command from unauthorised user %s (%s)", name, user)
            return (f"Not authorised. Add `{user}` to "
                    f"`MARKETSWARM_DISCORD_ALLOWED_USERS` if that is you.")

        if self._rate_limited(user):
            return None                      # silently drop, do not spam back

        parts = content[1:].split()
        cmd, args = parts[0].lower(), parts[1:]
        try:
            return await self.dispatch(cmd, args)
        except Exception as exc:  # noqa: BLE001 — a bad command must not kill the bot
            log.exception("command %r failed", cmd)
            return f"`!{cmd}` failed: {exc}"

    # ---------------------------------------------------------------- commands

    async def dispatch(self, cmd: str, args: list[str]) -> str:
        handlers = {
            "help": self.cmd_help,
            "status": self.cmd_status,
            "today": self.cmd_today,
            "report": self.cmd_today,
            "ticker": self.cmd_ticker,
            "look": self.cmd_ticker,
            "why": self.cmd_why,
            "calibration": self.cmd_calibration,
            "agents": self.cmd_agents,
        }
        handler = handlers.get(cmd)
        if not handler:
            return (f"Unknown command `!{cmd}`. Try `!help`.")
        return await handler(args)

    async def cmd_help(self, args: list[str]) -> str:
        return (
            "**MarketSwarm**\n"
            "`!status` — is the swarm healthy, when did it last run\n"
            "`!today` — what it found this morning, and what it rejected\n"
            "`!ticker SYM` — live quote plus what the swarm said about it today\n"
            "`!why SYM` — why that symbol was rejected today\n"
            "`!calibration` — the track record so far\n"
            "`!agents` — per-agent health\n"
            "\n_Read-only. The bot cannot run the swarm, publish, or trade._"
        )

    @contextmanager
    def _api(self):
        """A fresh read-only view per command.

        Opened and closed per command rather than held for the life of the
        process: the daemon writes to the same database, and a long-lived reader
        is how a bot ends up answering from a snapshot that predates the run it
        is being asked about.
        """
        store = MemoryStore(self.cfg.db_path)
        try:
            yield ReadOnlyAPI(store.conn)
        finally:
            store.close()

    async def cmd_status(self, args: list[str]) -> str:
        with self._api() as api:
            s = api.system_status()
        today = clock.now_et().date()
        lines = [f"**Status** · {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC"]
        for k, v in s.items():
            lines.append(f"• {str(k).replace('_', ' ')}: {v}")
        lines.append(f"• market open today: {clock.is_trading_day(today)}")
        return "\n".join(lines)

    async def cmd_today(self, args: list[str]) -> str:
        run_date = args[0] if args else clock.now_et().date().isoformat()
        with self._api() as api:
            recs = api.recommendations(run_date=run_date)
            rej = api.rejected(run_date=run_date)
        if not recs and not rej:
            return f"No run recorded for {run_date}."
        out = [f"**{run_date}** — {len(recs)} published, {len(rej)} rejected"]
        for r in recs[:8]:
            out.append(f"• **{r.get('subject')}** {r.get('direction', '')} "
                       f"P {float(r.get('probability') or 0):.0%} "
                       f"conf {r.get('confidence')}")
        if not recs:
            out.append("_Nothing cleared the gate. That is a result, not a gap._")
        for r in rej[:8]:
            out.append(f"• ~~{r.get('subject')}~~ rejected — "
                       f"{str(r.get('reason') or '')[:110]}")
        return "\n".join(out)

    async def cmd_why(self, args: list[str]) -> str:
        if not args:
            return "Usage: `!why SPY`"
        symbol = args[0].upper()
        run_date = clock.now_et().date().isoformat()
        with self._api() as api:
            rejected = api.rejected(run_date=run_date)
        for r in rejected:
            if str(r.get("subject", "")).upper() == symbol:
                out = [f"**{symbol}** — rejected {run_date}",
                       str(r.get("reason") or "no reason recorded")]
                findings = r.get("findings")
                if findings:
                    out.append(f"```{str(findings)[:700]}```")
                return "\n".join(out)
        return f"{symbol} was not rejected today. Try `!today`."

    async def cmd_ticker(self, args: list[str]) -> str:
        """Live quote, plus whatever the swarm concluded about it today.

        The quote is fetched now; the verdict is from the morning's run. Both are
        labelled with when they are from, because a fresh price beside a stale
        verdict reads as one coherent statement and is not.
        """
        if not args:
            return "Usage: `!ticker NVDA`"
        symbol = args[0].upper()[:12]

        from .providers.base import DataClient
        from .providers.market import MarketData

        quote = None
        try:
            async with DataClient(cache_dir=self.cfg.cache_dir, cache_ttl=60,
                                  user_agent=self.cfg.user_agent) as client:
                quote = await MarketData(client).quote(symbol)
        except Exception as exc:  # noqa: BLE001 — answer with what we have
            log.warning("quote %s failed: %s", symbol, exc)

        lines = [f"**{symbol}**"]
        if quote:
            lines.append(f"{quote.price:,.2f} ({quote.change_pct:+.2f}%) · "
                         f"{quote.market_state.lower()} · as of {quote.as_of or 'now'}")
            if quote.day_low and quote.day_high:
                lines.append(f"day range {quote.day_low:,.2f}–{quote.day_high:,.2f}")
        else:
            lines.append("_live quote unavailable right now_")

        run_date = clock.now_et().date().isoformat()
        verdict = None
        with self._api() as api:
            for r in api.recommendations(run_date=run_date):
                if str(r.get("subject", "")).upper() == symbol:
                    verdict = (f"published — {r.get('direction')} "
                               f"P {float(r.get('probability') or 0):.0%}, "
                               f"conf {r.get('confidence')}")
            if not verdict:
                for r in api.rejected(run_date=run_date):
                    if str(r.get("subject", "")).upper() == symbol:
                        verdict = f"rejected — {str(r.get('reason') or '')[:140]}"
        lines.append(f"\n**Swarm, {run_date}:** {verdict or 'not examined today'}")
        lines.append("_Research only — not financial advice._")
        return "\n".join(lines)

    async def cmd_calibration(self, args: list[str]) -> str:
        with self._api() as api:
            perf = api.performance()
        if not perf or not perf.get("n"):
            return "No resolved predictions yet — nothing to calibrate against."
        out = ["**Track record**"]
        for k, v in perf.items():
            label = str(k).replace("_", " ")
            out.append(f"• {label}: {v:.3f}" if isinstance(v, float)
                       else f"• {label}: {v}")
        out.append("_No demonstrated predictive edge. These are scores, not a claim._")
        return "\n".join(out)

    async def cmd_agents(self, args: list[str]) -> str:
        with self._api() as api:
            rows = api.agent_status()
        if not rows:
            return "No agent runs recorded yet."
        ok = [r for r in rows if r.get("status") == "ok"]
        bad = [r for r in rows if r.get("status") != "ok"]
        out = [f"**Agents** — {len(ok)} ok, {len(bad)} degraded"]
        for r in bad[:12]:
            out.append(f"• {r.get('agent')}: {r.get('status')} "
                       f"{str(r.get('error') or '')[:90]}")
        return "\n".join(out)

    # ---------------------------------------------------------------- loop

    async def run_forever(self) -> None:
        ok, why = self.configured()
        if not ok:
            raise RuntimeError(f"discord bot not configured: {why}")

        async with httpx.AsyncClient(timeout=20.0) as http:
            self._http = http
            log.info("bot listening on channel %s for %d authorised user(s)",
                     self.channel_id, len(self.allowed))
            # Start from now: a restart must not replay and re-answer the
            # channel's backlog.
            await self.poll_once()
            while True:
                try:
                    for message in await self.poll_once():
                        reply = await self.handle(message)
                        if reply:
                            await self.send(reply)
                except Exception:  # noqa: BLE001 — the loop outlives any one failure
                    log.exception("poll cycle failed; continuing")
                await asyncio.sleep(POLL_SECONDS)
