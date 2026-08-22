"""Permissioned community research from Discord, TradingView alerts, and X.

There is deliberately no TradingView scraper and no Discord self-bot here.
TradingView community pages are manual-use only, and Discord automation must use
a bot account.  The provider therefore reads only:

* Discord channels the configured bot can access (including a private intake
  channel that receives TradingView alerts or manually forwarded idea links),
* posts from explicitly allowlisted X handles through X's official API.

Community text is untrusted.  It is sanitised, parsed deterministically, and
kept at a low reliability prior. A single configured account can never become
independent confirmation by posting the same thesis repeatedly.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from ..security import sanitise_external_text
from .base import DataClient, ProviderError

DISCORD_API = "https://discord.com/api/v10"
X_API = "https://api.x.com/2"
log = logging.getLogger("marketswarm.providers.community")

_CASHTAG = re.compile(r"(?<![A-Za-z0-9])\$([A-Za-z]{1,5})(?![A-Za-z])")
_EXPLICIT_DIRECTION = re.compile(
    r"\b(?:direction|bias|side)\s*[:=]\s*"
    r"(long|short|bullish|bearish|call|calls|put|puts)\b",
    re.IGNORECASE,
)
_BULLISH = re.compile(
    r"\b(bullish|bull\s+case|going\s+long|long\s+setup|buy\s+setup|"
    r"call\s+setup|calls?\s+above|breakout\s+above|upside\s+target)\b",
    re.IGNORECASE,
)
_BEARISH = re.compile(
    r"\b(bearish|bear\s+case|going\s+short|short\s+setup|sell\s+setup|"
    r"put\s+setup|puts?\s+below|breakdown\s+below|downside\s+target)\b",
    re.IGNORECASE,
)


def _parse_time(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except (TypeError, ValueError):
        return None


def _direction(text: str) -> str | None:
    """Return long/short only when the language is unambiguous.

    Generic words such as ``call`` and ``put`` are intentionally insufficient:
    "sold puts" is bullish and "covered calls" is not a clean bearish view.
    Structured ``direction=...`` text is preferred for alerts.
    """
    explicit = _EXPLICIT_DIRECTION.search(text)
    if explicit:
        return "long" if explicit.group(1).lower() in {
            "long", "bullish", "call", "calls"
        } else "short"
    bullish = bool(_BULLISH.search(text))
    bearish = bool(_BEARISH.search(text))
    if bullish == bearish:
        return None
    return "long" if bullish else "short"


def _symbols(text: str, universe: set[str]) -> tuple[str, ...]:
    found = {m.group(1).upper() for m in _CASHTAG.finditer(text)}
    # A configured universe is small enough for exact token matching and avoids
    # treating ordinary words such as "IT" or "ALL" as tickers.
    upper = text.upper()
    for symbol in universe:
        if re.search(rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])", upper):
            found.add(symbol)
    return tuple(sorted(found & universe))


@dataclass(frozen=True)
class CommunityPost:
    platform: str                    # x | discord | tradingview
    source_id: str                   # platform-native immutable id
    author: str
    text: str
    created_at: str
    url: str | None
    symbols: tuple[str, ...]
    direction: str | None
    engagement: int = 0

    @property
    def source_key(self) -> str:
        """One author is one source, however many times they repeat a view."""
        # Matching handles across platforms are conservatively treated as the
        # same identity. Different aliases cannot be resolved automatically,
        # so documentation calls this an account-level independence check,
        # never proof that two posts came from two different humans.
        return self.author.casefold()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        return payload


def parse_post(
    *, platform: str, source_id: str, author: str, text: str,
    created_at: str, url: str | None, universe: Iterable[str],
    engagement: int = 0,
) -> CommunityPost | None:
    clean = sanitise_external_text(text or "", max_chars=1200,
                                   label=f"{platform}_community").text.strip()
    if not clean:
        return None
    allowed = {s.upper() for s in universe}
    symbols = _symbols(clean, allowed)
    if not symbols:
        return None
    return CommunityPost(
        platform=platform,
        source_id=str(source_id),
        author=(author or "unknown")[:100],
        text=clean,
        created_at=created_at,
        url=url,
        symbols=symbols,
        direction=_direction(clean),
        engagement=max(0, int(engagement or 0)),
    )


def aggregate_symbol_reads(posts: Iterable[CommunityPost],
                           min_sources: int = 2) -> dict[str, dict]:
    """Build low-weight per-symbol consensus from independent accounts.

    Engagement is retained for display but does not increase predictive weight;
    popularity is not a track record.  For each author, only their newest
    directional post about a symbol counts.
    """
    min_sources = max(2, min(int(min_sources), 10))
    grouped: dict[str, dict[str, CommunityPost]] = {}
    for post in sorted(posts, key=lambda p: p.created_at):
        if post.direction is None:
            continue
        for symbol in post.symbols:
            grouped.setdefault(symbol, {})[post.source_key] = post

    reads: dict[str, dict] = {}
    for symbol, by_source in grouped.items():
        directional = list(by_source.values())
        longs = [p for p in directional if p.direction == "long"]
        shorts = [p for p in directional if p.direction == "short"]
        total = len(directional)
        leader = max(len(longs), len(shorts))
        consensus = leader / total if total else 0.0
        direction = ("long" if len(longs) > len(shorts) else
                     "short" if len(shorts) > len(longs) else None)
        qualifies = bool(direction and total >= min_sources and consensus >= 2 / 3)

        # Cold-start social evidence may nudge a symbol read, never dominate it.
        # Two agreeing accounts produce only a 53/47 observation; the cap is 58/42.
        edge = min(0.08, 0.02 + 0.01 * max(0, leader - 1)) if qualifies else 0.0
        probability_up = 0.5
        if qualifies:
            probability_up += edge if direction == "long" else -edge

        reads[symbol] = {
            "symbol": symbol,
            "direction": direction,
            "probability_up": round(probability_up, 4),
            "qualifies": qualifies,
            "independent_sources": total,
            "long_sources": len(longs),
            "short_sources": len(shorts),
            "consensus": round(consensus, 4),
            "posts": [p.to_dict() for p in sorted(
                directional, key=lambda p: (p.created_at, p.engagement), reverse=True
            )[:6]],
        }
    return reads


class CommunityData:
    def __init__(self, client: DataClient, universe: Iterable[str],
                 lookback_hours: int = 24):
        self.client = client
        self.universe = tuple(s.upper() for s in universe)
        self.lookback_hours = max(1, min(int(lookback_hours), 168))

    @property
    def cutoff(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            hours=self.lookback_hours)

    async def discord(self, channel_ids: Iterable[str], bot_token: str) -> list[CommunityPost]:
        """Read channels the configured Discord bot is authorised to view."""
        if not bot_token:
            return []
        headers = {"Authorization": f"Bot {bot_token}"}
        bot_id = ""
        try:
            me = await self.client.get_json(
                f"{DISCORD_API}/users/@me", use_cache=True, headers=headers,
            )
            if isinstance(me, dict):
                bot_id = str(me.get("id") or "")
        except ProviderError as exc:
            log.warning("Discord bot identity unavailable; intake continues: %s", exc)
        posts: list[CommunityPost] = []
        for channel_id in dict.fromkeys(str(c).strip() for c in channel_ids if str(c).strip()):
            try:
                payload = await self.client.get_json(
                    f"{DISCORD_API}/channels/{channel_id}/messages",
                    params={"limit": 100}, use_cache=False, headers=headers,
                )
            except ProviderError as exc:
                # A removed server, revoked channel permission, or transient
                # failure must not discard posts already read elsewhere.
                log.warning("Discord community channel %s unavailable: %s",
                            channel_id, exc)
                continue
            if not isinstance(payload, list):
                continue
            for message in payload:
                created = _parse_time(message.get("timestamp"))
                if created and created < self.cutoff:
                    continue
                text = _discord_text(message)
                author = message.get("author") or {}
                if bot_id and str(author.get("id") or "") == bot_id:
                    continue                 # never feed MarketSwarm back into itself
                platform = "tradingview" if "tradingview.com" in text.lower() else "discord"
                guild_id = message.get("guild_id")
                message_id = str(message.get("id", ""))
                url = (f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
                       if guild_id and message_id else None)
                post = parse_post(
                    platform=platform,
                    source_id=message_id,
                    author=str(author.get("username") or author.get("id") or "unknown"),
                    text=text,
                    created_at=(created or dt.datetime.now(dt.timezone.utc)).isoformat(),
                    url=url,
                    universe=self.universe,
                )
                if post:
                    posts.append(post)
        return _dedupe(posts)

    async def x(self, handles: Iterable[str], bearer_token: str) -> list[CommunityPost]:
        """Read allowlisted public accounts through X API v2."""
        clean_handles = [str(h).strip().lstrip("@").lower() for h in handles
                         if str(h).strip()]
        clean_handles = list(dict.fromkeys(clean_handles))[:100]
        if not bearer_token or not clean_handles:
            return []

        headers = {"Authorization": f"Bearer {bearer_token}"}
        users_payload = await self.client.get_json(
            f"{X_API}/users/by",
            params={"usernames": ",".join(clean_handles)},
            use_cache=True,
            headers=headers,
        )
        users = users_payload.get("data", []) if isinstance(users_payload, dict) else []
        posts: list[CommunityPost] = []
        start_time = self.cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        for user in users:
            user_id = str(user.get("id", ""))
            username = str(user.get("username", ""))
            if not user_id or not username:
                continue
            try:
                payload = await self.client.get_json(
                    f"{X_API}/users/{user_id}/tweets",
                    params={
                        "max_results": 10,
                        "start_time": start_time,
                        "exclude": "replies,retweets",
                        "tweet.fields": "created_at,public_metrics",
                    },
                    use_cache=True,
                    headers=headers,
                )
            except ProviderError as exc:
                # One suspended, protected, or rate-limited account must not
                # discard successfully fetched accounts.
                log.warning("X community account @%s unavailable: %s", username, exc)
                continue
            for item in payload.get("data", []) if isinstance(payload, dict) else []:
                metrics = item.get("public_metrics") or {}
                engagement = sum(int(metrics.get(k, 0) or 0) for k in (
                    "like_count", "retweet_count", "repost_count", "quote_count"
                ))
                post_id = str(item.get("id", ""))
                post = parse_post(
                    platform="x",
                    source_id=post_id,
                    author=username,
                    text=str(item.get("text", "")),
                    created_at=str(item.get("created_at") or
                                   dt.datetime.now(dt.timezone.utc).isoformat()),
                    url=f"https://x.com/{username}/status/{post_id}" if post_id else None,
                    universe=self.universe,
                    engagement=engagement,
                )
                if post:
                    posts.append(post)
        return _dedupe(posts)


def _discord_text(message: dict) -> str:
    parts = [str(message.get("content") or "")]
    for embed in message.get("embeds", []) or []:
        parts.extend(str(embed.get(k) or "") for k in ("title", "description", "url"))
        for field in embed.get("fields", []) or []:
            parts.extend((str(field.get("name") or ""), str(field.get("value") or "")))
    return "\n".join(p for p in parts if p).strip()


def _dedupe(posts: Iterable[CommunityPost]) -> list[CommunityPost]:
    seen: set[tuple[str, str]] = set()
    out: list[CommunityPost] = []
    for post in sorted(posts, key=lambda p: p.created_at, reverse=True):
        key = (post.platform, post.source_id)
        if key not in seen:
            seen.add(key)
            out.append(post)
    return out
