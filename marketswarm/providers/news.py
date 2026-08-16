"""Breaking news and headline ingestion via public RSS.

No API keys required. Headlines are deduplicated across outlets, scored for
materiality, and tagged with source reliability. Corroboration across
independent outlets is what the cross-verification agent later turns into
confidence — a single-outlet claim never gets treated as established fact.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .base import DataClient, Evidence, ProviderError

log = logging.getLogger("marketswarm.news")

FEEDS: list[tuple[str, str, str, float]] = [
    # (name, url, source_class, reliability)
    ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews", "major_newswire", 0.85),
    ("Reuters Markets", "https://feeds.reuters.com/reuters/USmarketsNews", "major_newswire", 0.85),
    ("AP Business", "https://rsshub.app/apnews/topics/business", "major_newswire", 0.84),
    ("CNBC Top News", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", "financial_media", 0.72),
    ("CNBC Markets", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258", "financial_media", 0.72),
    ("MarketWatch Top", "https://feeds.content.dowjones.io/public/rss/mw_topstories", "financial_media", 0.72),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex", "aggregator", 0.62),
    ("Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml", "federal_reserve", 0.98),
    ("SEC Press", "https://www.sec.gov/news/pressreleases.rss", "sec_edgar", 0.97),
    ("Treasury", "https://home.treasury.gov/system/files/126/ofac.xml", "treasury", 0.95),
]

GOOGLE_NEWS = "https://news.google.com/rss/search"

# Words that separate a market-moving headline from newsroom filler.
MATERIAL_TERMS = {
    "high": [
        "guidance", "downgrade", "upgrade", "beats", "misses", "warns", "cuts outlook",
        "raises outlook", "acquisition", "merger", "bankruptcy", "recall", "halted",
        "investigation", "lawsuit", "fda approval", "sec charges", "restatement",
        "ceo resigns", "cfo resigns", "layoffs", "profit warning", "buyback", "dividend cut",
        "rate decision", "cpi", "ppi", "payrolls", "fomc", "tariff", "sanctions", "default",
    ],
    "medium": [
        "earnings", "revenue", "outlook", "forecast", "analyst", "price target", "partnership",
        "contract", "launch", "expansion", "stake", "activist", "settlement", "strike",
    ],
}

TICKER_RE = re.compile(r"\b(?:NYSE|NASDAQ):\s?([A-Z]{1,5})\b|\(([A-Z]{2,5})\)")


@dataclass
class Headline:
    title: str
    source: str
    source_class: str
    reliability: float
    url: str
    published: dt.datetime | None = None
    summary: str = ""
    tickers: list[str] = field(default_factory=list)
    materiality: str = "low"
    corroborations: int = 1

    @property
    def age_minutes(self) -> float:
        if not self.published:
            return 9999.0
        return (dt.datetime.now(dt.timezone.utc) - self.published).total_seconds() / 60

    def to_evidence(self) -> Evidence:
        return Evidence(
            claim=self.title,
            source=self.source,
            url=self.url,
            reliability=self.reliability * (1.0 if self.corroborations == 1 else min(1.0, 0.85 + 0.08 * self.corroborations)),
            observed_at=self.published.isoformat() if self.published else dt.datetime.now(dt.timezone.utc).isoformat(),
            value={"materiality": self.materiality, "corroborations": self.corroborations},
            tags=["news", self.materiality] + self.tickers,
        )


def _score_materiality(text: str) -> str:
    t = text.lower()
    if any(term in t for term in MATERIAL_TERMS["high"]):
        return "high"
    if any(term in t for term in MATERIAL_TERMS["medium"]):
        return "medium"
    return "low"


def _extract_tickers(text: str, universe: set[str] | None = None) -> list[str]:
    found = {m.group(1) or m.group(2) for m in TICKER_RE.finditer(text)}
    found = {f for f in found if f}
    if universe:
        # Bare uppercase words are matched only against a known universe;
        # otherwise every "CEO" and "GDP" becomes a ticker.
        for word in re.findall(r"\b[A-Z]{2,5}\b", text):
            if word in universe:
                found.add(word)
    return sorted(found)


def _parse_time(entry) -> dt.datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        val = getattr(entry, key, None) or (entry.get(key) if isinstance(entry, dict) else None)
        if val:
            try:
                return dt.datetime(*val[:6], tzinfo=dt.timezone.utc)
            except (TypeError, ValueError):
                continue
    return None


def deduplicate(headlines: list[Headline], threshold: float = 0.82) -> list[Headline]:
    """Merge near-identical headlines across outlets and count corroborations.

    Independent confirmation is the single most useful signal about whether a
    headline is real, so it is preserved as a count rather than thrown away.
    """
    kept: list[Headline] = []
    for h in sorted(headlines, key=lambda x: (-x.reliability, x.title)):
        match = None
        for k in kept:
            if SequenceMatcher(None, h.title.lower(), k.title.lower()).ratio() >= threshold:
                match = k
                break
        if match:
            match.corroborations += 1
            if h.reliability > match.reliability:
                match.source, match.url, match.reliability = h.source, h.url, h.reliability
        else:
            kept.append(h)
    return kept


class NewsData:
    def __init__(self, client: DataClient, universe: set[str] | None = None):
        self.client = client
        self.universe = universe or set()

    async def _fetch_feed(self, name: str, url: str, source_class: str, reliability: float) -> list[Headline]:
        try:
            import feedparser
        except ImportError:
            log.error("feedparser not installed — news layer disabled")
            return []
        try:
            raw = await self.client.get_text(url, use_cache=True)
        except ProviderError as exc:
            log.warning("feed %s unavailable: %s", name, exc)
            return []

        parsed = feedparser.parse(raw)
        out: list[Headline] = []
        for e in parsed.entries[:40]:
            title = (getattr(e, "title", "") or "").strip()
            if not title:
                continue
            summary = re.sub(r"<[^>]+>", "", getattr(e, "summary", "") or "")[:400]
            out.append(
                Headline(
                    title=title,
                    source=name,
                    source_class=source_class,
                    reliability=reliability,
                    url=getattr(e, "link", "") or url,
                    published=_parse_time(e),
                    summary=summary,
                    tickers=_extract_tickers(f"{title} {summary}", self.universe),
                    materiality=_score_materiality(f"{title} {summary}"),
                )
            )
        return out

    async def overnight_headlines(self, max_age_hours: float = 18.0) -> list[Headline]:
        batches = await self.client.gather(
            [self._fetch_feed(n, u, c, r) for n, u, c, r in FEEDS], label="news"
        )
        all_h: list[Headline] = []
        for b in batches:
            if b:
                all_h.extend(b)
        fresh = [h for h in all_h if h.age_minutes <= max_age_hours * 60]
        merged = deduplicate(fresh)
        order = {"high": 0, "medium": 1, "low": 2}
        merged.sort(key=lambda h: (order[h.materiality], -h.corroborations, h.age_minutes))
        return merged

    async def ticker_news(self, ticker: str, limit: int = 8) -> list[Headline]:
        """Company-specific headlines via Google News RSS."""
        try:
            import feedparser
        except ImportError:
            return []
        try:
            raw = await self.client.get_text(
                GOOGLE_NEWS,
                params={"q": f"{ticker} stock when:1d", "hl": "en-US", "gl": "US", "ceid": "US:en"},
            )
        except ProviderError as exc:
            log.warning("ticker news %s unavailable: %s", ticker, exc)
            return []

        parsed = feedparser.parse(raw)
        out = []
        for e in parsed.entries[:limit]:
            title = (getattr(e, "title", "") or "").strip()
            if not title:
                continue
            out.append(
                Headline(
                    title=title,
                    source=f"Google News ({getattr(getattr(e, 'source', None), 'title', 'aggregated')})",
                    source_class="aggregator",
                    reliability=0.58,
                    url=getattr(e, "link", ""),
                    published=_parse_time(e),
                    tickers=[ticker],
                    materiality=_score_materiality(title),
                )
            )
        return out
