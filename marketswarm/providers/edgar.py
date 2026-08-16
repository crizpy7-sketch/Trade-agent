"""SEC EDGAR — primary-source filings and insider transactions.

EDGAR is the highest-reliability source the agent has: it is the company's own
filing, timestamped by the regulator. An 8-K that a newswire has not yet
picked up is the closest thing to an information edge available legally and
free.

SEC requires a declared User-Agent with contact info and enforces 10 req/s.
Set MARKETSWARM_CONTACT so requests are not throttled or blocked.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field

from .base import DataClient, ProviderError

log = logging.getLogger("marketswarm.edgar")

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"
FULL_TEXT = "https://efts.sec.gov/LATEST/search-index?q={q}"
RECENT_FILINGS = "https://www.sec.gov/cgi-bin/browse-edgar"

# 8-K items that actually move a stock, mapped to plain English.
ITEM_MEANINGS = {
    "1.01": "material definitive agreement",
    "1.03": "bankruptcy or receivership",
    "2.01": "completion of acquisition/disposition",
    "2.02": "results of operations (earnings release)",
    "2.05": "costs associated with exit/disposal (restructuring)",
    "2.06": "material impairment",
    "3.01": "delisting notice / listing rule failure",
    "4.01": "change in certifying accountant",
    "4.02": "non-reliance on previously issued financials (restatement)",
    "5.02": "departure/appointment of directors or principal officers",
    "7.01": "Regulation FD disclosure",
    "8.01": "other events",
}

HIGH_IMPACT_ITEMS = {"1.03", "2.02", "2.06", "3.01", "4.02", "5.02"}
HIGH_IMPACT_FORMS = {"8-K", "SC 13D", "SC 13D/A", "4", "424B5", "S-3", "NT 10-K", "NT 10-Q"}


@dataclass
class Filing:
    ticker: str
    company: str
    form: str
    filed_at: dt.datetime
    accession: str
    url: str
    items: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def age_hours(self) -> float:
        return (dt.datetime.now(dt.timezone.utc) - self.filed_at).total_seconds() / 3600

    @property
    def impact(self) -> str:
        if self.form == "8-K" and any(i in HIGH_IMPACT_ITEMS for i in self.items):
            return "high"
        if self.form in HIGH_IMPACT_FORMS:
            return "high" if self.form != "4" else "medium"
        return "low"

    def plain_english(self) -> str:
        if self.items:
            meanings = [ITEM_MEANINGS.get(i, f"item {i}") for i in self.items]
            return f"{self.form}: {', '.join(meanings)}"
        return f"{self.form} filing"


class EdgarData:
    def __init__(self, client: DataClient, contact: str | None = None):
        self.client = client
        self.headers = {
            "User-Agent": contact or "marketswarm research (set MARKETSWARM_CONTACT)",
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }
        self._cik_cache: dict[str, str] = {}

    async def cik_for(self, ticker: str) -> str | None:
        if not self._cik_cache:
            try:
                payload = await self.client.get_json(
                    TICKER_MAP, headers={"User-Agent": self.headers["User-Agent"]}
                )
                for row in payload.values():
                    self._cik_cache[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
            except (ProviderError, KeyError, TypeError, AttributeError) as exc:
                log.warning("EDGAR ticker map unavailable: %s", exc)
                return None
        return self._cik_cache.get(ticker.upper())

    async def recent_filings(self, ticker: str, lookback_hours: float = 24.0, limit: int = 15) -> list[Filing]:
        cik = await self.cik_for(ticker)
        if not cik:
            return []
        try:
            payload = await self.client.get_json(SUBMISSIONS.format(cik=cik), headers=self.headers)
        except ProviderError as exc:
            log.warning("EDGAR submissions %s unavailable: %s", ticker, exc)
            return []

        recent = payload.get("filings", {}).get("recent", {})
        company = payload.get("name", ticker)
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        times = recent.get("acceptanceDateTime", [])
        accs = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])
        items = recent.get("items", [])

        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=lookback_hours)
        out: list[Filing] = []
        for i in range(min(len(forms), limit * 4)):
            try:
                stamp = times[i] if i < len(times) and times[i] else f"{dates[i]}T12:00:00.000Z"
                filed = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except (ValueError, IndexError, TypeError):
                continue
            if filed < cutoff:
                continue
            acc_clean = accs[i].replace("-", "") if i < len(accs) else ""
            doc = docs[i] if i < len(docs) else ""
            out.append(
                Filing(
                    ticker=ticker.upper(),
                    company=company,
                    form=forms[i],
                    filed_at=filed,
                    accession=accs[i] if i < len(accs) else "",
                    url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}",
                    items=[x.strip() for x in (items[i].split(",") if i < len(items) and items[i] else []) if x.strip()],
                )
            )
            if len(out) >= limit:
                break
        return out

    async def scan_universe(self, tickers: list[str], lookback_hours: float = 24.0) -> list[Filing]:
        """Overnight filing sweep across the watchlist, most material first."""
        batches = await self.client.gather(
            [self.recent_filings(t, lookback_hours) for t in tickers], label="edgar"
        )
        all_f: list[Filing] = []
        for b in batches:
            if b:
                all_f.extend(b)
        rank = {"high": 0, "medium": 1, "low": 2}
        all_f.sort(key=lambda f: (rank[f.impact], f.age_hours))
        return all_f

    async def insider_transactions(self, ticker: str, lookback_days: int = 30) -> dict:
        """Form 4 activity — the only legally disclosed view of what the people
        running the company are doing with their own money.

        Cluster buying by multiple officers is one of the few insider signals
        with documented predictive value; routine 10b5-1 sales are noise and are
        flagged as such.
        """
        filings = await self.recent_filings(ticker, lookback_hours=lookback_days * 24, limit=40)
        form4 = [f for f in filings if f.form.startswith("4")]
        if not form4:
            return {"ticker": ticker, "count": 0, "signal": "no recent Form 4 activity"}
        return {
            "ticker": ticker,
            "count": len(form4),
            "most_recent": form4[0].filed_at.isoformat(),
            "urls": [f.url for f in form4[:5]],
            "signal": (
                f"{len(form4)} Form 4 filings in {lookback_days}d — cluster activity worth reading"
                if len(form4) >= 4
                else f"{len(form4)} Form 4 filings in {lookback_days}d — routine"
            ),
            "caveat": "Form 4 shows transaction type but not motive; 10b5-1 plan sales are pre-scheduled and carry no signal.",
        }

    async def institutional_13f_hint(self, ticker: str) -> dict:
        """13F/13D context.

        13F holdings are filed 45 days after quarter-end, so they describe the
        past, not the present — the agent reports them as background, never as
        a same-day catalyst. 13D/G filings are the timely ones: they signal an
        activist or >5% stake within 10 days of crossing the threshold.
        """
        filings = await self.recent_filings(ticker, lookback_hours=24 * 90, limit=40)
        stakes = [f for f in filings if f.form.startswith("SC 13")]
        return {
            "ticker": ticker,
            "recent_13d_13g": [
                {"form": f.form, "filed": f.filed_at.date().isoformat(), "url": f.url} for f in stakes[:5]
            ],
            "signal": (
                f"{len(stakes)} beneficial-ownership filings in 90d — a >5% holder is moving"
                if stakes
                else "no recent >5% ownership changes on file"
            ),
            "staleness_warning": "13F data lags by up to 45 days and cannot inform an intraday decision.",
        }


def parse_8k_items(text: str) -> list[str]:
    return sorted(set(re.findall(r"Item\s+(\d\.\d{2})", text)))
