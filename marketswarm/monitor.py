"""Intraday invalidation monitor.

The 8:15 report is stale by 9:35. Every published idea carries explicit
invalidation conditions, and until now nothing checked them.

This runs during the session, re-reads the tape, and reports each live idea as
one of: working, invalidated, target hit, or stopped. Being told "the NVDA
thesis died at 9:41 when it lost the pre-market low" is worth more than the
morning report that proposed it.

It never places or cancels an order. It watches and tells you.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from dataclasses import dataclass, field
from enum import Enum

from . import clock
from .memory import MemoryStore
from .providers.base import DataClient
from .providers.market import MarketData

log = logging.getLogger("marketswarm.monitor")


class IdeaState(str, Enum):
    PENDING = "pending"          # entry not yet touched
    WORKING = "working"          # live and intact
    TARGET_HIT = "target_hit"
    STOPPED = "stopped"
    INVALIDATED = "invalidated"  # a stated condition triggered before either barrier
    EXPIRED = "expired"          # session ended flat


@dataclass
class IdeaStatus:
    prediction_id: int
    symbol: str
    direction: str
    entry: float
    target: float
    stop: float
    state: IdeaState
    last_price: float
    excursion_r: float           # current P&L in R
    max_favorable_r: float
    max_adverse_r: float
    reasons: list[str] = field(default_factory=list)
    checked_at: str = ""

    @property
    def resolved(self) -> bool:
        return self.state in (IdeaState.TARGET_HIT, IdeaState.STOPPED,
                              IdeaState.INVALIDATED, IdeaState.EXPIRED)

    def line(self) -> str:
        icon = {
            IdeaState.PENDING: "·", IdeaState.WORKING: "→",
            IdeaState.TARGET_HIT: "✓", IdeaState.STOPPED: "✗",
            IdeaState.INVALIDATED: "!", IdeaState.EXPIRED: "—",
        }[self.state]
        base = (f"{icon} {self.symbol} {self.direction} @{self.entry:.2f} "
                f"→{self.target:.2f} ⊥{self.stop:.2f} | now {self.last_price:.2f} "
                f"({self.excursion_r:+.2f}R, MFE {self.max_favorable_r:+.2f}R, "
                f"MAE {self.max_adverse_r:+.2f}R) | {self.state.value}")
        if self.reasons:
            base += "\n    " + "\n    ".join(self.reasons)
        return base


class InvalidationMonitor:
    """Checks today's published ideas against live prices."""

    def __init__(self, store: MemoryStore, vix_spike_pct: float = 10.0):
        self.store = store
        self.vix_spike_pct = vix_spike_pct
        self._peak: dict[int, float] = {}
        self._trough: dict[int, float] = {}
        self._announced: dict[int, IdeaState] = {}

    def todays_ideas(self, run_date: dt.date | None = None) -> list:
        d = (run_date or clock.now_et().date()).isoformat()
        return list(self.store.conn.execute(
            "SELECT * FROM predictions WHERE run_date = ? AND resolved = 0 "
            "AND kind IN ('call_idea','put_idea','stock_setup') AND entry IS NOT NULL",
            (d,),
        ))

    async def check(self, run_date: dt.date | None = None) -> list[IdeaStatus]:
        rows = self.todays_ideas(run_date)
        if not rows:
            return []

        symbols = sorted({r["symbol"] for r in rows})
        async with DataClient(cache_ttl=20) as client:
            market = MarketData(client)
            quotes = await market.quotes({s: s for s in symbols})
            vix = await market.quote("^VIX", "VIX")
            profiles = await client.gather(
                [market.premarket_profile(s) for s in symbols], label="monitor"
            )
        pm = {p["symbol"]: p for p in profiles if p}

        vix_spiking = bool(vix and vix.change_pct >= self.vix_spike_pct)
        session = clock.session_at()

        out: list[IdeaStatus] = []
        for r in rows:
            q = quotes.get(r["symbol"])
            if not q:
                continue
            out.append(self._evaluate(r, q.price, pm.get(r["symbol"]), vix, vix_spiking, session))
        return out

    def _evaluate(self, row, price: float, profile: dict | None, vix, vix_spiking: bool,
                  session: clock.Session) -> IdeaStatus:
        pid = row["id"]
        entry, target, stop = row["entry"], row["target"], row["stop"]
        long_side = row["direction"] == "long"
        risk = abs(entry - stop) or 1e-9

        peak = self._peak.get(pid, price)
        trough = self._trough.get(pid, price)
        self._peak[pid] = max(peak, price)
        self._trough[pid] = min(trough, price)

        excursion = ((price - entry) if long_side else (entry - price)) / risk
        mfe = ((self._peak[pid] - entry) if long_side else (entry - self._trough[pid])) / risk
        mae = ((self._trough[pid] - entry) if long_side else (entry - self._peak[pid])) / risk

        reasons: list[str] = []
        state = IdeaState.WORKING

        if long_side and price >= target:
            state = IdeaState.TARGET_HIT
            reasons.append(f"target {target:.2f} reached")
        elif not long_side and price <= target:
            state = IdeaState.TARGET_HIT
            reasons.append(f"target {target:.2f} reached")
        elif long_side and price <= stop:
            state = IdeaState.STOPPED
            reasons.append(f"stop {stop:.2f} breached")
        elif not long_side and price >= stop:
            state = IdeaState.STOPPED
            reasons.append(f"stop {stop:.2f} breached")
        else:
            # Structural invalidations — the conditions the morning report stated.
            if vix_spiking:
                state = IdeaState.INVALIDATED
                reasons.append(
                    f"VIX {vix.change_pct:+.1f}% intraday exceeds the {self.vix_spike_pct:.0f}% "
                    f"regime-change threshold; the setup's base rates no longer apply"
                )
            if profile:
                pmh, pml = profile.get("premarket_high"), profile.get("premarket_low")
                if long_side and pmh and price < pmh and session == clock.Session.REGULAR:
                    reasons.append(
                        f"trading below the pre-market high {pmh:.2f} — the stated "
                        f"'hold above' condition is not met"
                    )
                if not long_side and pml and price > pml and session == clock.Session.REGULAR:
                    reasons.append(
                        f"trading above the pre-market low {pml:.2f} — the stated "
                        f"'break below' condition is not met"
                    )

            if session in (clock.Session.AFTERHOURS, clock.Session.POST_CLOSE,
                           clock.Session.OVERNIGHT):
                state = IdeaState.EXPIRED
                reasons.append(f"session over; marked to {price:.2f} ({excursion:+.2f}R)")

        return IdeaStatus(
            prediction_id=pid, symbol=row["symbol"], direction=row["direction"],
            entry=entry, target=target, stop=stop, state=state, last_price=price,
            excursion_r=excursion, max_favorable_r=mfe, max_adverse_r=mae,
            reasons=reasons, checked_at=clock.now_et().isoformat(),
        )

    def newly_changed(self, statuses: list[IdeaStatus]) -> list[IdeaStatus]:
        """Only what changed since the last check — so a webhook does not
        re-announce the same thing every minute."""
        changed = []
        for s in statuses:
            if self._announced.get(s.prediction_id) != s.state:
                changed.append(s)
                self._announced[s.prediction_id] = s.state
        return changed

    def persist_resolutions(self, statuses: list[IdeaStatus]) -> int:
        """Close out finished ideas so the nightly scoring pass has less to do
        and the learning loop sees the same resolution the monitor saw."""
        n = 0
        for s in statuses:
            if s.state == IdeaState.TARGET_HIT:
                r = abs(s.target - s.entry) / max(abs(s.entry - s.stop), 1e-9)
                self.store.resolve(s.prediction_id, 1, r, "target reached (intraday monitor)")
                n += 1
            elif s.state == IdeaState.STOPPED:
                self.store.resolve(s.prediction_id, 0, -1.0, "stopped out (intraday monitor)")
                n += 1
            elif s.state == IdeaState.INVALIDATED:
                self.store.resolve(s.prediction_id, 0 if s.excursion_r <= 0 else 1,
                                   s.excursion_r, "; ".join(s.reasons)[:400])
                n += 1
            elif s.state == IdeaState.EXPIRED:
                self.store.resolve(s.prediction_id, 1 if s.excursion_r > 0 else 0,
                                   s.excursion_r, "session close (intraday monitor)")
                n += 1
        return n


async def watch(store: MemoryStore, interval_seconds: int = 300,
                webhook: str | None = None, once: bool = False) -> None:
    """Poll until the session ends."""
    from .notify import send_webhook

    monitor = InvalidationMonitor(store)
    while True:
        session = clock.session_at()
        if session not in (clock.Session.REGULAR, clock.Session.PREMARKET):
            log.info("session is %s — monitor idle", session.value)
            if once:
                return
            await asyncio.sleep(min(interval_seconds * 4, 1800))
            continue

        try:
            statuses = await monitor.check()
        except Exception:  # noqa: BLE001 — a bad poll must not end the watch
            log.exception("monitor check failed")
            statuses = []

        if statuses:
            print(f"\n[{clock.now_et():%H:%M:%S ET}] {len(statuses)} live ideas")
            for s in statuses:
                print("  " + s.line())

            changed = monitor.newly_changed(statuses)
            resolved = [s for s in changed if s.resolved]
            if resolved:
                monitor.persist_resolutions(resolved)
                if webhook:
                    body = "\n".join(f"• {s.symbol} {s.direction}: {s.state.value} "
                                     f"({s.excursion_r:+.2f}R) — {'; '.join(s.reasons)}"
                                     for s in resolved)
                    send_webhook(webhook, f"**Idea status change**\n{body}\n\n"
                                          f"_Research only — not financial advice._")
        else:
            log.info("no live ideas to monitor")

        if once:
            return
        await asyncio.sleep(interval_seconds)
