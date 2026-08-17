"""The Event Brain — cheap broad scan, then targeted depth.

1.x spent identical effort on every symbol every session. A quiet AAPL cost the
same as an NVDA earnings gap, which is both wasteful and backwards: the whole
point of attention is that it is scarce.

This module runs a cheap pass over everything, classifies what it finds, and
hands the Chief Investigator a priority-ordered list of what actually deserves
work. Detection is deterministic and threshold-driven — an LLM adds nothing to
"is this gap larger than 2 ATR" and would make it slower, costlier and
non-reproducible.

All thresholds are configurable. None of them are hidden inside one giant
function, because they will need tuning and tuning requires finding them.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from enum import Enum


class EventPriority(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return {"LOW": 0, "NORMAL": 1, "ELEVATED": 2, "HIGH": 3, "CRITICAL": 4}[self.value]


class EventType(str, Enum):
    PRICE_GAP = "price_gap"
    RELATIVE_VOLUME = "relative_volume"
    VOLATILITY_SPIKE = "volatility_spike"
    INDEX_MOVE = "index_move"
    EARNINGS_REACTION = "earnings_reaction"
    EARNINGS_UPCOMING = "earnings_upcoming"
    GUIDANCE = "guidance"
    SEC_FILING = "sec_filing"
    MACRO_RELEASE = "macro_release"
    OPTIONS_ANOMALY = "options_anomaly"
    ANALYST_ACTION = "analyst_action"
    SECTOR_DIVERGENCE = "sector_divergence"
    COMPANY_NEWS = "company_news"
    GEOPOLITICAL = "geopolitical"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    QUIET = "quiet"


@dataclass
class TriggerThresholds:
    """Every number the brain fires on. Tune here, not in the logic."""

    # gaps, in ATR multiples and in percent — both, because a 2% gap on a
    # 4%-ATR name is noise and on a 1%-ATR name is an event
    gap_atr_elevated: float = 1.0
    gap_atr_high: float = 2.0
    gap_atr_critical: float = 3.5
    gap_pct_floor: float = 0.75          # below this, never fire regardless of ATR

    rvol_elevated: float = 1.8
    rvol_high: float = 3.0
    rvol_critical: float = 6.0

    vix_change_elevated: float = 6.0     # percent
    vix_change_high: float = 12.0
    vix_change_critical: float = 25.0
    vix_absolute_high: float = 28.0

    index_move_elevated: float = 0.5     # percent on ES/SPY
    index_move_high: float = 1.2
    index_move_critical: float = 2.5

    pc_ratio_extreme_low: float = 0.55
    pc_ratio_extreme_high: float = 1.60
    unusual_turnover: float = 3.0        # option volume / open interest
    unusual_notional: float = 1_000_000

    sector_divergence_pct: float = 1.5   # symbol vs its index, same session
    correlation_break_z: float = 2.5

    news_high_materiality_corroborations: int = 2
    filing_high_impact_forms: tuple = ("8-K", "SC 13D", "4/A", "NT 10-K", "NT 10-Q")


@dataclass
class DetectedEvent:
    subject: str
    event_type: EventType
    priority: EventPriority
    description: str
    magnitude: float | None = None
    evidence: dict = field(default_factory=dict)
    detected_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    suggested_hypotheses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "event_type": self.event_type.value,
            "priority": self.priority.value,
            "description": self.description,
            "magnitude": self.magnitude,
            "evidence": self.evidence,
            "detected_at": self.detected_at,
            "hypotheses": self.suggested_hypotheses,
        }


# Hypotheses proposed per event type. Deterministic scaffolding: the Chief
# Investigator narrows and the specialists test, but the candidate list does not
# need a model to enumerate.
HYPOTHESES: dict[EventType, list[str]] = {
    EventType.PRICE_GAP: [
        "company-specific news broke overnight",
        "an SEC filing was made after the close",
        "earnings were reported and the market is repricing",
        "an analyst action moved it",
        "the whole sector is moving and this is beta, not alpha",
        "a macro catalyst repriced the entire market",
        "a supplier, customer or peer reported something material",
        "index rebalancing or a large mechanical flow",
        "no identifiable catalyst — likely liquidity, and likely to revert",
    ],
    EventType.VOLATILITY_SPIKE: [
        "a macro release or scheduled event is imminent",
        "a geopolitical catalyst emerged",
        "systematic de-risking or a liquidation is under way",
        "options positioning is forcing dealer hedging",
    ],
    EventType.EARNINGS_REACTION: [
        "the headline number drove the move",
        "guidance contradicted the headline",
        "positioning into the print dominated the reaction",
        "a segment detail or margin figure mattered more than EPS",
    ],
    EventType.OPTIONS_ANOMALY: [
        "informed positioning ahead of a catalyst",
        "a hedge against an existing equity position, carrying no directional view",
        "a dealer hedging flow rather than a speculative view",
        "a closing trade being misread as an opening one",
    ],
    EventType.SEC_FILING: [
        "a material event was disclosed that the tape has not absorbed",
        "a routine filing with no information content",
        "insider or >5% holder activity",
    ],
    EventType.MACRO_RELEASE: [
        "the print will reprice rate expectations",
        "the market has already discounted the expected value",
        "the reaction will reverse once the initial impulse clears",
    ],
    EventType.SECTOR_DIVERGENCE: [
        "company-specific information is driving the divergence",
        "a rotation is under way at the sector level",
        "it is noise and will converge intraday",
    ],
    EventType.CORRELATION_BREAKDOWN: [
        "a regime change is in progress",
        "an idiosyncratic shock hit one leg",
        "the historical relationship has genuinely broken",
    ],
}


class EventBrain:
    """Cheap scan → classified events. No network, no LLM, no side effects."""

    def __init__(self, thresholds: TriggerThresholds | None = None):
        self.t = thresholds or TriggerThresholds()

    # ---------- individual detectors ----------

    def detect_gap(self, symbol: str, gap_pct: float | None,
                   atr_pct: float | None) -> DetectedEvent | None:
        if gap_pct is None or abs(gap_pct) < self.t.gap_pct_floor:
            return None
        # Normalise by the name's own volatility; a raw percentage compares
        # nothing meaningful across symbols.
        atr = atr_pct if atr_pct and atr_pct > 0 else 1.5
        z = abs(gap_pct) / atr

        if z >= self.t.gap_atr_critical:
            pri = EventPriority.CRITICAL
        elif z >= self.t.gap_atr_high:
            pri = EventPriority.HIGH
        elif z >= self.t.gap_atr_elevated:
            pri = EventPriority.ELEVATED
        else:
            return None

        return DetectedEvent(
            subject=symbol, event_type=EventType.PRICE_GAP, priority=pri,
            description=(f"{symbol} gapping {gap_pct:+.2f}%, {z:.1f}× its "
                         f"{atr:.2f}% average true range"),
            magnitude=z,
            evidence={"gap_pct": gap_pct, "atr_pct": atr, "atr_multiple": round(z, 2)},
            suggested_hypotheses=list(HYPOTHESES[EventType.PRICE_GAP]),
        )

    def detect_relative_volume(self, symbol: str, rvol: float | None) -> DetectedEvent | None:
        if rvol is None or math.isnan(rvol) or rvol < self.t.rvol_elevated:
            return None
        pri = (EventPriority.CRITICAL if rvol >= self.t.rvol_critical
               else EventPriority.HIGH if rvol >= self.t.rvol_high
               else EventPriority.ELEVATED)
        return DetectedEvent(
            subject=symbol, event_type=EventType.RELATIVE_VOLUME, priority=pri,
            description=f"{symbol} trading at {rvol:.1f}× normal volume",
            magnitude=rvol, evidence={"rvol": rvol},
            suggested_hypotheses=["someone knows something",
                                  "index or mechanical flow",
                                  "news the scan has not yet found"],
        )

    def detect_volatility(self, vix: float | None,
                          vix_change_pct: float | None) -> DetectedEvent | None:
        if vix is None and vix_change_pct is None:
            return None
        chg = abs(vix_change_pct or 0.0)
        pri = None
        if chg >= self.t.vix_change_critical or (vix and vix >= self.t.vix_absolute_high * 1.4):
            pri = EventPriority.CRITICAL
        elif chg >= self.t.vix_change_high or (vix and vix >= self.t.vix_absolute_high):
            pri = EventPriority.HIGH
        elif chg >= self.t.vix_change_elevated:
            pri = EventPriority.ELEVATED
        if pri is None:
            return None
        return DetectedEvent(
            subject="MARKET", event_type=EventType.VOLATILITY_SPIKE, priority=pri,
            description=(f"VIX {vix:.1f} ({vix_change_pct:+.1f}%)" if vix is not None
                         else f"VIX {vix_change_pct:+.1f}%"),
            magnitude=chg, evidence={"vix": vix, "vix_change_pct": vix_change_pct},
            suggested_hypotheses=list(HYPOTHESES[EventType.VOLATILITY_SPIKE]),
        )

    def detect_index_move(self, index_pct: float | None) -> DetectedEvent | None:
        if index_pct is None or abs(index_pct) < self.t.index_move_elevated:
            return None
        a = abs(index_pct)
        pri = (EventPriority.CRITICAL if a >= self.t.index_move_critical
               else EventPriority.HIGH if a >= self.t.index_move_high
               else EventPriority.ELEVATED)
        return DetectedEvent(
            subject="MARKET", event_type=EventType.INDEX_MOVE, priority=pri,
            description=f"index futures {index_pct:+.2f}% pre-open",
            magnitude=a, evidence={"index_pct": index_pct},
            suggested_hypotheses=["macro catalyst", "overnight global session",
                                  "large scheduled flow"],
        )

    def detect_earnings(self, symbol: str, reacting: bool,
                        surprise_pct: float | None = None,
                        reporting_tonight: bool = False) -> DetectedEvent | None:
        if reacting:
            mag = abs(surprise_pct) if surprise_pct is not None else 0.0
            pri = EventPriority.HIGH if mag >= 5 else EventPriority.ELEVATED
            return DetectedEvent(
                subject=symbol, event_type=EventType.EARNINGS_REACTION, priority=pri,
                description=(f"{symbol} trading on results"
                             + (f", EPS surprise {surprise_pct:+.1f}%"
                                if surprise_pct is not None else "")),
                magnitude=mag, evidence={"surprise_pct": surprise_pct},
                suggested_hypotheses=list(HYPOTHESES[EventType.EARNINGS_REACTION]),
            )
        if reporting_tonight:
            return DetectedEvent(
                subject=symbol, event_type=EventType.EARNINGS_UPCOMING,
                priority=EventPriority.ELEVATED,
                description=f"{symbol} reports after today's close — IV inflated into the event",
                evidence={"timing": "amc"},
                suggested_hypotheses=["premium is rich and will crush",
                                      "price compresses into the print"],
            )
        return None

    def detect_filing(self, symbol: str, form: str, items: list[str],
                      age_hours: float) -> DetectedEvent | None:
        high_items = {"1.03", "2.02", "2.06", "3.01", "4.02", "5.02"}
        is_high = form in self.t.filing_high_impact_forms and (
            not items or any(i in high_items for i in items))
        if not is_high and form not in self.t.filing_high_impact_forms:
            return None
        pri = EventPriority.HIGH if is_high and age_hours <= 24 else EventPriority.ELEVATED
        return DetectedEvent(
            subject=symbol, event_type=EventType.SEC_FILING, priority=pri,
            description=(f"{symbol} filed {form}"
                         + (f" (items {', '.join(items)})" if items else "")
                         + f" {age_hours:.0f}h ago"),
            evidence={"form": form, "items": items, "age_hours": age_hours},
            suggested_hypotheses=list(HYPOTHESES[EventType.SEC_FILING]),
        )

    def detect_options_anomaly(self, symbol: str, put_call: float | None,
                               unusual: list[dict] | None) -> DetectedEvent | None:
        reasons, mag = [], 0.0
        if put_call is not None and not math.isnan(put_call):
            if put_call >= self.t.pc_ratio_extreme_high:
                reasons.append(f"put/call {put_call:.2f} — heavy hedging")
                mag = max(mag, put_call)
            elif put_call <= self.t.pc_ratio_extreme_low:
                reasons.append(f"put/call {put_call:.2f} — call-heavy")
                mag = max(mag, 1 / max(put_call, 0.01))

        big = [u for u in (unusual or [])
               if u.get("turnover", 0) >= self.t.unusual_turnover
               and u.get("notional_estimate", 0) >= self.t.unusual_notional]
        if big:
            top = max(big, key=lambda u: u.get("notional_estimate", 0))
            reasons.append(
                f"{top['volume']:,} {top['kind']}s at {top['strike']:g} on "
                f"{top['open_interest']:,} OI (~${top['notional_estimate']:,.0f})")
            mag = max(mag, float(top.get("turnover", 0)))

        if not reasons:
            return None
        pri = EventPriority.HIGH if len(reasons) > 1 or mag >= 6 else EventPriority.ELEVATED
        return DetectedEvent(
            subject=symbol, event_type=EventType.OPTIONS_ANOMALY, priority=pri,
            description=f"{symbol} unusual option activity: " + "; ".join(reasons),
            magnitude=mag,
            evidence={"put_call": put_call, "unusual": big[:3],
                      "inferred": True,
                      "caveat": "derived from volume/open-interest, not observed trade flow"},
            suggested_hypotheses=list(HYPOTHESES[EventType.OPTIONS_ANOMALY]),
        )

    def detect_sector_divergence(self, symbol: str, symbol_pct: float | None,
                                 index_pct: float | None) -> DetectedEvent | None:
        if symbol_pct is None or index_pct is None:
            return None
        spread = symbol_pct - index_pct
        if abs(spread) < self.t.sector_divergence_pct:
            return None
        return DetectedEvent(
            subject=symbol, event_type=EventType.SECTOR_DIVERGENCE,
            priority=EventPriority.ELEVATED,
            description=(f"{symbol} {symbol_pct:+.2f}% versus the index {index_pct:+.2f}% "
                         f"— {spread:+.2f}pp of idiosyncratic move"),
            magnitude=abs(spread),
            evidence={"symbol_pct": symbol_pct, "index_pct": index_pct, "spread": spread},
            suggested_hypotheses=list(HYPOTHESES[EventType.SECTOR_DIVERGENCE]),
        )

    def detect_macro(self, events: list[dict]) -> list[DetectedEvent]:
        out = []
        for e in events or []:
            impact = str(e.get("impact", "")).lower()
            if impact not in ("high", "very high"):
                continue
            out.append(DetectedEvent(
                subject="MARKET", event_type=EventType.MACRO_RELEASE,
                priority=(EventPriority.CRITICAL if impact == "very high"
                          else EventPriority.HIGH),
                description=(f"{e.get('name')} at {e.get('time_et')} ET "
                             f"({'pre-open' if e.get('before_open') else 'intraday'})"),
                evidence=dict(e),
                suggested_hypotheses=list(HYPOTHESES[EventType.MACRO_RELEASE]),
            ))
        return out

    def detect_news(self, symbol: str, headlines: list[dict]) -> DetectedEvent | None:
        material = [h for h in headlines or []
                    if h.get("materiality") == "high"
                    and h.get("corroborations", 1) >= self.t.news_high_materiality_corroborations]
        if not material:
            return None
        return DetectedEvent(
            subject=symbol, event_type=EventType.COMPANY_NEWS,
            priority=EventPriority.HIGH,
            description=f"{symbol}: {material[0].get('title', '')[:120]}",
            evidence={"headlines": material[:3]},
            suggested_hypotheses=["the market has not fully absorbed it",
                                  "already priced in the gap",
                                  "the headline overstates the substance"],
        )

    # ---------- the scan ----------

    def scan(self, snapshot: dict) -> list[DetectedEvent]:
        """Run every detector over a cheap market snapshot.

        `snapshot` shape (all keys optional — missing data means no detection,
        never a fabricated one):
            symbols: {sym: {gap_pct, atr_pct, rvol, put_call, unusual,
                            earnings_reacting, surprise_pct, reporting_tonight,
                            filings: [{form, items, age_hours}], headlines: [...]}}
            market:  {vix, vix_change_pct, index_pct, econ_events: [...]}
        """
        events: list[DetectedEvent] = []
        market = snapshot.get("market", {}) or {}

        v = self.detect_volatility(market.get("vix"), market.get("vix_change_pct"))
        if v:
            events.append(v)
        im = self.detect_index_move(market.get("index_pct"))
        if im:
            events.append(im)
        events.extend(self.detect_macro(market.get("econ_events", [])))

        index_pct = market.get("index_pct")

        for sym, d in (snapshot.get("symbols", {}) or {}).items():
            d = d or {}
            for ev in (
                self.detect_gap(sym, d.get("gap_pct"), d.get("atr_pct")),
                self.detect_relative_volume(sym, d.get("rvol")),
                self.detect_earnings(sym, bool(d.get("earnings_reacting")),
                                     d.get("surprise_pct"),
                                     bool(d.get("reporting_tonight"))),
                self.detect_options_anomaly(sym, d.get("put_call"), d.get("unusual")),
                self.detect_sector_divergence(sym, d.get("gap_pct"), index_pct),
                self.detect_news(sym, d.get("headlines", [])),
            ):
                if ev:
                    events.append(ev)

            for f in d.get("filings", []) or []:
                ev = self.detect_filing(sym, f.get("form", ""), f.get("items", []) or [],
                                        float(f.get("age_hours", 999)))
                if ev:
                    events.append(ev)

        events.sort(key=lambda e: (-e.priority.rank, -(e.magnitude or 0)))
        return events

    def quiet_symbols(self, snapshot: dict, events: list[DetectedEvent]) -> list[str]:
        """Names with nothing going on. They still get a cheap look — absence of
        a detected event is not a guarantee that nothing is happening."""
        noisy = {e.subject for e in events}
        return [s for s in (snapshot.get("symbols", {}) or {}) if s not in noisy]

    def triage(self, snapshot: dict) -> dict:
        events = self.scan(snapshot)
        by_subject: dict[str, list[DetectedEvent]] = {}
        for e in events:
            by_subject.setdefault(e.subject, []).append(e)

        priorities = {
            subj: max(evs, key=lambda x: x.priority.rank).priority
            for subj, evs in by_subject.items()
        }
        return {
            "events": events,
            "by_subject": by_subject,
            "priority_by_subject": priorities,
            "quiet": self.quiet_symbols(snapshot, events),
            "n_events": len(events),
            "highest": (max(events, key=lambda e: e.priority.rank).priority
                        if events else EventPriority.LOW),
        }
