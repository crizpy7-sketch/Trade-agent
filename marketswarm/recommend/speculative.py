"""The daily contract board: three calls and three puts, every trading day.

The rest of the pipeline is built to refuse. Eleven separate paths can reduce a
day's output to nothing — a coin-flip probability band, three high-severity
review findings, a bracket with no room in it, a failed playbook agent, a
suppressed publication — and on a genuinely quiet session several of them fire
at once. That refusal is correct for a *recommendation*: a forced pick dressed
as conviction is a lie, and the comment at report.py:346 puts it well, "without
turning symmetry into endorsement".

This module exists because silence is not the only honest answer. The owner
wants contracts every day, including days with no edge. Both can be true at
once, but only if filling a slot stops meaning "we like this". So every row
here carries a tier that says exactly how much belief is behind it:

    QUALIFIED    the swarm's own gates passed this. Rare.
    SPECULATIVE  no edge established. Structurally sound, directionally a guess.
    LOTTERY      thin or one-sided data. The contract is real; the thesis is not.
    NO CHAIN     no option data for anything we could screen. Names what failed.

A LOTTERY row is not a recommendation and does not pretend to be. What it is,
is the least stupid version of the gamble: a liquid contract, a real strike, a
real expiry, and the exact number of dollars lost when it expires worthless.

Nothing here trades, routes, or sizes a position. It reads a chain and prints
arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

QUALIFIED = "QUALIFIED"
SPECULATIVE = "SPECULATIVE"
LOTTERY = "LOTTERY"
NO_CHAIN = "NO CHAIN"

#: Slots per side. Three each, matching the existing screening board.
DEFAULT_SLOTS = 3

#: A contract cheaper than this is usually a far-OTM stub whose quoted mid is
#: an artefact of a one-cent bid. Cost is not the objection — a mid that low
#: means the spread is most of the trade.
MIN_PREMIUM = 0.05

#: Above this, the bid/ask spread costs more than a plausible day's move. Not a
#: hard exclusion — it is a tier demotion, because on a bad chain every contract
#: breaches it and excluding them all would put us back to silence.
WIDE_SPREAD_PCT = 12.0

#: Target delta for the speculative pick. Deliberately the same 40-delta the
#: playbook uses (synthesis.py:582): enough delta that a normal move actually
#: moves the option, rather than a cheap lottery ticket that needs a gap.
TARGET_DELTA = 0.40


def tradability(contract) -> float:
    """0-1, honest about a broken quote.

    Chain.liquidity_score treats a NaN spread as a perfect 1.0 (options.py:216).
    A NaN spread means ask <= bid — a crossed or one-sided quote — which is the
    *least* tradable state there is, not the most. Scoring it 1.0 makes the
    worst contracts sort first, so this scores it 0.0 and says why.
    """
    sp = contract.spread_pct
    if math.isnan(sp):
        return 0.0
    spread_score = max(0.0, 1.0 - sp / 15.0)
    depth = min(1.0, (contract.open_interest + contract.volume) / 2_000)
    return round(0.65 * spread_score + 0.35 * depth, 3)


@dataclass
class ContractIdea:
    """One row of the board. Every number here is arithmetic on a live quote."""

    symbol: str
    right: str                      # "call" | "put"
    tier: str
    reason: str
    strike: float | None = None
    expiration: str | None = None
    premium: float | None = None    # per share, the mid
    cost: float | None = None       # one contract, 100 shares
    max_loss: float | None = None   # a long option: the premium. That is the point.
    breakeven: float | None = None
    delta: float | None = None
    spread_pct: float | None = None
    open_interest: int | None = None
    volume: int | None = None
    tradability: float | None = None
    underlying: float | None = None
    days_to_expiry: float | None = None

    @property
    def filled(self) -> bool:
        return self.strike is not None and self.premium is not None

    def to_dict(self) -> dict:
        return asdict(self)


def _price(contract) -> float | None:
    mid = contract.mid
    if mid is None or mid <= 0 or math.isnan(mid):
        return None
    return float(mid)


def _pick(chain, right: str):
    """The 40-delta contract, falling back outward until something is quoted.

    by_delta already filters to positive IV and positive mid (options.py:85) and
    returns None when that empties the pool. ATM is the fallback because a chain
    with no priced 40-delta contract may still have a priced at-the-money one,
    and one real contract beats an empty slot.
    """
    for candidate in (chain.by_delta(TARGET_DELTA, right), chain.atm(right)):
        if candidate is not None and _price(candidate) is not None:
            return candidate
    return None


def _idea_from(chain, contract, right: str, *, qualified: bool) -> ContractIdea:
    premium = _price(contract)
    cost = round(premium * 100, 2)
    breakeven = (contract.strike + premium if right == "call"
                 else contract.strike - premium)

    try:
        delta = chain.greeks_for(contract).get("delta")
    except Exception:  # noqa: BLE001 — greeks are decoration, the quote is not
        delta = None

    score = tradability(contract)
    sp = contract.spread_pct
    wide = math.isnan(sp) or sp > WIDE_SPREAD_PCT
    thin = contract.open_interest < 100 and contract.volume < 50

    if qualified:
        tier = QUALIFIED
        reason = ("Cleared the swarm's own gates today. Sized and reasoned "
                  "elsewhere in this report.")
    elif wide or thin or premium < MIN_PREMIUM:
        tier = LOTTERY
        bits = []
        if math.isnan(sp):
            bits.append("the quote is crossed or one-sided")
        elif sp > WIDE_SPREAD_PCT:
            bits.append(f"the spread is {sp:.1f}% of the mid")
        if thin:
            bits.append(f"only {contract.open_interest:,} OI and {contract.volume:,} traded")
        if premium < MIN_PREMIUM:
            bits.append(f"the mid is ${premium:.2f}, near the tick")
        reason = ("No edge established, and " + ", and ".join(bits)
                  + f". Expect a fill worse than ${premium:.2f}.")
    else:
        tier = SPECULATIVE
        reason = ("No edge established today — this is a directional guess with "
                  f"defined risk. You lose ${cost:,.2f} per contract if it "
                  "expires worthless, and that is the whole downside.")

    return ContractIdea(
        symbol=chain.symbol, right=right, tier=tier, reason=reason,
        strike=float(contract.strike), expiration=chain.expiration.isoformat(),
        premium=round(premium, 2), cost=cost, max_loss=cost,
        breakeven=round(breakeven, 2),
        delta=round(delta, 3) if delta is not None else None,
        spread_pct=None if math.isnan(sp) else round(sp, 2),
        open_interest=int(contract.open_interest), volume=int(contract.volume),
        tradability=score, underlying=round(chain.underlying_price, 2),
        days_to_expiry=round(chain.days_to_expiry, 2),
    )


def _empty(right: str, reason: str) -> ContractIdea:
    return ContractIdea(symbol="", right=right, tier=NO_CHAIN, reason=reason)


def build_board(flows: dict | None, *, slots: int = DEFAULT_SLOTS,
                qualified_symbols: set[str] | None = None,
                excluded_symbols: set[str] | None = None) -> dict:
    """Exactly `slots` calls and `slots` puts. Always.

    `excluded_symbols` are names the review gate rejected today. They are not
    merely low-conviction — the swarm reached a negative conclusion about them,
    and a CRITICAL objection is the strongest signal it produces. "No edge
    established" and "we examined this and concluded against it" are different
    states, and only the first is honestly expressible as a gamble. Printing a
    rejected name as a contract to buy would let the duty to fill six slots
    override a safety gate, which is the failure this board is most likely to
    cause and the one it must not.

    `flows` is reports["options_flow"].data["flows"] — symbol -> snapshot, where
    the snapshot carries the live Chain under "chain" (flow_agents.py:60). A
    missing, empty or chainless `flows` is not an error here: it produces NO
    CHAIN rows that name what was unavailable, because "we could not look" and
    "we looked and found nothing" are different answers and the reader needs to
    know which one they got.
    """
    qualified_symbols = qualified_symbols or set()
    excluded_symbols = excluded_symbols or set()
    flows = flows or {}

    board: dict[str, list[ContractIdea]] = {"calls": [], "puts": []}
    tried: list[str] = []
    barred: list[str] = []

    for right, key in (("call", "calls"), ("put", "puts")):
        ideas: list[ContractIdea] = []
        for symbol, snap in flows.items():
            if symbol in excluded_symbols:
                if symbol not in barred:
                    barred.append(symbol)
                continue
            chain = (snap or {}).get("chain")
            if chain is None:
                if symbol not in tried:
                    tried.append(symbol)
                continue
            contract = _pick(chain, right)
            if contract is None:
                if symbol not in tried:
                    tried.append(symbol)
                continue
            ideas.append(_idea_from(chain, contract, right,
                                    qualified=symbol in qualified_symbols))

        # Qualified first, then by how tradable it actually is. Belief outranks
        # liquidity; among equals, the one you can get out of wins.
        rank = {QUALIFIED: 0, SPECULATIVE: 1, LOTTERY: 2}
        ideas.sort(key=lambda i: (rank.get(i.tier, 3), -(i.tradability or 0.0)))
        board[key] = ideas[:slots]

        while len(board[key]) < slots:
            board[key].append(_empty(right, _no_chain_reason(flows, tried, barred)))

    return {"calls": [i.to_dict() for i in board["calls"]],
            "puts": [i.to_dict() for i in board["puts"]]}


def _no_chain_reason(flows: dict, tried: list[str], barred: list[str] | None = None) -> str:
    if barred:
        # Named as a count rather than a list: the reader needs to know a name
        # was withheld and why, and the rejections themselves are already
        # itemised in the report's review-gate section.
        n = len(barred)
        return (f"{n} name{'s' if n > 1 else ''} withheld — the review gate "
                "rejected them today, and a rejected name is not a gamble the "
                "swarm will offer. See the review gate section for which.")
    if not flows:
        return ("No option chains reached this run — the options provider "
                "returned nothing for any symbol. This slot is empty because "
                "the data was missing, not because nothing qualified.")
    if tried:
        names = ", ".join(tried[:6])
        return (f"No quotable contract in the chains we could read ({names}). "
                "Every candidate had no bid, no ask, or no priced strike.")
    return ("Fewer symbols had usable chains than there are slots. This slot is "
            "empty for want of data, not for want of conviction.")


def board_is_complete(board: dict, *, slots: int = DEFAULT_SLOTS) -> bool:
    """The invariant, in one callable place so tests and callers agree."""
    return (len(board.get("calls", [])) == slots
            and len(board.get("puts", [])) == slots)


def filled_count(board: dict) -> int:
    """How many slots carry a real contract. Reported, never used to suppress."""
    return sum(1 for side in ("calls", "puts")
               for row in board.get(side, []) if row.get("strike") is not None)
