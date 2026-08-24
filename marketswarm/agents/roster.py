"""Human names for the agents, and the beat each one covers.

The agents are addressed by machine names — `options_flow`, `cross_verify` —
which are precise and unmemorable. Asking "what did options_flow find" is a
different act from asking "what did Sasha find", and only the second one gets
asked casually at 8am on a phone.

The name is presentation. Nothing dispatches on it, nothing is authorised by
it, and the machine name stays the identifier everywhere it matters — the
database, the logs, the routing. A rename here can never change behaviour,
which is the whole reason it is a lookup table and not an attribute on the
agent classes.

`beat` is what that agent goes and looks at, in the words you would use to a
person. It is deliberately not the class docstring: the docstring says what the
code does, the beat says what the human would say they cover.
"""

from __future__ import annotations

#: machine name -> (person, beat)
ROSTER: dict[str, tuple[str, str]] = {
    "overnight_scan": ("Nadia", "who moved while you were asleep, and on what volume"),
    "global_markets": ("Kenji", "how Asia and Europe closed, and what they handed us"),
    "futures": ("Fiona", "index futures, rates, the dollar and commodities before the bell"),
    "volatility_regime": ("Vera", "the VIX complex and whether the regime is calm, choppy or breaking"),
    "technicals": ("Theo", "trend, levels and volatility bands on the index and the leaders"),
    "breaking_news": ("Priya", "overnight headlines, deduplicated and weighted by corroboration"),
    "econ_calendar": ("Elena", "scheduled macro releases and how close they land to the open"),
    "earnings": ("Quinn", "who is trading on results this morning and who reports tonight"),
    "sec_filings": ("Ruth", "overnight EDGAR filings — the primary-source catalysts"),
    "options_flow": ("Sasha", "implied moves, skew, open-interest walls and unusual activity"),
    "institutional": ("Ivan", "insider Form 4 activity and material ownership filings"),
    "sentiment": ("Mira", "measured positioning, plus permissioned community reads at low weight"),
    "cross_verify": ("Clara", "where the swarm agrees, where it contradicts itself, and what rests on one source"),
    "risk": ("Rosa", "what would make today's read wrong, and how much to put behind it"),
    "playbook": ("Paulo", "turning the read into bracketed ideas with Monte-Carlo probabilities"),
    "red_team": ("Dmitri", "arguing against everything the others concluded"),
}


def person(agent: str) -> str:
    """The human name, falling back to the machine name for an unknown agent.

    An agent added without a roster entry must still be displayable. Silently
    dropping it from the board would hide a working agent from its owner.
    """
    entry = ROSTER.get(agent)
    return entry[0] if entry else agent


def beat(agent: str) -> str:
    entry = ROSTER.get(agent)
    return entry[1] if entry else "no beat recorded for this agent"


def resolve(query: str) -> str | None:
    """Machine name for a person, a machine name, or a prefix of either.

    Case-insensitive, because nobody types `Sasha` with a capital on a phone.
    Returns None when the query is ambiguous rather than guessing — answering
    for the wrong agent is worse than asking again.
    """
    q = (query or "").strip().lower()
    if not q:
        return None
    if q in ROSTER:
        return q
    for agent, (name, _) in ROSTER.items():
        if name.lower() == q:
            return agent
    hits = [a for a, (name, _) in ROSTER.items()
            if a.startswith(q) or name.lower().startswith(q)]
    return hits[0] if len(hits) == 1 else None


def everyone() -> list[tuple[str, str, str]]:
    """(machine name, person, beat) in roster order."""
    return [(agent, name, what) for agent, (name, what) in ROSTER.items()]
