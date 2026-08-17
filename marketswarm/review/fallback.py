"""A deterministic reviewer for when the red-team agent does not complete.

The `RedTeamAgent` already has a mechanical layer that needs no model and no
network. When the agent itself fails — a provider blew up inside it, it timed
out, it was never selected — that mechanical reasoning is still available, and
running it is strictly better than publishing an unreviewed idea or suppressing
a session that could have been reviewed perfectly well.

So this module re-implements the *structural* checks directly against the agent
reports, with no dependency on the red-team agent having run. It is
deliberately a subset: it catches the failure modes that are visible in the
numbers, and it does not pretend to replace an adversary that can read a
headline and notice the thesis is nonsense.

Because it is a subset, a review completed this way is marked
`COMPLETED_BY_FALLBACK` rather than `COMPLETED`, and that distinction is
carried all the way into the run trace and the report. Degraded review is not
the same as review, and the reader is told which one they got.
"""

from __future__ import annotations

import logging

from .gate import Finding, Severity

log = logging.getLogger("marketswarm.review.fallback")


def deterministic_review(reports: dict, symbol: str | None = None) -> list[Finding]:
    """Structural objections derived from the numbers alone.

    Mirrors the mechanical checks in `agents/redteam.py`. Kept here rather than
    imported from the agent so that a failure *inside* the agent cannot take
    the fallback down with it.
    """
    findings: list[Finding] = []

    def data(agent, key, default=None):
        rep = reports.get(agent)
        if not rep or not getattr(rep, "usable", False):
            return default
        return rep.data.get(key, default)

    # --- 1. thin evidence dressed as conviction ---
    eff_n = data("cross_verify", "effective_n", 0) or 0
    raw_n = data("cross_verify", "raw_signal_count", 0) or 0
    if raw_n and eff_n < 2.5:
        findings.append(Finding(
            objection=(f"{raw_n} signals collapse to {eff_n:.1f} independent ones — "
                       f"the apparent agreement is mostly one piece of information "
                       f"counted repeatedly."),
            severity=Severity.HIGH,
            test="Name the single input that, if wrong, would invalidate the whole read.",
            category="independence",
            targets=[symbol] if symbol else [],
        ))

    # --- 2. one-sided book ---
    stocks = data("playbook", "stocks", []) or []
    if len(stocks) >= 2:
        longs = sum(1 for s in stocks if s.get("direction") == "long")
        if longs in (0, len(stocks)):
            side = "long" if longs else "short"
            findings.append(Finding(
                objection=(f"Every stock setup is {side} — one bet on market direction "
                           f"wearing {len(stocks)} costumes."),
                severity=Severity.HIGH,
                test="Size the whole book as a single position, not as several.",
                category="concentration",
            ))

    # --- 3. scheduled event risk the playbook ignored ---
    very_high = data("econ_calendar", "very_high_impact", []) or []
    if very_high and stocks:
        findings.append(Finding(
            objection=(f"{', '.join(very_high)} prints today, yet directional ideas "
                       f"are proposed anyway."),
            severity=Severity.HIGH,
            test="Pre-release ranges compress and the first post-release move often reverses.",
            category="event_risk",
        ))

    # --- 4. degraded inputs ---
    failed = [name for name, rep in reports.items()
              if not getattr(rep, "usable", False)]
    if failed:
        findings.append(Finding(
            objection=(f"{len(failed)} agent(s) produced nothing usable "
                       f"({', '.join(sorted(failed)[:4])}) — the evidence base is "
                       f"narrower than it appears."),
            severity=Severity.MEDIUM,
            test="Check whether the missing inputs are the ones that would argue the "
                 "other side.",
            category="data_quality",
        ))

    # --- 5. the review itself is degraded, and the reader must be told ---
    findings.append(Finding(
        objection=("Adversarial review ran in deterministic fallback mode: the "
                   "red-team agent did not complete, so only structural checks "
                   "were applied and no argument about the substance was made."),
        severity=Severity.MEDIUM,
        test="Treat conviction as provisional until a full review is available.",
        category="review_degraded",
    ))

    log.info("deterministic fallback review produced %d finding(s)", len(findings))
    return findings
