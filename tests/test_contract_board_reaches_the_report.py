"""The board has to survive the days the rest of the pipeline gives up.

A generator that emits six rows into a variable nobody renders is not a daily
board. render_markdown has three exits, and the board must appear on all of
them — most importantly the two that are taken precisely when the swarm has
decided it has nothing to say, because those are the days the owner asked
about.
"""

from __future__ import annotations

import datetime as dt

import marketswarm.orchestrator as orch
from marketswarm.publication import PublicationSet
from marketswarm.report import render_markdown

RUN_DATE = dt.date(2026, 8, 24)

HEADING = "Daily Contract Board"


def test_the_board_renders_when_the_whole_publication_is_suppressed():
    """The commonest silent day: review incomplete, everything withheld."""
    pub = PublicationSet.suppressed_set(
        "adversarial review did not complete — an unreviewed recommendation "
        "is not publishable"
    )
    result = orch.SwarmResult(run_date=RUN_DATE, market_open=True, publication=pub)

    md = render_markdown(result)

    assert HEADING in md, "the board vanished on a suppressed day"
    assert "NO CHAIN" in md, "the empty slots did not explain themselves"


def test_the_board_renders_when_the_playbook_could_not_be_built():
    """The second exit — no playbook and nothing active.

    This path returned before the supporting sections, so a failed playbook
    agent took the board down with it. That is the exact failure the board
    exists to survive: on that day the swarm has nothing, and the owner still
    asked for six rows.
    """
    result = orch.SwarmResult(run_date=RUN_DATE, market_open=True,
                              publication=PublicationSet())

    md = render_markdown(result)

    assert "playbook could not be constructed" in md, "wrong branch under test"
    assert HEADING in md, "the board vanished when the playbook failed"


def test_the_board_never_calls_itself_a_recommendation():
    """Filling a slot must not read as endorsement, on any path."""
    pub = PublicationSet.suppressed_set("review incomplete")
    md = render_markdown(orch.SwarmResult(run_date=RUN_DATE, market_open=True,
                                          publication=pub))

    start = md.index(HEADING)
    section = md[start:start + 1200]
    assert "Not recommendations" in section
    assert "not scored" in section or "track record" in section
