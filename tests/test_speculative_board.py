"""The daily contract board must never be empty, and never oversell.

Two failure modes, pulling in opposite directions, and the tests exist to hold
both at once:

  going silent     The rest of the pipeline has eleven ways to reduce a day to
                   nothing, and on a quiet session several fire together. The
                   board is the one thing that must still produce six rows.

  overselling      A filled slot must not read as endorsement. A contract with
                   no edge behind it has to say so in its own row, or the
                   never-silent guarantee becomes a machine for manufacturing
                   false confidence — which is worse than silence.

So every test here checks one of: six rows came back, or the row is honest
about how little is behind it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math

import pytest

from marketswarm.providers.options import Chain, Contract
from marketswarm.recommend.speculative import (
    LOTTERY, NO_CHAIN, QUALIFIED, SPECULATIVE,
    board_is_complete, build_board, filled_count, tradability,
)
from tests.fakes import synthetic_chain


def flows(*chains) -> dict:
    return {c.symbol: {"symbol": c.symbol, "chain": c} for c in chains}


def rows(board) -> list[dict]:
    return board["calls"] + board["puts"]


# ------------------------------------------------------- never silent

def test_a_normal_day_fills_every_slot():
    board = build_board(flows(synthetic_chain("SPY", 500.0),
                              synthetic_chain("NVDA", 120.0),
                              synthetic_chain("AAPL", 220.0)))
    assert board_is_complete(board)
    assert filled_count(board) == 6


def test_no_option_data_at_all_still_returns_six_rows():
    """The worst case: the provider gave us nothing. Six rows, all explained."""
    board = build_board({})

    assert board_is_complete(board)
    assert all(r["tier"] == NO_CHAIN for r in rows(board))
    # And it must say the slot is empty for want of data, not want of conviction.
    for r in rows(board):
        assert "data" in r["reason"] or "returned nothing" in r["reason"]


def test_none_flows_is_treated_as_no_data_not_an_error():
    assert board_is_complete(build_board(None))


def test_one_symbol_still_fills_all_six_slots():
    """Fewer chains than slots is the common case, not an edge case.

    Only ~9 symbols get chains at all (flow_agents.py:17), and on a bad morning
    that can collapse to one.
    """
    board = build_board(flows(synthetic_chain("SPY", 500.0)))

    assert board_is_complete(board)
    assert board["calls"][0]["strike"] is not None
    assert board["calls"][1]["tier"] == NO_CHAIN
    assert "for want of data" in board["calls"][1]["reason"]


def test_a_chainless_snapshot_names_the_symbol_it_could_not_read():
    """'We could not look' and 'we looked and found nothing' are different."""
    board = build_board({"SPY": {"symbol": "SPY", "chain": None}})

    assert board_is_complete(board)
    assert "SPY" in board["calls"][0]["reason"]


def test_an_unquotable_chain_does_not_produce_a_phantom_contract():
    """Every contract bid and ask at zero. Better an empty slot than a fake fill."""
    dead = Chain("DEAD", 100.0, dt.date(2026, 9, 18), 5.0,
                 [Contract(100.0, 0.0, 0.0, 0.0, 0, 0, 0.0, False, "call")],
                 [Contract(100.0, 0.0, 0.0, 0.0, 0, 0, 0.0, False, "put")])
    board = build_board(flows(dead))

    assert board_is_complete(board)
    assert all(r["tier"] == NO_CHAIN for r in rows(board))
    assert all(r["strike"] is None for r in rows(board))


# ------------------------------------------------------- never oversell

def test_an_ordinary_pick_is_labelled_speculative_not_recommended():
    board = build_board(flows(synthetic_chain("SPY", 500.0)))
    top = board["calls"][0]

    assert top["tier"] == SPECULATIVE
    assert "No edge established" in top["reason"]


def test_the_speculative_reason_states_the_dollar_downside():
    """A gamble is only honest if the loss is on the same line as the idea."""
    board = build_board(flows(synthetic_chain("SPY", 500.0)))
    top = board["calls"][0]

    assert f"${top['cost']:,.2f}" in top["reason"]
    assert top["max_loss"] == top["cost"]


def test_only_the_swarms_own_gates_can_promote_a_row_to_qualified():
    ok = build_board(flows(synthetic_chain("SPY", 500.0)),
                     qualified_symbols={"SPY"})
    assert ok["calls"][0]["tier"] == QUALIFIED

    # Absent that, nothing self-promotes.
    plain = build_board(flows(synthetic_chain("SPY", 500.0)))
    assert plain["calls"][0]["tier"] != QUALIFIED


def test_a_qualified_row_sorts_above_a_more_liquid_speculative_one():
    """Belief outranks liquidity; among equals, tradability decides."""
    board = build_board(flows(synthetic_chain("SPY", 500.0),
                              synthetic_chain("NVDA", 120.0)),
                        qualified_symbols={"NVDA"})
    assert board["calls"][0]["symbol"] == "NVDA"


def test_a_thin_contract_is_demoted_to_lottery_and_says_why():
    thin = Chain("THIN", 100.0, dt.date(2026, 9, 18), 5.0,
                 [Contract(100.0, 1.00, 0.97, 1.03, 3, 11, 0.30, False, "call")],
                 [Contract(100.0, 1.00, 0.97, 1.03, 2, 9, 0.30, False, "put")])
    board = build_board(flows(thin))
    top = board["calls"][0]

    assert top["tier"] == LOTTERY
    assert "OI" in top["reason"]
    assert top["strike"] == 100.0, "a lottery row is still a real contract"


def test_a_wide_spread_is_demoted_and_the_fill_warning_is_explicit():
    wide = Chain("WIDE", 100.0, dt.date(2026, 9, 18), 5.0,
                 [Contract(100.0, 2.00, 1.40, 2.60, 900, 4000, 0.30, False, "call")],
                 [Contract(100.0, 2.00, 1.40, 2.60, 900, 4000, 0.30, False, "put")])
    board = build_board(flows(wide))
    top = board["calls"][0]

    assert top["tier"] == LOTTERY
    assert "spread" in top["reason"]
    assert "worse than" in top["reason"]


# ------------------------------------------------------- the arithmetic

def test_breakeven_moves_the_right_way_for_each_side():
    board = build_board(flows(synthetic_chain("SPY", 500.0)))
    call, put = board["calls"][0], board["puts"][0]

    assert call["breakeven"] == pytest.approx(call["strike"] + call["premium"], abs=0.01)
    assert put["breakeven"] == pytest.approx(put["strike"] - put["premium"], abs=0.01)


def test_cost_is_one_contract_of_a_hundred_shares():
    board = build_board(flows(synthetic_chain("SPY", 500.0)))
    row = board["calls"][0]
    assert row["cost"] == pytest.approx(row["premium"] * 100, abs=0.01)


def test_max_loss_on_a_long_option_is_the_premium_and_nothing_more():
    for row in rows(build_board(flows(synthetic_chain("SPY", 500.0)))):
        if row["strike"] is not None:
            assert row["max_loss"] == row["cost"]


# ------------------------------------------------------- the liquidity bug

def test_a_crossed_quote_scores_zero_not_perfect():
    """Chain.liquidity_score treats a NaN spread as 1.0 (options.py:216).

    NaN means ask <= bid — the least tradable state there is. Scored as perfect,
    the worst contracts sort first, so the board would preferentially recommend
    exactly the contracts nobody can get out of.
    """
    crossed = Contract(100.0, 1.0, 1.20, 1.00, 500, 5000, 0.3, False, "call")
    assert math.isnan(crossed.spread_pct)
    assert tradability(crossed) == 0.0

    healthy = Contract(100.0, 1.0, 0.98, 1.02, 500, 5000, 0.3, False, "call")
    assert tradability(healthy) > tradability(crossed)


def test_the_chains_own_liquidity_score_no_longer_rates_a_dead_chain_perfect():
    """The same bug, fixed at source in options.py rather than worked around.

    liquidity_score feeds engine.choose_type (engine.py:261, AVOID below 0.25)
    and edge.rank_opportunities, so scoring a crossed chain 1.0 did not only
    mislead this board — it let an untradeable chain pass the AVOID gate in the
    recommendations you actually act on.
    """
    crossed = Chain("CROSSED", 100.0, dt.date(2026, 9, 18), 5.0,
                    [Contract(100.0, 1.0, 1.30, 1.00, 500, 5000, 0.3, False, "call")],
                    [Contract(100.0, 1.0, 1.30, 1.00, 500, 5000, 0.3, False, "put")])
    healthy = synthetic_chain("HEALTHY", 100.0)

    assert crossed.liquidity_score() < 0.25, "a crossed chain must not clear the AVOID gate"
    assert healthy.liquidity_score() > crossed.liquidity_score()


def test_a_tradable_contract_outranks_a_crossed_one_in_the_same_tier():
    good = synthetic_chain("GOOD", 100.0)
    bad = Chain("BAD", 100.0, dt.date(2026, 9, 18), 5.0,
                [Contract(100.0, 1.0, 1.30, 1.00, 900, 9000, 0.3, False, "call")],
                [Contract(100.0, 1.0, 1.30, 1.00, 900, 9000, 0.3, False, "put")])
    board = build_board(flows(bad, good))

    assert board["calls"][0]["symbol"] == "GOOD"


# ------------------------------------------------------- shape

def test_every_row_carries_a_tier_and_a_reason_even_when_empty():
    """A row with no explanation is the thing this whole module exists to avoid."""
    for board in (build_board({}), build_board(flows(synthetic_chain("SPY", 500.0)))):
        for row in rows(board):
            assert row["tier"]
            assert row["reason"].strip()


def test_the_board_is_plain_data_so_it_can_be_persisted_and_rendered():
    board = build_board(flows(synthetic_chain("SPY", 500.0)))
    import json
    json.dumps(board)  # raises if anything non-serialisable leaked in


# ============================================================ GATE 1
# Arithmetic checked against values computed by hand, not against a golden file
# this module produced. A golden file only proves the code still does what it
# did; it cannot tell you the first answer was right. These five were worked out
# on paper first, and the code is asserted against them.

@pytest.mark.parametrize(
    "right, strike, bid, ask, want_premium, want_cost, want_breakeven",
    [
        # ITM call: mid of 1.00/1.10 is 1.05. 500 + 1.05 = 501.05.
        ("call", 500.0, 1.00, 1.10, 1.05, 105.00, 501.05),
        # OTM call, cheap: mid 0.20/0.30 = 0.25. 510 + 0.25 = 510.25.
        ("call", 510.0, 0.20, 0.30, 0.25, 25.00, 510.25),
        # ITM put: mid 2.00/2.20 = 2.10. Puts subtract: 500 - 2.10 = 497.90.
        ("put", 500.0, 2.00, 2.20, 2.10, 210.00, 497.90),
        # OTM put: mid 0.40/0.60 = 0.50. 490 - 0.50 = 489.50.
        ("put", 490.0, 0.40, 0.60, 0.50, 50.00, 489.50),
        # Pathological spread: mid 0.50/4.50 = 2.50. The arithmetic is unchanged
        # by the spread being absurd — the spread governs the tier, not the maths.
        ("call", 505.0, 0.50, 4.50, 2.50, 250.00, 507.50),
    ],
)
def test_contract_economics_match_values_computed_by_hand(
        right, strike, bid, ask, want_premium, want_cost, want_breakeven):
    c = Contract(strike=strike, last=(bid + ask) / 2, bid=bid, ask=ask,
                 volume=900, open_interest=9000, implied_volatility=0.25,
                 in_the_money=False, kind=right)
    chain = Chain("HAND", 500.0, dt.date(2026, 9, 18), 5.0,
                  [c] if right == "call" else [],
                  [c] if right == "put" else [])

    row = build_board(flows(chain))["calls" if right == "call" else "puts"][0]

    assert row["premium"] == pytest.approx(want_premium, abs=0.005)
    assert row["cost"] == pytest.approx(want_cost, abs=0.01)
    assert row["breakeven"] == pytest.approx(want_breakeven, abs=0.01)
    # A long single-leg option cannot lose more than it cost. If these two
    # numbers ever diverge, the build is wrong.
    assert row["max_loss"] == row["cost"]


# ============================================================ THE 60-DAY PROOF
# A single passing day proves nothing about a system whose failure mode is
# intermittent. This runs sixty consecutive sessions with randomised provider
# failures and asserts the board never once came up short.

def test_sixty_consecutive_sessions_never_produce_a_short_board():
    import random
    rng = random.Random(20260824)

    universe = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "TSLA"]
    boards, total_rows, blank_days = 0, 0, 0

    for day in range(60):
        day_flows = {}
        for sym in universe:
            roll = rng.random()
            if roll < 0.25:
                continue                                  # provider skipped it
            if roll < 0.35:
                day_flows[sym] = {"symbol": sym, "chain": None}   # chain failed
                continue
            if roll < 0.42:                                        # dead quotes
                day_flows[sym] = {"symbol": sym, "chain": Chain(
                    sym, 100.0, dt.date(2026, 9, 18), 3.0,
                    [Contract(100.0, 0.0, 0.0, 0.0, 0, 0, 0.0, False, "call")],
                    [Contract(100.0, 0.0, 0.0, 0.0, 0, 0, 0.0, False, "put")])}
                continue
            day_flows[sym] = {"symbol": sym,
                              "chain": synthetic_chain(sym, 100.0 + rng.random() * 400)}

        if rng.random() < 0.08:      # total outage: the provider returned nothing
            day_flows = {}
            blank_days += 1

        board = build_board(day_flows)
        assert board_is_complete(board), f"short board on simulated day {day}"
        boards += 1
        total_rows += len(board["calls"]) + len(board["puts"])

    assert boards == 60
    assert total_rows == 360, "sixty sessions must yield exactly 360 rows"
    assert blank_days > 0, "the simulation never exercised a total outage"


# ============================================================ NO-LEAK
# The emission floor must never become a confidence floor. A row that exists
# only because a slot had to be filled must not be able to enter the track
# record, where it would later read as a prediction the swarm made and got right.

def test_a_speculative_row_is_not_shaped_like_a_scoreable_prediction():
    """The resolver needs entry/target/stop (learning.py:66-68).

    A board row deliberately carries none of them, so it cannot be fed to
    resolve_prediction even by accident — the separation is structural, not a
    convention someone has to remember.
    """
    for row in rows(build_board(flows(synthetic_chain("SPY", 500.0)))):
        for scoreable in ("entry", "target", "stop", "confidence",
                          "probability", "expected_r"):
            assert scoreable not in row, (
                f"board row carries {scoreable!r} — it could be scored as a "
                "prediction, which would turn a forced pick into a track record")


def test_no_row_claims_a_probability_or_an_edge():
    """Nothing on this board may imply a measured likelihood of being right."""
    for row in rows(build_board(flows(synthetic_chain("SPY", 500.0)))):
        text = row["reason"].lower()
        for forbidden in ("likely", "probability", "we expect", "should reach",
                          "high conviction", "strong signal"):
            assert forbidden not in text, f"{forbidden!r} in a no-edge row"


# ============================================ rejected names are not gambles

def test_a_name_the_review_gate_rejected_is_never_offered():
    """"No edge established" and "we concluded against this" are different.

    A CRITICAL objection is the strongest negative signal the swarm produces.
    Offering that exact name as a contract — even labelled SPECULATIVE — lets
    the duty to fill six slots override a safety gate, which is the failure this
    board is most likely to cause. The board reads chains directly and so does
    not otherwise know a name was rejected; it has to be told.
    """
    board = build_board(flows(synthetic_chain("SPY", 500.0),
                              synthetic_chain("MSFT", 415.0)),
                        excluded_symbols={"MSFT"})

    assert board_is_complete(board)
    assert all(r["symbol"] != "MSFT" for r in rows(board))
    assert any(r["symbol"] == "SPY" for r in rows(board)), \
        "the exclusion took the whole board down with it"


def test_excluding_every_name_leaves_a_full_board_that_says_why():
    board = build_board(flows(synthetic_chain("SPY", 500.0)),
                        excluded_symbols={"SPY"})

    assert board_is_complete(board)
    assert all(r["tier"] == NO_CHAIN for r in rows(board))
    reason = board["calls"][0]["reason"]
    assert "withheld" in reason and "review gate" in reason
    # The count is stated; the names are not re-listed here, because the
    # rejections are itemised in the report's review-gate section and repeating
    # them beside a contract offer is what we are trying to avoid.
    assert "SPY" not in reason


def test_an_exclusion_does_not_reduce_the_slot_count():
    """The floor holds even when the exclusion empties the candidate pool."""
    board = build_board(flows(synthetic_chain("SPY", 500.0),
                              synthetic_chain("NVDA", 120.0)),
                        excluded_symbols={"SPY", "NVDA"})
    assert len(board["calls"]) == 3 and len(board["puts"]) == 3
