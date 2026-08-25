"""The on-demand board poster.

What is testable here is everything except the fetch: this sandbox has no route
to finance.yahoo.com, so the network half is exercised on the host that has one.
What is checked is the part that decides *whether* to post and *what* the
message says — which is where a wrong answer reaches a channel and is believed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from marketswarm.recommend.speculative import build_board
from tests.fakes import synthetic_chain

SPEC = importlib.util.spec_from_file_location(
    "post_board", Path(__file__).resolve().parents[1] / "deploy" / "post-board.py")
post_board = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(post_board)


def flows(*cs):
    return {c.symbol: {"symbol": c.symbol, "chain": c} for c in cs}


def test_the_message_carries_the_numbers_needed_to_look_the_contract_up():
    """A board post that omits the expiry is not checkable against a broker."""
    board = build_board(flows(synthetic_chain("SPY", 588.0),
                              synthetic_chain("NVDA", 121.5)))
    text = post_board.format_for_discord(board, [])

    row = board["calls"][0]
    assert row["symbol"] in text
    assert f"{row['strike']:g}" in text
    assert row["expiration"] in text, "no expiry — the contract cannot be found"
    assert "max loss" in text
    assert "breakeven" in text


def test_the_message_never_calls_itself_a_recommendation():
    board = build_board(flows(synthetic_chain("SPY", 588.0)))
    text = post_board.format_for_discord(board, [])

    assert "Not recommendations" in text
    assert "not financial advice" in text.lower()
    assert "SPECULATIVE" in text or "LOTTERY" in text, "no tier reached the reader"


def test_symbols_with_no_chain_are_named_rather_than_quietly_dropped():
    """Silence about a missing name reads as 'nothing interesting there'."""
    text = post_board.format_for_discord(
        build_board(flows(synthetic_chain("SPY", 588.0))), ["TSLA", "META"])
    assert "TSLA" in text and "META" in text
    assert "No chain" in text


def test_the_message_fits_discords_limit():
    """Eight symbols is the default; the message must survive a full board."""
    chains = [synthetic_chain(s, 100.0 + i * 40)
              for i, s in enumerate(["SPY", "QQQ", "NVDA", "AAPL", "MSFT",
                                     "AMD", "TSLA", "META"])]
    text = post_board.format_for_discord(build_board(flows(*chains)), [])

    assert len(text) <= post_board.MAX_DISCORD
    assert not text.endswith("…"), "truncated mid-board rather than fitting"


def test_a_long_message_is_marked_truncated_rather_than_silently_cut():
    board = build_board(flows(synthetic_chain("SPY", 588.0)))
    text = post_board.format_for_discord(board, ["X" * 4000])

    assert len(text) <= post_board.MAX_DISCORD
    assert text.endswith("(truncated)")


def test_nothing_is_posted_when_no_chain_was_reachable(monkeypatch, capsys):
    """An all-empty board in the channel reads as the swarm having an opinion
    about a quiet market. The truth — nothing was reachable — is a different
    message, and posting the first one in place of the second is the failure
    this whole session has been about."""
    def _no_chains(coro):
        coro.close()          # the fetch is never awaited; do not leak it
        return {}, ["SPY"]
    monkeypatch.setattr(post_board.asyncio, "run", _no_chains)
    monkeypatch.setattr(post_board.Config, "load",
                        classmethod(lambda cls: type("C", (), {"webhook_url": "https://x"})()))
    posted = []
    monkeypatch.setattr(post_board, "send_webhook",
                        lambda url, text: posted.append(text) or True)
    monkeypatch.setattr(post_board.sys, "argv", ["post-board.py"])

    assert post_board.main() == 1
    assert not posted, "an empty board was posted to the channel"
    assert "check-options.py" in capsys.readouterr().err, "no next step offered"


def test_a_missing_webhook_refuses_rather_than_failing_obscurely(monkeypatch, capsys):
    monkeypatch.setattr(post_board.Config, "load",
                        classmethod(lambda cls: type("C", (), {"webhook_url": None})()))
    monkeypatch.setattr(post_board.sys, "argv", ["post-board.py"])

    assert post_board.main() == 1
    assert "--dry-run" in capsys.readouterr().err
