"""Names, and being able to ask one analyst what they found.

Two things are being protected here.

The first is that a name is presentation and nothing else. Nothing dispatches
on it and nothing is authorised by it, so a rename can never change behaviour.
If that ever stops being true, the roster becomes a security surface, which is
not what a display name is for.

The second is that the answer has to be real. The bot reads the database and
never recomputes, so "what did Sasha research" is only answerable if the run
actually wrote it down. Before M008 the schema recorded that an agent
succeeded and not one word of what it found, which made the obvious question
unanswerable — and a bot that says "no data" to the obvious question gets
closed and not reopened.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from marketswarm.agents import roster
from marketswarm.bot import DiscordBot
from marketswarm.config import Config


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.data_dir = tmp_path
    c.report_dir = tmp_path / "reports"
    c.ensure_dirs()
    return c


def bot(cfg):
    return DiscordBot(cfg, token="t", channel_id="1", allowed={"owner"})


def say(b, text):
    return asyncio.run(b.handle(
        {"id": "1", "content": text, "author": {"id": "owner", "username": "u", "bot": False}}))


# ------------------------------------------------------------- the roster

def test_every_shipped_agent_has_a_name():
    """An agent with no entry is invisible on the board unless handled."""
    from marketswarm.agents import ALL_AGENTS
    missing = [c.name for c in ALL_AGENTS if c.name not in roster.ROSTER]
    assert not missing, f"agents with no name assigned: {missing}"


def test_names_are_unique():
    """Two Sashas makes `!who sasha` a coin toss."""
    names = [name for _, name, _ in roster.everyone()]
    assert len(names) == len(set(names))


def test_a_person_resolves_to_their_machine_name():
    assert roster.resolve("Sasha") == "options_flow"
    assert roster.resolve("sasha") == "options_flow"
    assert roster.resolve("options_flow") == "options_flow"


def test_an_unambiguous_prefix_resolves_and_an_ambiguous_one_does_not():
    assert roster.resolve("dmi") == "red_team"
    # Several names begin with S; guessing between them would answer for the
    # wrong analyst, which is worse than asking again.
    assert roster.resolve("s") is None
    assert roster.resolve("") is None
    assert roster.resolve("nobody") is None


def test_an_unknown_agent_still_displays_rather_than_vanishing():
    """A working agent missing from a hand-kept list is the list's fault."""
    assert roster.person("brand_new_agent") == "brand_new_agent"
    assert "no beat recorded" in roster.beat("brand_new_agent")


# ------------------------------------------------- the schema must store it

def test_the_schema_records_what_an_agent_found_not_only_that_it_ran(cfg):
    """M008. Without these columns the bot cannot answer the obvious question."""
    from marketswarm.memory.store import MemoryStore
    from marketswarm.memory.migrations import migrate
    store = MemoryStore(cfg.db_path)
    migrate(store.conn)
    cols = {r[1] for r in store.conn.execute("PRAGMA table_info(agent_runs)")}
    assert "headline" in cols and "findings" in cols
    store.conn.close()


def test_a_persisted_trace_round_trips_its_findings(cfg):
    from marketswarm.memory.store import MemoryStore
    from marketswarm.observability import Observatory
    from marketswarm.memory.migrations import migrate
    from marketswarm.agents.base import AgentReport

    store = MemoryStore(cfg.db_path)
    migrate(store.conn)
    obs = Observatory(conn=store.conn, run_id=1)
    rep = AgentReport(agent="options_flow", headline="SPY implied move +/-1.4%")
    rep.add("SPY call walls 590, 595")
    rep.add("NVDA unusual: 9,000 calls at 125")
    obs.record_report(rep)

    row = store.conn.execute(
        "SELECT headline, findings FROM agent_runs WHERE agent='options_flow'"
    ).fetchone()
    assert "implied move" in row[0]
    assert "call walls" in json.loads(row[1])[0]
    store.conn.close()


# ------------------------------------------------------------- the commands

def test_who_without_a_name_lists_the_desk(cfg):
    reply = say(bot(cfg), "!who")
    assert "Usage" in reply
    assert "Sasha" in reply and "Dmitri" in reply


def test_who_names_an_unknown_analyst_rather_than_guessing(cfg):
    reply = say(bot(cfg), "!who Gerald")
    assert "Gerald" in reply
    assert "!agents" in reply


def test_who_states_the_beat_even_before_any_run_has_happened(cfg):
    """An empty database must still answer 'who is Sasha and what do they do'."""
    reply = say(bot(cfg), "!who Sasha")
    assert "Sasha" in reply
    assert "implied moves" in reply
    assert "not reported" in reply


def test_the_desk_is_an_alias_for_agents(cfg):
    assert say(bot(cfg), "!desk") == say(bot(cfg), "!agents")


def test_help_advertises_the_new_commands(cfg):
    reply = say(bot(cfg), "!help")
    assert "!who" in reply and "Sasha" in reply
    # The limits must survive the additions.
    assert "cannot run the swarm" in reply


# --------------------------------------------- the two bugs found by asking

def test_agent_runs_are_written_with_the_run_id_that_makes_them_findable(cfg):
    """`!agents` answered "no runs recorded" on databases that had them.

    Pipeline2 built its Observatory with no run_id, so every agent_runs row was
    written with run_id NULL, and agent_status() keys off MAX(run_id) — which
    is NULL when every row is NULL. The rows existed and were unreachable, so
    the command had never worked on any machine.
    """
    import asyncio
    import marketswarm.orchestrator as orch
    from marketswarm.memory.store import MemoryStore
    from tests.fakes import (FakeEarnings, FakeEcon, FakeEdgar, FakeMarket,
                             FakeNews, FakeOptions)

    universe = ["SPY", "QQQ", "NVDA"]
    market = FakeMarket(universe)
    orig = (orch.MarketData, orch.OptionsData, orch.NewsData,
            orch.EconData, orch.EdgarData, orch.EarningsData)
    orch.MarketData = lambda c: market
    orch.OptionsData = lambda c: FakeOptions(market)
    orch.NewsData = lambda c, universe=None: FakeNews()
    orch.EconData = lambda c, k: FakeEcon()
    orch.EdgarData = lambda c, ua: FakeEdgar()
    orch.EarningsData = lambda c: FakeEarnings()
    try:
        cfg.universe = list(universe)
        cfg.index_symbols = ["SPY", "QQQ"]
        asyncio.run(orch.Swarm(cfg).run())
    finally:
        (orch.MarketData, orch.OptionsData, orch.NewsData,
         orch.EconData, orch.EdgarData, orch.EarningsData) = orig

    store = MemoryStore(cfg.db_path)
    nulls = store.conn.execute(
        "SELECT COUNT(*) FROM agent_runs WHERE run_id IS NULL").fetchone()[0]
    total = store.conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0]
    store.conn.close()

    assert total > 0, "the run recorded no agents at all"
    assert nulls == 0, f"{nulls} of {total} agent rows are unreachable by run_id"

    # And the whole point: the desk can now be read back by name.
    reply = say(bot(cfg), "!agents")
    assert "no agent runs" not in reply.lower()
    assert "Sasha" in reply, "the desk did not come back with names"
    assert "reporting" in reply


def test_a_completed_agent_is_not_reported_as_down(cfg):
    """agent_runs stores AgentStatus values, not AgentReport strings.

    The status column holds "COMPLETE"; the old check compared against "ok",
    so every healthy agent was rendered as degraded.
    """
    from marketswarm.memory.store import MemoryStore
    from marketswarm.memory.migrations import migrate
    from marketswarm.observability import Observatory
    from marketswarm.agents.base import AgentReport

    store = MemoryStore(cfg.db_path)
    migrate(store.conn)
    obs = Observatory(conn=store.conn, run_id=7)
    rep = AgentReport(agent="risk", headline="per-idea risk budget 0.38%")
    rep.add("1 high-severity risk")
    obs.record_report(rep)
    store.conn.close()

    reply = say(bot(cfg), "!agents")
    assert "1 of 1 reporting" in reply
    assert "DOWN" not in reply
    assert "Rosa" in reply
