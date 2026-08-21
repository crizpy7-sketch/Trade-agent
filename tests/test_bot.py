"""The interactive Discord bot.

Most of what matters here is refusal. A research command spends provider rate
limit and API quota, and the bot sits in a channel other people can be invited
to, so the interesting cases are the ones where it declines: no allowlist, wrong
user, its own messages, and a command that throws.

The Discord REST API is faked. These do not prove the bot talks to Discord
correctly — only that it decides correctly.
"""

from __future__ import annotations

import asyncio

import pytest

from marketswarm.bot import COOLDOWN_SECONDS, DiscordBot
from marketswarm.config import Config


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.data_dir = tmp_path
    c.report_dir = tmp_path / "reports"
    c.ensure_dirs()
    return c


def bot(cfg, **kw):
    kw.setdefault("token", "t")
    kw.setdefault("channel_id", "123")
    kw.setdefault("allowed", {"owner-1"})
    return DiscordBot(cfg, **kw)


def msg(content, user="owner-1", is_bot=False):
    return {"id": "1", "content": content,
            "author": {"id": user, "username": "u", "bot": is_bot}}


def handle(b, m):
    return asyncio.run(b.handle(m))


# ---------------------------------------------------------------- refusal

def test_an_empty_allowlist_is_not_configured(cfg):
    """Fail closed. No allowlist must mean nobody, not everybody."""
    ok, why = bot(cfg, allowed=set()).configured()
    assert ok is False
    assert "refuses to answer anyone" in why


def test_a_missing_token_is_not_configured(cfg):
    ok, why = bot(cfg, token="").configured()
    assert ok is False and "TOKEN" in why


def test_a_stranger_is_refused_and_told_why(cfg):
    reply = handle(bot(cfg), msg("!status", user="someone-else"))
    assert "Not authorised" in reply
    # The refusal names the id so the owner can allowlist themselves without
    # having to go digging for it.
    assert "someone-else" in reply


def test_the_bot_never_answers_itself(cfg):
    """Its own reports arrive in the same channel. Answering them would loop."""
    assert handle(bot(cfg), msg("!status", is_bot=True)) is None


def test_ordinary_chat_is_ignored(cfg):
    for text in ("hello", "what do you think about NVDA", "", "  "):
        assert handle(bot(cfg), msg(text)) is None


def test_a_second_command_inside_the_cooldown_is_dropped(cfg):
    b = bot(cfg)
    assert handle(b, msg("!help")) is not None
    assert handle(b, msg("!help")) is None, "cooldown did not apply"
    assert COOLDOWN_SECONDS > 0


def test_the_cooldown_is_per_user(cfg):
    b = bot(cfg, allowed={"a", "b"})
    assert handle(b, msg("!help", user="a")) is not None
    assert handle(b, msg("!help", user="b")) is not None, \
        "one user's command silenced another's"


def test_an_unknown_command_is_answered_not_ignored(cfg):
    reply = handle(bot(cfg), msg("!frobnicate"))
    assert "Unknown command" in reply and "!help" in reply


def test_a_failing_command_reports_instead_of_killing_the_bot(cfg):
    b = bot(cfg)

    async def boom(args):
        raise RuntimeError("database on fire")

    b.cmd_status = boom
    reply = handle(b, msg("!status"))
    assert "failed" in reply and "database on fire" in reply


# ---------------------------------------------------------------- answers

def test_help_lists_the_commands_and_states_the_limits(cfg):
    reply = handle(bot(cfg), msg("!help"))
    for cmd in ("!status", "!today", "!plays", "!ticker", "!why", "!calibration"):
        assert cmd in reply
    # The bot must not imply it can act.
    assert "cannot run the swarm" in reply


def test_ticker_without_a_symbol_explains_the_usage(cfg):
    assert "Usage" in handle(bot(cfg), msg("!ticker"))


def test_why_without_a_symbol_explains_the_usage(cfg):
    assert "Usage" in handle(bot(cfg), msg("!why"))


def test_today_on_an_empty_database_says_so_rather_than_inventing(cfg):
    reply = handle(bot(cfg), msg("!today 2020-01-02"))
    assert "No run recorded" in reply


def test_calibration_with_no_history_makes_no_claim(cfg):
    reply = handle(bot(cfg), msg("!calibration"))
    assert "nothing to calibrate" in reply.lower() or "No resolved" in reply


def test_plays_always_has_three_calls_and_three_puts_even_without_data(cfg):
    reply = handle(bot(cfg), msg("!plays"))
    assert "__Calls__" in reply and "__Puts__" in reply
    assert reply.count("**DATA UNAVAILABLE**") == 6
    assert "does not mean six trades" in reply


def test_a_ticker_lookup_survives_the_quote_provider_being_down(cfg, monkeypatch):
    """No network in this sandbox, which is the same shape as a dead provider:
    the answer degrades and says so instead of erroring."""
    reply = handle(bot(cfg), msg("!ticker NVDA"))
    assert "NVDA" in reply
    assert "unavailable" in reply or "not examined" in reply
    assert "not financial advice" in reply.lower()


# ---------------------------------------------------------------- plumbing

def test_a_long_reply_is_marked_as_truncated_not_silently_cut(cfg):
    """A clipped answer that looks complete is worse than a short one."""
    b = bot(cfg)
    sent = {}

    class FakeResponse:
        status_code = 200

    class FakeHTTP:
        async def post(self, url, json=None, headers=None):
            sent["content"] = json["content"]
            return FakeResponse()

    b._http = FakeHTTP()
    asyncio.run(b.send("x" * 5000))
    assert sent["content"].endswith("(truncated)")
    assert len(sent["content"]) < 2000, "reply exceeds Discord's limit"


def test_the_allowlist_parses_commas_and_spaces(monkeypatch, cfg):
    from marketswarm.bot import _env_ids
    monkeypatch.setenv("X_IDS", "111, 222   333,444")
    assert _env_ids("X_IDS") == {"111", "222", "333", "444"}
    monkeypatch.setenv("X_IDS", "")
    assert _env_ids("X_IDS") == set()
