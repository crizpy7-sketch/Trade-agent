import os
from pathlib import Path

from marketswarm.config import Config


def test_data_dir_override_moves_reports(tmp_path, monkeypatch):
    """Relocating the data dir must take the reports with it — a sandboxed
    service cannot write to the home directory."""
    monkeypatch.setenv("MARKETSWARM_DATA_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("MARKETSWARM_REPORT_DIR", raising=False)
    cfg = Config.load(tmp_path / "no-such-config.yaml")
    assert cfg.data_dir == tmp_path / "state"
    assert cfg.report_dir == tmp_path / "state" / "reports"
    assert cfg.db_path == tmp_path / "state" / "memory.db"
    assert cfg.report_dir.exists() and cfg.cache_dir.exists()


def test_explicit_report_dir_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKETSWARM_DATA_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MARKETSWARM_REPORT_DIR", str(tmp_path / "elsewhere"))
    cfg = Config.load(tmp_path / "no-such-config.yaml")
    assert cfg.report_dir == tmp_path / "elsewhere"


def test_universe_override_and_normalisation(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKETSWARM_DATA_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("MARKETSWARM_UNIVERSE", "spy, qqq ,nvda")
    cfg = Config.load(tmp_path / "none.yaml")
    assert cfg.universe == ["SPY", "QQQ", "NVDA"]


def test_yaml_file_is_read_and_env_overrides_it(tmp_path, monkeypatch):
    cfgfile = tmp_path / "config.yaml"
    cfgfile.write_text(
        "base_risk_pct: 0.25\nrun_time_et: '07:00'\nsignal_correlation: 0.5\n"
        f"data_dir: {tmp_path / 'fromfile'}\n"
    )
    monkeypatch.delenv("MARKETSWARM_DATA_DIR", raising=False)
    monkeypatch.setenv("MARKETSWARM_RUN_TIME", "09:00")
    cfg = Config.load(cfgfile)
    assert cfg.base_risk_pct == 0.25
    assert cfg.signal_correlation == 0.5
    assert cfg.run_time_et == "09:00"          # env beats file
    assert cfg.data_dir == tmp_path / "fromfile"
    assert cfg.report_dir == tmp_path / "fromfile" / "reports"


def test_unknown_config_keys_are_ignored(tmp_path, monkeypatch):
    cfgfile = tmp_path / "config.yaml"
    cfgfile.write_text("nonsense_key: 1\nbase_risk_pct: 0.5\n")
    monkeypatch.setenv("MARKETSWARM_DATA_DIR", str(tmp_path / "s"))
    cfg = Config.load(cfgfile)
    assert cfg.base_risk_pct == 0.5
    assert not hasattr(cfg, "nonsense_key")


def test_secrets_never_serialised_to_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKETSWARM_DATA_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    monkeypatch.setenv("FRED_API_KEY", "fred-secret-value")
    cfg = Config.load(tmp_path / "none.yaml")
    assert cfg.anthropic_api_key == "sk-ant-secret-value"
    dumped = cfg.to_yaml()
    assert "sk-ant-secret-value" not in dumped
    assert "fred-secret-value" not in dumped


def test_secrets_in_yaml_are_ignored(tmp_path, monkeypatch):
    cfgfile = tmp_path / "config.yaml"
    cfgfile.write_text(
        "anthropic_api_key: yaml-secret\n"
        "discord_bot_token: yaml-discord-secret\n"
        "x_bearer_token: yaml-x-secret\n"
    )
    monkeypatch.setenv("MARKETSWARM_DATA_DIR", str(tmp_path / "s"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MARKETSWARM_DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    cfg = Config.load(cfgfile)
    assert cfg.anthropic_api_key is None
    assert cfg.discord_bot_token is None
    assert cfg.x_bearer_token is None


def test_community_env_parses_allowlists_bounds_values_and_keeps_tokens_secret(
        tmp_path, monkeypatch):
    monkeypatch.setenv("MARKETSWARM_DATA_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("MARKETSWARM_SOCIAL_DISCORD_CHANNEL_IDS", "111, 222 111")
    monkeypatch.setenv("MARKETSWARM_X_HANDLES", "@Alice, bob @Alice")
    monkeypatch.setenv("MARKETSWARM_DISCORD_BOT_TOKEN", "discord-secret-token")
    monkeypatch.setenv("X_BEARER_TOKEN", "x-secret-token")
    monkeypatch.setenv("MARKETSWARM_COMMUNITY_LOOKBACK_HOURS", "999")
    monkeypatch.setenv("MARKETSWARM_COMMUNITY_MIN_SOURCES", "1")

    cfg = Config.load(tmp_path / "none.yaml")
    assert cfg.community_discord_channel_ids == ["111", "222"]
    assert cfg.x_handles == ["alice", "bob"]
    assert cfg.community_lookback_hours == 168
    assert cfg.community_min_sources == 2
    dumped = cfg.to_yaml()
    assert "discord-secret-token" not in dumped
    assert "x-secret-token" not in dumped
