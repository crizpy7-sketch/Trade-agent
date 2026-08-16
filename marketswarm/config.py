"""Configuration: YAML file + environment overrides.

Secrets come from the environment only and are never written to the config
file, the report, or the database.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("marketswarm.config")

DEFAULT_CONFIG_PATH = Path("~/.marketswarm/config.yaml").expanduser()

# Liquid, optionable, tight-spread names. The playbook only ever trades from
# this list, because idea quality is worthless if the fill is not.
DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    "NFLX", "AVGO", "JPM", "XLE", "XLF", "SMH",
]

DEFAULT_INDEX_SYMBOLS = ["SPY", "QQQ", "IWM"]


@dataclass
class Config:
    # --- what to watch ---
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    index_symbols: list[str] = field(default_factory=lambda: list(DEFAULT_INDEX_SYMBOLS))

    # --- statistics ---
    signal_correlation: float = 0.35   # assumed pairwise correlation in fusion
    base_risk_pct: float = 0.75        # per-idea risk budget before conditions
    friction_r: float = 0.06           # round-trip cost as a fraction of R
    max_concurrent_ideas: int = 3
    min_expected_r: float = 0.0        # ideas below this are dropped

    # --- runtime ---
    run_time_et: str = "08:15"         # daily pre-market run
    score_time_et: str = "16:45"       # post-close learning pass
    timeout_seconds: float = 45.0
    cache_ttl_seconds: int = 300
    max_retries: int = 3
    rate_per_second: float = 5.0

    # --- paths ---
    data_dir: Path = field(default_factory=lambda: Path("~/.marketswarm").expanduser())
    report_dir: Path = field(default_factory=lambda: Path("~/.marketswarm/reports").expanduser())

    # --- optional integrations ---
    fred_api_key: str | None = None
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-4-5"
    llm_enabled: bool = True
    contact_email: str = "set MARKETSWARM_CONTACT"
    webhook_url: str | None = None
    notify_on: str = "always"          # always | high_confidence | never

    # --- injected at runtime, not persisted ---
    calibrator: Any = None
    agent_weights: dict[str, float] = field(default_factory=dict)
    lessons: list[dict] = field(default_factory=list)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "memory.db"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def user_agent(self) -> str:
        return f"marketswarm/1.0 (research agent; {self.contact_email})"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.report_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---------- loading ----------

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        cfg = cls()
        default_reports = cfg.report_dir

        p = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
        if p.exists():
            cfg._apply_file(p)
        cfg._apply_env()

        # Relocating the data directory must take the reports with it. Without
        # this, setting only MARKETSWARM_DATA_DIR silently leaves reports in
        # the home directory — where a sandboxed service cannot write them.
        if cfg.report_dir == default_reports and cfg.data_dir != Config().data_dir:
            cfg.report_dir = cfg.data_dir / "reports"

        cfg.ensure_dirs()
        return cfg

    def _apply_file(self, path: Path) -> None:
        try:
            import yaml
            raw = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read config %s: %s — using defaults", path, exc)
            return
        for key, val in raw.items():
            if not hasattr(self, key):
                log.warning("unknown config key %r ignored", key)
                continue
            if key in ("data_dir", "report_dir"):
                val = Path(str(val)).expanduser()
            setattr(self, key, val)

    def _apply_env(self) -> None:
        env_map = {
            "MARKETSWARM_FRED_KEY": ("fred_api_key", str),
            "FRED_API_KEY": ("fred_api_key", str),
            "ANTHROPIC_API_KEY": ("anthropic_api_key", str),
            "MARKETSWARM_MODEL": ("llm_model", str),
            "MARKETSWARM_CONTACT": ("contact_email", str),
            "MARKETSWARM_WEBHOOK": ("webhook_url", str),
            "MARKETSWARM_DATA_DIR": ("data_dir", lambda v: Path(v).expanduser()),
            "MARKETSWARM_REPORT_DIR": ("report_dir", lambda v: Path(v).expanduser()),
            "MARKETSWARM_RUN_TIME": ("run_time_et", str),
            "MARKETSWARM_BASE_RISK": ("base_risk_pct", float),
            "MARKETSWARM_LLM_ENABLED": ("llm_enabled", lambda v: v.lower() not in ("0", "false", "no")),
        }
        for env, (attr, cast) in env_map.items():
            val = os.environ.get(env)
            if val:
                try:
                    setattr(self, attr, cast(val))
                except (TypeError, ValueError) as exc:
                    log.warning("bad value for %s: %s", env, exc)

        universe = os.environ.get("MARKETSWARM_UNIVERSE")
        if universe:
            self.universe = [s.strip().upper() for s in universe.split(",") if s.strip()]

    def to_yaml(self) -> str:
        import yaml
        payload = {
            "universe": self.universe,
            "index_symbols": self.index_symbols,
            "signal_correlation": self.signal_correlation,
            "base_risk_pct": self.base_risk_pct,
            "friction_r": self.friction_r,
            "max_concurrent_ideas": self.max_concurrent_ideas,
            "run_time_et": self.run_time_et,
            "score_time_et": self.score_time_et,
            "timeout_seconds": self.timeout_seconds,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "llm_model": self.llm_model,
            "llm_enabled": self.llm_enabled,
            "contact_email": self.contact_email,
            "notify_on": self.notify_on,
            "data_dir": str(self.data_dir),
            "report_dir": str(self.report_dir),
        }
        return yaml.safe_dump(payload, sort_keys=False)
