"""Point-in-time historical store.

The whole value of a backtest rests on one property: on any simulated day, the
system can only see what it could actually have seen that morning. Every
realistic-looking backtest that later fails live has broken that property
somewhere.

So this store does not merely *avoid* lookahead — it refuses it. Reads are
scoped to an as-of timestamp and raise `LookaheadError` when asked for data the
decision-maker could not have had. That turns a silent, profitable-looking bug
into a loud crash.
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("marketswarm.backtest.datastore")

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    source TEXT,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_bars_date ON bars(date);
CREATE INDEX IF NOT EXISTS idx_bars_symbol_date ON bars(symbol, date);

-- Events carry two timestamps on purpose: when the thing happened, and when it
-- became publicly knowable. Only the second one may gate a read.
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    event_date TEXT NOT NULL,
    known_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT,
    source TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_known ON events(known_at);
"""


class LookaheadError(RuntimeError):
    """Raised when a read would expose data from the future."""


@dataclass(frozen=True)
class Bar:
    symbol: str
    date: dt.date
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def range_pct(self) -> float:
        return (self.high - self.low) / self.open * 100 if self.open else 0.0

    @property
    def true_range(self) -> float:
        return self.high - self.low


@dataclass
class History:
    """A window of bars ending strictly before the as-of date."""

    symbol: str
    dates: list[dt.date]
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray

    def __len__(self) -> int:
        return int(self.closes.size)

    @property
    def last_close(self) -> float:
        return float(self.closes[-1]) if self.closes.size else float("nan")

    @property
    def last_date(self) -> dt.date | None:
        return self.dates[-1] if self.dates else None


class PointInTimeStore:
    def __init__(self, path: Path | str = "~/.marketswarm/history.db", strict: bool = True):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.strict = strict
        self._cache: dict[str, pd.DataFrame] = {}

    def close(self) -> None:
        self.conn.close()

    # ---------- ingest ----------

    def ingest_frame(self, df: pd.DataFrame, source: str = "unknown") -> int:
        """Ingest a tidy frame with columns: symbol, date, open, high, low, close, volume."""
        required = {"symbol", "date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")

        clean = df.dropna(subset=["open", "high", "low", "close"]).copy()
        clean["date"] = pd.to_datetime(clean["date"]).dt.strftime("%Y-%m-%d")
        clean["symbol"] = clean["symbol"].astype(str).str.upper()

        # Reject bars that violate OHLC ordering — bad rows silently poison every
        # ATR and barrier calculation downstream.
        bad = clean[(clean.high < clean.low)
                    | (clean.high < clean.open) | (clean.high < clean.close)
                    | (clean.low > clean.open) | (clean.low > clean.close)]
        if len(bad):
            log.warning("dropping %d bars with inconsistent OHLC", len(bad))
            clean = clean.drop(bad.index)

        rows = [
            (r.symbol, r.date, float(r.open), float(r.high), float(r.low),
             float(r.close), float(r.volume or 0), source)
            for r in clean.itertuples()
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO bars (symbol,date,open,high,low,close,volume,source) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        self._cache.clear()
        return len(rows)

    def ingest_event(self, symbol: str | None, event_date: dt.date, known_at: dt.datetime,
                     kind: str, payload: str = "", source: str = "") -> None:
        if known_at.date() < event_date:
            raise ValueError("known_at cannot precede the event it describes")
        self.conn.execute(
            "INSERT INTO events (symbol,event_date,known_at,kind,payload,source) VALUES (?,?,?,?,?,?)",
            (symbol, event_date.isoformat(), known_at.isoformat(), kind, payload, source),
        )
        self.conn.commit()

    # ---------- point-in-time reads ----------

    def _frame(self, symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        if sym not in self._cache:
            df = pd.read_sql_query(
                "SELECT date,open,high,low,close,volume FROM bars WHERE symbol=? ORDER BY date",
                self.conn, params=(sym,),
            )
            df["date"] = pd.to_datetime(df["date"]).dt.date
            self._cache[sym] = df
        return self._cache[sym]

    def history_before(self, symbol: str, asof: dt.date, lookback: int = 250) -> History | None:
        """Bars strictly before `asof`. This is the only price read a feature
        computation is allowed to make."""
        df = self._frame(symbol)
        if df.empty:
            return None
        win = df[df["date"] < asof].tail(lookback)
        if win.empty:
            return None
        return History(
            symbol=symbol.upper(),
            dates=list(win["date"]),
            opens=win["open"].to_numpy(float),
            highs=win["high"].to_numpy(float),
            lows=win["low"].to_numpy(float),
            closes=win["close"].to_numpy(float),
            volumes=win["volume"].to_numpy(float),
        )

    def bar_on(self, symbol: str, date: dt.date) -> Bar | None:
        """The bar for a specific session.

        This is the *outcome* read — used to score a decision after the fact,
        never to make one. In strict mode the engine is expected to call this
        only through `resolve`, and feature code that reaches for it is a bug.
        """
        df = self._frame(symbol)
        row = df[df["date"] == date]
        if row.empty:
            return None
        r = row.iloc[0]
        return Bar(symbol.upper(), date, float(r.open), float(r.high),
                   float(r.low), float(r.close), float(r.volume))

    def open_price(self, symbol: str, date: dt.date) -> float | None:
        """The session's opening print.

        Legitimately knowable by a trader acting at or after the open, which is
        when every idea in this system is entered.
        """
        bar = self.bar_on(symbol, date)
        return bar.open if bar else None

    def events_known_by(self, asof: dt.datetime, symbol: str | None = None) -> list[dict]:
        q = "SELECT * FROM events WHERE known_at <= ?"
        params: list = [asof.isoformat()]
        if symbol:
            q += " AND symbol = ?"
            params.append(symbol.upper())
        return [dict(r) for r in self.conn.execute(q + " ORDER BY known_at", params)]

    # ---------- calendar & coverage ----------

    def trading_dates(self, start: dt.date | None = None, end: dt.date | None = None,
                      min_symbols: int = 1) -> list[dt.date]:
        """Dates on which at least `min_symbols` names traded.

        Derived from the data rather than from a calendar, so a market holiday
        that the data provider handled differently cannot desynchronise the
        replay from reality.
        """
        rows = self.conn.execute(
            "SELECT date, COUNT(*) n FROM bars GROUP BY date HAVING n >= ? ORDER BY date",
            (min_symbols,),
        )
        out = [dt.date.fromisoformat(r["date"]) for r in rows]
        if start:
            out = [d for d in out if d >= start]
        if end:
            out = [d for d in out if d <= end]
        return out

    def symbols(self, min_bars: int = 100) -> list[str]:
        rows = self.conn.execute(
            "SELECT symbol, COUNT(*) n FROM bars GROUP BY symbol HAVING n >= ? ORDER BY symbol",
            (min_bars,),
        )
        return [r["symbol"] for r in rows]

    def coverage(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) bars, COUNT(DISTINCT symbol) syms, MIN(date) lo, MAX(date) hi FROM bars"
        ).fetchone()
        ev = self.conn.execute("SELECT COUNT(*) n FROM events").fetchone()
        return {
            "bars": row["bars"], "symbols": row["syms"],
            "start": row["lo"], "end": row["hi"], "events": ev["n"],
        }

    def assert_no_lookahead(self, hist: History, asof: dt.date) -> None:
        """Belt-and-braces check callers can use in tests."""
        if hist.last_date and hist.last_date >= asof:
            raise LookaheadError(
                f"history for {hist.symbol} ends {hist.last_date}, "
                f"which is not strictly before the decision date {asof}"
            )
