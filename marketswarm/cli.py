"""Command-line interface.

    marketswarm run              pre-market research pass (the main job)
    marketswarm score            resolve and learn from past predictions
    marketswarm calibration      show the track record and calibration curve
    marketswarm daemon           long-running scheduler for a VPS
    marketswarm status           today's session, config, and readiness
    marketswarm init             write a starter config file
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import sys
from pathlib import Path

from . import clock
from .config import Config
from .llm import Narrator
from .memory import LearningEngine, MemoryStore
from .notify import send_webhook, should_notify, summarize
from .orchestrator import Swarm
from .providers.base import DataClient
from .providers.market import MarketData
from .report import write_reports

log = logging.getLogger("marketswarm")


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

async def cmd_run(args, cfg: Config) -> int:
    run_date = dt.date.fromisoformat(args.date) if args.date else clock.now_et().date()
    status = clock.day_status(run_date)

    if not status.is_trading_day and not args.force:
        print(f"\n{status.summary}")
        print(f"No pre-market session needed. Next trading day: "
              f"{clock.next_trading_day(run_date):%A, %d %B %Y}.\n")
        return 0

    swarm = Swarm(cfg)
    result = await swarm.run(run_date, force=args.force)

    narrative = None
    if cfg.llm_enabled and cfg.anthropic_api_key and not args.no_llm:
        narrative = Narrator(cfg.anthropic_api_key, cfg.llm_model).narrate(result)
        result.narrative = narrative

    paths = write_reports(result, cfg.report_dir, narrative)
    swarm.finalize(result, paths["markdown"])

    if args.json:
        print(json.dumps(
            {
                "date": result.run_date.isoformat(),
                "market_open": result.market_open,
                "probability": result.probability,
                "confidence": result.confidence,
                "ideas": result.ideas,
                "agents": {k: v.to_dict() for k, v in result.reports.items()},
                "reports": {k: str(v) for k, v in paths.items()},
            },
            indent=2, default=str,
        ))
    else:
        print(paths["markdown"].read_text(encoding="utf-8"))
        print(f"\n---\nMarkdown: {paths['markdown']}\nHTML: {paths['html']}", file=sys.stderr)

    if cfg.webhook_url and should_notify(result, cfg.notify_on):
        send_webhook(cfg.webhook_url, summarize(result))

    swarm.store.close()
    return 0


# --------------------------------------------------------------------------
# score  (the learning pass)
# --------------------------------------------------------------------------

async def cmd_score(args, cfg: Config) -> int:
    store = MemoryStore(cfg.db_path)
    engine = LearningEngine(store)

    target = dt.date.fromisoformat(args.date) if args.date else clock.now_et().date()
    pending = [r for r in store.unresolved_predictions(before_date=target.isoformat())]

    if not pending:
        print("No unresolved predictions to score.")
    else:
        by_date: dict[str, list] = {}
        for row in pending:
            by_date.setdefault(row["run_date"], []).append(row)

        resolved = 0
        async with DataClient(cache_dir=cfg.cache_dir, cache_ttl=3600,
                              user_agent=cfg.user_agent) as client:
            market = MarketData(client)
            for run_date, rows in sorted(by_date.items()):
                d = dt.date.fromisoformat(run_date)
                if d >= clock.now_et().date() and clock.session_at() not in (
                    clock.Session.AFTERHOURS, clock.Session.POST_CLOSE, clock.Session.OVERNIGHT
                ):
                    log.info("skipping %s — session not finished", run_date)
                    continue

                symbols = sorted({r["symbol"] for r in rows})
                bars = await client.gather(
                    [market.history(s, range_="5d", interval="5m") for s in symbols], label="score"
                )
                bar_map = {s: b for s, b in zip(symbols, bars) if b is not None}

                for row in rows:
                    hist = bar_map.get(row["symbol"])
                    if hist is None:
                        continue
                    day_bars = _bars_for_date(hist, d)
                    if not day_bars["highs"]:
                        continue
                    outcome = engine.resolve_prediction(row, day_bars)
                    if outcome:
                        store.resolve(row["id"], outcome[0], outcome[1], outcome[2])
                        resolved += 1
                        log.info("resolved %s %s: %s", row["run_date"], row["symbol"], outcome[2])
        print(f"Resolved {resolved} of {len(pending)} pending predictions.")

    result = engine.score_and_learn()
    print("\n" + result.verdict)
    if result.resolved >= 5:
        print(f"\nHit rate       {result.hit_rate:.1%}")
        print(f"Brier score    {result.brier:.4f}  (0.25 = always saying 50%)")
        print(f"Log loss       {result.log_loss:.4f}")
        print(f"Skill score    {result.skill_score:+.4f}  (>0 beats the base rate)")
        print(f"Expectancy     {result.expectancy_r:+.3f}R per idea")
        print(f"Recalibration  a={result.calibrator['a']:.3f} b={result.calibrator['b']:+.3f} "
              f"(fit on {result.calibrator['n_fit']} samples)")

    lessons = store.active_lessons()
    if lessons:
        print("\nLearned conditions:")
        for l in lessons:
            print(f"  • {l['lesson']}")

    if args.postmortem and cfg.anthropic_api_key:
        recent = [dict(r) for r in store.scored_history(limit=60)]
        text = Narrator(cfg.anthropic_api_key, cfg.llm_model).postmortem(
            engine.calibration_report(), recent
        )
        if text:
            print("\n--- Post-mortem ---\n" + text)

    store.close()
    return 0


def _bars_for_date(hist, day: dt.date) -> dict:
    """Slice intraday bars down to one regular-hours session."""
    highs, lows, closes = [], [], []
    for i, ts in enumerate(hist.timestamps):
        t = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).astimezone(clock.ET)
        if t.date() != day:
            continue
        if not (clock.REGULAR_OPEN <= t.time() <= clock.REGULAR_CLOSE):
            continue
        highs.append(float(hist.highs[i]))
        lows.append(float(hist.lows[i]))
        closes.append(float(hist.closes[i]))
    return {"highs": highs, "lows": lows, "closes": closes}


# --------------------------------------------------------------------------
# calibration / status / init
# --------------------------------------------------------------------------

def cmd_calibration(args, cfg: Config) -> int:
    store = MemoryStore(cfg.db_path)
    rep = LearningEngine(store).calibration_report(args.days)

    if rep.get("n", 0) < 5:
        print(f"Only {rep.get('n', 0)} resolved predictions — not enough to calibrate yet.")
        print("Run `marketswarm run` each morning and `marketswarm score` after each close.")
        store.close()
        return 0

    print(f"\nCalibration over {rep['n']} resolved predictions ({args.days} days)\n")
    print(f"  Mean forecast   {rep['mean_forecast']:.1%}")
    print(f"  Actual hit rate {rep['hit_rate']:.1%}")
    print(f"  Gap             {rep['mean_forecast'] - rep['hit_rate']:+.1%} "
          f"({'overconfident' if rep['mean_forecast'] > rep['hit_rate'] else 'under-confident'})")
    print(f"\n  Brier {rep['brier']:.4f} = reliability {rep['reliability']:.4f} "
          f"- resolution {rep['resolution']:.4f} + uncertainty {rep['uncertainty']:.4f}")
    print(f"  Skill score {rep['skill_score']:+.4f} — {rep['verdict']}")
    print(f"  SPRT: {rep['sprt']['decision']} (LLR {rep['sprt']['llr']:+.2f})")

    print("\n  Reliability curve")
    print("  bin        n    forecast  observed   gap")
    for row in rep["curve"]:
        if not row["n"]:
            continue
        print(f"  {row['bin']:<9} {row['n']:>4}   {row['mean_forecast']:>7.1%}  "
              f"{row['observed_rate']:>7.1%}  {row['gap']:>+6.1%}")

    c = rep["calibrator"]
    print(f"\n  Active recalibration: a={c['a']:.3f} b={c['b']:+.3f} (fit on {c['n_fit']})")

    perf = store.performance_summary(args.days)
    if perf.get("by_kind"):
        print("\n  By idea type")
        for kind, b in perf["by_kind"].items():
            print(f"    {kind:<14} n={b['n']:<4} hit {b['hit_rate']:.0%}  "
                  f"expectancy {b['expectancy_r']:+.2f}R  calibration gap {b['calibration_gap']:+.1%}")

    if rep.get("lessons"):
        print("\n  Learned conditions")
        for l in rep["lessons"]:
            print(f"    • {l['lesson']}")
    print()
    store.close()
    return 0


def cmd_status(args, cfg: Config) -> int:
    now = clock.now_et()
    today = now.date()
    status = clock.day_status(today)

    print(f"\nNow            {now:%A %d %B %Y, %H:%M:%S} ET")
    print(f"Session        {clock.session_at().value}")
    print(f"Today          {status.summary}")
    if status.is_trading_day:
        mins = clock.minutes_to_open()
        print(f"Open in        {mins:.0f} minutes" if mins > 0 else "Regular session is live or finished")
    print(f"Next trading   {clock.next_trading_day(today):%A %d %B %Y}")

    print(f"\nData dir       {cfg.data_dir}")
    print(f"Reports        {cfg.report_dir}")
    print(f"Database       {cfg.db_path} ({'exists' if cfg.db_path.exists() else 'not created yet'})")
    print(f"Universe       {len(cfg.universe)} symbols: {', '.join(cfg.universe[:10])}"
          + (" …" if len(cfg.universe) > 10 else ""))
    print(f"Schedule       run {cfg.run_time_et} ET, score {cfg.score_time_et} ET")

    print("\nIntegrations")
    print(f"  FRED         {'configured' if cfg.fred_api_key else 'not set (macro levels limited)'}")
    print(f"  Anthropic    {'configured' if cfg.anthropic_api_key else 'not set (no narrative)'}")
    print(f"  Webhook      {'configured' if cfg.webhook_url else 'not set'}")
    print(f"  SEC contact  {cfg.contact_email}")
    if cfg.contact_email.startswith("set "):
        print("               ⚠ set MARKETSWARM_CONTACT to a real email — SEC throttles anonymous clients")

    if cfg.db_path.exists():
        store = MemoryStore(cfg.db_path)
        perf = store.performance_summary(90)
        if perf.get("n"):
            print(f"\nTrack record   {perf['n']} resolved predictions in 90 days")
            for kind, b in perf["by_kind"].items():
                print(f"  {kind:<14} hit {b['hit_rate']:.0%}, expectancy {b['expectancy_r']:+.2f}R")
        else:
            print("\nTrack record   none yet")
        store.close()

    holidays = clock.holidays(today.year)
    upcoming = sorted(d for d in holidays if d >= today)[:3]
    if upcoming:
        print("\nUpcoming closures")
        for d in upcoming:
            print(f"  {d:%a %d %b %Y}  {holidays[d]}")
    print()
    return 0


def cmd_init(args, cfg: Config) -> int:
    path = Path(args.path).expanduser() if args.path else Path("~/.marketswarm/config.yaml").expanduser()
    if path.exists() and not args.force:
        print(f"{path} already exists — pass --force to overwrite.")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg.to_yaml(), encoding="utf-8")
    print(f"Wrote {path}")
    print("\nSet these in the environment (never in the config file):")
    print("  MARKETSWARM_CONTACT=you@example.com    # required by SEC EDGAR")
    print("  ANTHROPIC_API_KEY=sk-ant-...           # optional, enables the narrative")
    print("  FRED_API_KEY=...                       # optional, adds macro series")
    print("  MARKETSWARM_WEBHOOK=https://...        # optional, pushes the summary")
    return 0


def cmd_holidays(args, cfg: Config) -> int:
    year = args.year or clock.now_et().year
    print(f"\nNYSE full closures {year}")
    for d, name in sorted(clock.holidays(year).items()):
        print(f"  {d:%a %d %b}  {name}")
    print(f"\nEarly closes (13:00 ET) {year}")
    for d, name in sorted(clock.early_closes(year).items()):
        print(f"  {d:%a %d %b}  {name}")
    print()
    return 0


# --------------------------------------------------------------------------
# daemon
# --------------------------------------------------------------------------

async def cmd_daemon(args, cfg: Config) -> int:
    """Self-scheduling loop for a VPS.

    Sleeps until the next scheduled action rather than polling, so it costs
    nothing between runs and survives clock changes (times are resolved in ET
    each iteration, so DST is handled automatically).
    """
    log.info("daemon started — run %s ET, score %s ET", cfg.run_time_et, cfg.score_time_et)
    ran: set[tuple[str, str]] = set()

    while True:
        now = clock.now_et()
        today = now.date()
        key_run = (today.isoformat(), "run")
        key_score = (today.isoformat(), "score")

        run_at = _parse_hhmm(cfg.run_time_et)
        score_at = _parse_hhmm(cfg.score_time_et)
        trading = clock.is_trading_day(today)

        try:
            if trading and now.time() >= run_at and key_run not in ran:
                log.info("scheduled pre-market run")
                await cmd_run(_ns(date=None, force=False, json=False, no_llm=False), cfg)
                ran.add(key_run)
            elif trading and now.time() >= score_at and key_score not in ran:
                log.info("scheduled scoring pass")
                await cmd_score(_ns(date=None, postmortem=False), cfg)
                ran.add(key_score)
        except Exception:  # noqa: BLE001 — a bad day must not kill the daemon
            log.exception("scheduled task failed; continuing")
            ran.add(key_run if now.time() < score_at else key_score)

        sleep_for = _seconds_until_next(now, [run_at, score_at], trading)
        log.info("sleeping %.0f minutes", sleep_for / 60)
        await asyncio.sleep(sleep_for)
        # Trim the completed-task set so it cannot grow without bound.
        if len(ran) > 20:
            ran = {k for k in ran if k[0] >= (today - dt.timedelta(days=3)).isoformat()}


def _parse_hhmm(value: str) -> dt.time:
    try:
        h, m = (int(x) for x in value.split(":"))
        return dt.time(h, m)
    except (ValueError, AttributeError):
        log.warning("bad time %r — falling back to 08:15", value)
        return dt.time(8, 15)


def _seconds_until_next(now: dt.datetime, times: list[dt.time], trading_today: bool) -> float:
    candidates = []
    if trading_today:
        for t in times:
            target = dt.datetime.combine(now.date(), t, tzinfo=clock.ET)
            if target > now:
                candidates.append(target)
    nxt = clock.next_trading_day(now.date())
    candidates.append(dt.datetime.combine(nxt, min(times), tzinfo=clock.ET))
    soonest = min(candidates)
    return max(60.0, min((soonest - now).total_seconds(), 6 * 3600))


class _ns:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.__dict__.setdefault("date", None)
        self.__dict__.setdefault("force", False)
        self.__dict__.setdefault("json", False)
        self.__dict__.setdefault("no_llm", False)
        self.__dict__.setdefault("postmortem", False)


# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# historical data / backtest / validate / monitor
# --------------------------------------------------------------------------

def cmd_fetch(args, cfg: Config) -> int:
    from .backtest.datastore import PointInTimeStore
    from .backtest.fetch import load_into_store

    store = PointInTimeStore(cfg.data_dir / "history.db")
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    start = dt.date.fromisoformat(args.start) if args.start else None
    end = dt.date.fromisoformat(args.end) if args.end else None

    print(f"Fetching from {args.source}…")
    cov = load_into_store(store, args.source, symbols=symbols, start=start, end=end,
                          cache_dir=cfg.data_dir / "raw",
                          path=Path(args.path).expanduser() if args.path else None)
    print(f"\nStore now holds {cov['bars']:,} bars across {cov['symbols']} symbols "
          f"({cov['start']} → {cov['end']})")
    store.close()
    return 0


def cmd_backtest(args, cfg: Config) -> int:
    import json as _json
    import pickle

    import numpy as np

    from .backtest.datastore import PointInTimeStore
    from .backtest.features import build_dataset
    from .backtest.replay import BacktestEngine
    from .backtest.validation import deflated_sharpe_ratio

    store = PointInTimeStore(cfg.data_dir / "history.db")
    cov = store.coverage()
    if not cov["bars"]:
        print("No history. Run `marketswarm fetch --source github_sp500` first.")
        return 1

    available = set(store.symbols(min_bars=args.min_bars))
    universe = [s for s in (args.symbols.split(",") if args.symbols else cfg.universe)
                if s.strip().upper() in available]
    universe = [s.strip().upper() for s in universe]
    if not universe:
        print(f"None of the requested symbols have {args.min_bars}+ bars. "
              f"Available: {', '.join(sorted(available)[:20])}…")
        return 1

    start = dt.date.fromisoformat(args.start) if args.start else None
    end = dt.date.fromisoformat(args.end) if args.end else None
    dates = store.trading_dates(start, end, min_symbols=max(3, len(universe) // 4))
    print(f"Universe {len(universe)} symbols, {len(dates)} sessions "
          f"({dates[0]} → {dates[-1]})\n")

    cache = cfg.data_dir / "dataset.pkl"
    if args.reuse_dataset and cache.exists():
        print(f"Reusing cached dataset {cache}")
        d = pickle.load(open(cache, "rb"))
        X, y, meta = d["X"], d["y"], d["meta"]
    else:
        print("Building point-in-time features (no lookahead)…")
        X, y, meta = build_dataset(store, universe, dates,
                                   target_atr=args.target_atr, stop_atr=args.stop_atr)
        pickle.dump({"X": X, "y": y, "meta": meta, "universe": universe}, open(cache, "wb"))
    if len(X) == 0:
        print("No usable samples.")
        return 1
    print(f"{len(X):,} samples, base rate {y.mean():.1%}\n")

    engine = BacktestEngine(store, universe, target_atr=args.target_atr,
                            stop_atr=args.stop_atr, min_expected_r=args.min_expected_r,
                            mc_paths=args.mc_paths)

    print("=== Baselines the strategy must beat ===")
    for k, v in engine.run_baselines(X, y, meta).items():
        print(f"  {k}: {v}")

    print(f"\n=== Purged walk-forward ({args.folds} folds, {args.embargo}-day embargo) ===")
    res = engine.run_walk_forward(X, y, meta, n_splits=args.folds, embargo_days=args.embargo)
    summary = res.summary()

    if summary.get("n") == 0 or summary.get("n_taken", 0) == 0:
        print("\nNo trade cleared the expectancy gate in any fold.")
        print("That is a finding, not an error: on this data the strategy has no edge.")
        store.close()
        return 0

    net = summary["net"]
    cal = summary["calibration"]
    print(f"\n  Trades taken      {summary['n_taken']:,} of {summary['n_candidates']:,} "
          f"candidates ({summary['selectivity']:.1%})")
    print(f"  Net mean          {net['mean_r']:+.4f}R per trade")
    print(f"  Hit rate          {net['hit_rate']:.1%}")
    print(f"  Profit factor     {net['profit_factor']:.3f}")
    print(f"  Annualised Sharpe {net['sharpe_annual']:+.2f}")
    print(f"  Max drawdown      {net['max_drawdown_r']:.1f}R")
    print(f"  Cost drag         {summary['cost_drag_r']:.4f}R per trade")
    print(f"\n  Forecast {cal['mean_forecast']:.1%} vs actual {cal['actual_hit_rate']:.1%}")
    print(f"  Brier {cal['brier']:.4f}, skill {cal['skill_score']:+.4f} — {cal['verdict']}")

    if summary.get("by_regime"):
        print("\n  By regime")
        for reg, b in sorted(summary["by_regime"].items(), key=lambda kv: -kv[1]["n"]):
            print(f"    {reg:16} n={b['n']:5d}  net {b['mean_r']:+.4f}R  hit {b['hit_rate']:.1%}")

    if res.feature_importances:
        print("\n  Feature weights (log-odds per SD, final fold)")
        for nm, c in res.feature_importances[:10]:
            print(f"    {nm:>16} {c:+.4f}")

    r = res.returns(net=True)
    if len(r) >= 20:
        dsr = deflated_sharpe_ratio(r, n_trials=args.n_trials)
        print(f"\n=== Deflated Sharpe (n_trials={args.n_trials}) ===")
        print(f"  {dsr.verdict}")

    if args.out:
        Path(args.out).expanduser().write_text(_json.dumps(summary, indent=2, default=str))
        print(f"\nWrote {args.out}")

    print("\nReminder: a positive result here is necessary, not sufficient. Paper-trade "
          "before believing it.")
    store.close()
    return 0


async def cmd_monitor(args, cfg: Config) -> int:
    from .monitor import InvalidationMonitor, watch

    store = MemoryStore(cfg.db_path)
    if args.once:
        monitor = InvalidationMonitor(store)
        statuses = await monitor.check()
        if not statuses:
            print("No live ideas for today.")
        for s in statuses:
            print(s.line())
        store.close()
        return 0
    await watch(store, interval_seconds=args.interval, webhook=cfg.webhook_url)
    store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="marketswarm",
        description="Autonomous pre-market intelligence swarm (research/educational use only)",
    )
    p.add_argument("-c", "--config", help="path to config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--log-file", help="also write logs to this file")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the pre-market research pass")
    r.add_argument("--date", help="YYYY-MM-DD (defaults to today ET)")
    r.add_argument("--force", action="store_true", help="run even when the market is closed")
    r.add_argument("--json", action="store_true", help="emit JSON instead of the report")
    r.add_argument("--no-llm", action="store_true", help="skip the narrative layer")

    s = sub.add_parser("score", help="resolve past predictions and learn from them")
    s.add_argument("--date", help="score everything up to this date")
    s.add_argument("--postmortem", action="store_true", help="add an LLM post-mortem")

    c = sub.add_parser("calibration", help="show the track record and calibration curve")
    c.add_argument("--days", type=int, default=365)

    sub.add_parser("status", help="show session, config and readiness")
    sub.add_parser("daemon", help="run the scheduler in the foreground (for systemd)")

    i = sub.add_parser("init", help="write a starter config file")
    i.add_argument("path", nargs="?")
    i.add_argument("--force", action="store_true")


    f = sub.add_parser("fetch", help="download historical data into the point-in-time store")
    f.add_argument("--source", default="github_sp500",
                   choices=["github_sp500", "github_spy", "yahoo", "stooq", "csv"])
    f.add_argument("--symbols", help="comma-separated (required for yahoo/stooq)")
    f.add_argument("--start"); f.add_argument("--end"); f.add_argument("--path")

    b = sub.add_parser("backtest", help="purged walk-forward replay over stored history")
    b.add_argument("--symbols", help="comma-separated; defaults to the configured universe")
    b.add_argument("--start"); b.add_argument("--end")
    b.add_argument("--folds", type=int, default=5)
    b.add_argument("--embargo", type=int, default=5)
    b.add_argument("--target-atr", type=float, default=1.0, dest="target_atr")
    b.add_argument("--stop-atr", type=float, default=0.6, dest="stop_atr")
    b.add_argument("--min-expected-r", type=float, default=0.0, dest="min_expected_r")
    b.add_argument("--mc-paths", type=int, default=1200, dest="mc_paths")
    b.add_argument("--min-bars", type=int, default=400, dest="min_bars")
    b.add_argument("--n-trials", type=int, default=1, dest="n_trials",
                   help="how many configurations you have tried in total — be honest, "
                        "it is the deflation hurdle")
    b.add_argument("--reuse-dataset", action="store_true", dest="reuse_dataset")
    b.add_argument("--out", help="write the summary as JSON")

    m = sub.add_parser("monitor", help="watch today's ideas for invalidation")
    m.add_argument("--interval", type=int, default=300)
    m.add_argument("--once", action="store_true")

    h = sub.add_parser("holidays", help="list market closures")
    h.add_argument("year", nargs="?", type=int)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose, Path(args.log_file).expanduser() if args.log_file else None)
    cfg = Config.load(args.config)

    if args.command == "run":
        return asyncio.run(cmd_run(args, cfg))
    if args.command == "score":
        return asyncio.run(cmd_score(args, cfg))
    if args.command == "daemon":
        try:
            return asyncio.run(cmd_daemon(args, cfg))
        except KeyboardInterrupt:
            log.info("daemon stopped")
            return 0
    if args.command == "fetch":
        return cmd_fetch(args, cfg)
    if args.command == "backtest":
        return cmd_backtest(args, cfg)
    if args.command == "monitor":
        return asyncio.run(cmd_monitor(args, cfg))
    if args.command == "calibration":
        return cmd_calibration(args, cfg)
    if args.command == "status":
        return cmd_status(args, cfg)
    if args.command == "init":
        return cmd_init(args, cfg)
    if args.command == "holidays":
        return cmd_holidays(args, cfg)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
