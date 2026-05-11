"""Walk-forward validation: rolling out-of-sample windows.

Splits the data into N (train, test) windows. Runs the backtest on each test
window — train segments aren't used to refit anything in this default config
(strategies are rule-based), but they're reserved for future use when params
are optimized per window.

The point: report performance on data the strategy *did not see during
parameter selection*. Wide gaps between in-sample and out-of-sample metrics
indicate overfitting.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import pandas as pd
from dateutil.relativedelta import relativedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import load_config
from utils.logging import setup_logging, get_logger
from data.feed import load_historical
from backtest.engine import Backtester
from backtest.metrics import compute_metrics, summary_str

log = get_logger(__name__)


@dataclass
class WFWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_windows(start: pd.Timestamp, end: pd.Timestamp,
                 train_months: int, test_months: int, n_windows: int) -> list[WFWindow]:
    """Anchored walk-forward: each window slides by `test_months`. Drops windows
    that would extend past `end`."""
    windows: list[WFWindow] = []
    cursor = start
    for _ in range(n_windows):
        ts = cursor
        te = cursor + relativedelta(months=train_months)
        ps = te
        pe = ps + relativedelta(months=test_months)
        if pe > end:
            break
        windows.append(WFWindow(ts, te, ps, pe))
        cursor = cursor + relativedelta(months=test_months)
    return windows


def run_walkforward(cfg: dict, ltf: pd.DataFrame, htf: pd.DataFrame, symbol: str,
                     asset_class: str, timeframe_low: str) -> list[dict]:
    wf_cfg = cfg["backtest"]["walkforward"]
    start = pd.Timestamp(cfg["backtest"]["start_date"], tz="UTC")
    end = pd.Timestamp(cfg["backtest"]["end_date"], tz="UTC")
    windows = make_windows(start, end, wf_cfg["train_months"], wf_cfg["test_months"],
                            wf_cfg["windows"])
    log.info("walk-forward: %d windows", len(windows))

    bt = Backtester(cfg)
    results: list[dict] = []
    for i, w in enumerate(windows, 1):
        trades, eq = bt.run(ltf, htf, symbol, asset_class, timeframe_low,
                             start_date=w.test_start.isoformat(),
                             end_date=w.test_end.isoformat())
        m = compute_metrics(trades, cfg["backtest"]["initial_equity"])
        m["window"] = i
        m["test_start"] = w.test_start.date().isoformat()
        m["test_end"] = w.test_end.date().isoformat()
        results.append(m)
        log.info("=== Window %d (%s → %s) ===", i, m["test_start"], m["test_end"])
        log.info("\n%s", summary_str(m))
    return results


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {}
    keys = ["win_rate", "profit_factor", "sharpe", "sortino",
            "max_drawdown_pct", "total_return_pct", "expectancy_R"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in results if r["trades"] > 0]
        if vals:
            agg[f"{k}_mean"] = sum(vals) / len(vals)
            agg[f"{k}_min"] = min(vals)
            agg[f"{k}_max"] = max(vals)
    agg["windows_with_trades"] = sum(1 for r in results if r["trades"] > 0)
    agg["total_trades"] = sum(r["trades"] for r in results)
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--data-ltf", required=True)
    p.add_argument("--data-htf", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--asset-class", default="crypto")
    p.add_argument("--timeframe-low", default="5m")
    args = p.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["level"], cfg["logging"]["file"])

    ltf = load_historical(args.data_ltf)
    htf = load_historical(args.data_htf)
    results = run_walkforward(cfg, ltf, htf, args.symbol, args.asset_class,
                                args.timeframe_low)
    log.info("=== Walk-forward aggregate ===")
    for k, v in aggregate(results).items():
        log.info("  %-30s %s", k, f"{v:.4f}" if isinstance(v, float) else v)


if __name__ == "__main__":
    main()
