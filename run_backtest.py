"""CLI entry point for backtesting, walk-forward, Monte Carlo, sensitivity.

The Backtester needs both LTF and HTF bars (multi-timeframe). Provide
either two data files via --data-ltf / --data-htf, or a single LTF file
via --data and the HTF will be resampled from it.

Usage:
    # Single LTF file (HTF auto-resampled)
    python run_backtest.py --data data/btc_5m.csv --symbol BTC/USDT \
        --asset-class crypto --timeframe-low 5m --timeframe-high 1h

    # Separate LTF and HTF files
    python run_backtest.py --data-ltf data/btc_5m.csv --data-htf data/btc_1h.csv \
        --symbol BTC/USDT --asset-class crypto

    # Walk-forward + Monte Carlo + sensitivity
    python run_backtest.py --data-ltf ... --data-htf ... --symbol BTC/USDT \
        --walkforward --monte-carlo --sensitivity
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from backtest.engine import Backtester
from backtest.metrics import compute_metrics, summary_str
from backtest.montecarlo import run_monte_carlo
from backtest.robustness import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    returns_matrix_from_sensitivity,
)
from backtest.sensitivity import run_sensitivity
from backtest.walkforward import run_walkforward, aggregate
from data.feed import load_historical
from utils.config import load_config
from utils.logging import setup_logging, get_logger

log = get_logger(__name__)


_TF_TO_PANDAS = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D",
}


def _resample_to_htf(ltf: pd.DataFrame, tf_high: str) -> pd.DataFrame:
    """Resample lower-timeframe bars to higher timeframe."""
    rule = _TF_TO_PANDAS.get(tf_high, tf_high)
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return ltf.resample(rule).agg(agg).dropna()


def main() -> int:
    p = argparse.ArgumentParser(description="Backtest the trading bot.")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--data", help="LTF data file; HTF auto-resampled")
    p.add_argument("--data-ltf", help="LTF data file (alternative to --data)")
    p.add_argument("--data-htf", help="HTF data file (otherwise resampled from LTF)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--asset-class", default="crypto")
    p.add_argument("--timeframe-low", default="5m")
    p.add_argument("--timeframe-high", default="1h")
    p.add_argument("--walkforward", action="store_true")
    p.add_argument("--monte-carlo", action="store_true")
    p.add_argument("--mc-runs", type=int, default=1000)
    p.add_argument("--sensitivity", action="store_true")
    p.add_argument("--robustness", action="store_true",
                   help="Compute Deflated Sharpe and (with --sensitivity) PBO")
    p.add_argument("--pbo-blocks", type=int, default=16,
                   help="Number of CSCV blocks for PBO (must be even, default 16)")
    p.add_argument("--out", default="logs/backtest_results")
    args = p.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"),
                  cfg.get("logging", {}).get("file"))

    # Resolve data files
    ltf_path = args.data_ltf or args.data
    if not ltf_path:
        log.error("must provide --data or --data-ltf")
        return 1

    ltf = load_historical(ltf_path)
    if ltf.empty:
        log.error("no data loaded from %s", ltf_path)
        return 1

    if args.data_htf:
        htf = load_historical(args.data_htf)
    else:
        log.info("resampling HTF from LTF (%s -> %s)", args.timeframe_low, args.timeframe_high)
        htf = _resample_to_htf(ltf, args.timeframe_high)

    log.info("LTF: %d bars (%s → %s)", len(ltf), ltf.index[0], ltf.index[-1])
    log.info("HTF: %d bars (%s → %s)", len(htf), htf.index[0], htf.index[-1])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    starting_equity = cfg["backtest"]["initial_equity"]

    # In-sample backtest
    bt = Backtester(cfg)
    trades, equity = bt.run(ltf, htf, args.symbol, args.asset_class, args.timeframe_low)

    metrics = compute_metrics(trades, starting_equity)
    print("\n=== In-sample backtest ===")
    print(summary_str(metrics))

    if trades:
        pd.DataFrame([{
            "symbol": t.symbol, "side": t.side.value, "strategy": t.strategy,
            "entry_time": t.entry_time, "exit_time": t.exit_time,
            "entry_price": t.entry_price, "exit_price": t.exit_price,
            "qty": t.quantity, "pnl": t.pnl, "r_multiple": t.r_multiple,
            "exit_reason": t.exit_reason,
        } for t in trades]).to_csv(out / "trades.csv", index=False)
    if len(equity):
        equity.to_csv(out / "equity.csv", header=["equity"])
    log.info("saved trades and equity to %s", out)

    # Walk-forward
    if args.walkforward:
        print("\n=== Walk-forward validation ===")
        try:
            results = run_walkforward(cfg, ltf, htf, args.symbol,
                                      args.asset_class, args.timeframe_low)
            agg = aggregate(results)
            print("\nWalk-forward aggregate:")
            for k, v in agg.items():
                print(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")
            pd.DataFrame(results).to_csv(out / "walkforward.csv", index=False)
        except Exception as e:
            log.exception("walk-forward failed: %s", e)

    # Monte Carlo
    if args.monte_carlo:
        print("\n=== Monte Carlo simulation ===")
        if not trades:
            print("(no trades from in-sample run, skipping)")
        else:
            mc = run_monte_carlo(trades, starting_equity=starting_equity,
                                 n_runs=args.mc_runs, seed=42)
            print(mc.summary())

    # Sensitivity
    sens = None
    if args.sensitivity:
        print("\n=== Parameter sensitivity ===")
        grid = cfg.get("backtest", {}).get("sensitivity", {}).get("grid", {})
        if not grid:
            print("(no sensitivity grid configured)")
        else:
            sens = run_sensitivity(ltf, htf, cfg, grid, args.symbol,
                                   args.asset_class, args.timeframe_low)
            print(sens.summary())
            sens.to_dataframe().to_csv(out / "sensitivity.csv", index=False)

    # Robustness: Deflated Sharpe + Probability of Backtest Overfitting
    if args.robustness:
        print("\n=== Robustness (anti-overfitting) ===")
        if len(equity) < 5:
            print("(not enough equity points; need a longer backtest)")
        else:
            # Daily returns for the in-sample run
            eq_daily = equity.resample("1D").last().ffill()
            rets = eq_daily.pct_change().dropna()
            # How many configs did we effectively try? If sensitivity ran, that's N.
            # Otherwise we assume 1 (still useful: it just won't deflate much).
            n_trials = len(sens.rows) if sens is not None and sens.rows else 1
            dsr = deflated_sharpe_ratio(rets, n_trials=n_trials,
                                         periods_per_year=252)
            print(dsr.summary())
            pd.DataFrame([{
                "observed_sharpe": dsr.observed_sharpe,
                "expected_max_sharpe": dsr.expected_max_sharpe,
                "deflated_sharpe": dsr.deflated_sharpe,
                "n_trials": dsr.n_trials,
                "n_observations": dsr.n_observations,
                "skew": dsr.skew,
                "kurtosis_excess": dsr.kurtosis_excess,
            }]).to_csv(out / "deflated_sharpe.csv", index=False)

        if sens is not None and len(sens.equity_curves) >= 2:
            print()
            ret_mat = returns_matrix_from_sensitivity(
                sens.rows, sens.equity_curves, resample="1D"
            )
            if ret_mat.shape[0] < args.pbo_blocks or ret_mat.shape[1] < 2:
                print(f"(insufficient data for PBO: shape={ret_mat.shape}, "
                      f"need >= {args.pbo_blocks} rows × 2 cols)")
            else:
                pbo = probability_of_backtest_overfitting(
                    ret_mat, n_blocks=args.pbo_blocks
                )
                print(pbo.summary())
                pd.DataFrame([{
                    "pbo": pbo.pbo,
                    "n_strategies": pbo.n_strategies,
                    "n_blocks": pbo.n_blocks,
                    "n_splits": pbo.n_splits,
                }]).to_csv(out / "pbo.csv", index=False)
        elif args.robustness:
            print("(PBO requires --sensitivity with >= 2 configs)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
