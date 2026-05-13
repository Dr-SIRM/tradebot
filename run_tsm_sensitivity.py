"""Sensitivity sweep for time-series momentum on a single asset.

The TSM headline numbers from `run_tsm_backtest.py` used the literature defaults
(252/21/60). That's a defensible choice but it's still *one set of parameters
chosen with knowledge that the literature works*. To put real teeth in the
"parameter-free" claim, we run the strategy across a grid of plausible values
and apply two anti-overfitting corrections:

1. **Deflated Sharpe Ratio (DSR)** with `n_trials` = grid size. Penalizes the
   best Sharpe for the multiple-comparisons advantage of having tried many
   configs.

2. **Probability of Backtest Overfitting (PBO)** via Combinatorially Symmetric
   Cross-Validation (Bailey et al., 2014). Estimates the probability that the
   best-IS strategy is *below the median OOS* — a real edge should show PBO
   well under 10%.

Plus per-config stability stats: how many grid points were positive, the
distribution of Sharpe across the grid, etc.

Usage:
    python run_tsm_sensitivity.py --data data/raw/gold_1d.csv --symbol XAU/USD
    python run_tsm_sensitivity.py --data ... --grid grid_compact.yaml
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from backtest.robustness import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from backtest.tsm import TSMParams, simulate_tsm, summarize
from data.feed import load_historical
from utils.logging import get_logger, setup_logging

log = get_logger(__name__)


# Plausible TSM parameter ranges. These are *not* fitted — they're a reasonable
# search space drawn from the literature (MOP 2012 used 12-1m as the headline
# but tested 1-12 month lookbacks).
DEFAULT_GRID = {
    "lookback_days": [120, 180, 252, 365],          # ~6mo, 9mo, 12mo, 17mo
    "skip_days": [0, 10, 21, 42],                   # no skip, ~2wk, ~1mo, ~2mo
    "vol_lookback_days": [30, 60, 90],              # 1.5mo, 3mo, 4.5mo
}


def _to_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    delta = df.index.to_series().diff().median()
    if delta < pd.Timedelta(days=1):
        return df.resample("1D").agg({"close": "last"}).dropna()
    return df


def _grid_combos(grid: dict) -> list[dict]:
    """Cartesian product of the grid → list of param-override dicts."""
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*vals)]


def run_sensitivity(close: pd.Series, base_params: TSMParams,
                     grid: dict) -> dict:
    """Run TSM across every combination in `grid`. Returns a dict of arrays
    suitable for downstream PBO / DSR computation."""
    combos = _grid_combos(grid)
    rows: list[dict] = []
    rets_columns: list[pd.Series] = []
    col_names: list[str] = []

    for combo in combos:
        params = TSMParams(
            lookback_days=combo.get("lookback_days", base_params.lookback_days),
            skip_days=combo.get("skip_days", base_params.skip_days),
            vol_lookback_days=combo.get("vol_lookback_days", base_params.vol_lookback_days),
            target_vol=base_params.target_vol,
            max_leverage=base_params.max_leverage,
            cost_bps_per_turnover=base_params.cost_bps_per_turnover,
            rebalance=base_params.rebalance,
            initial_equity=base_params.initial_equity,
        )
        sim = simulate_tsm(close, params)
        summary = summarize(sim)
        row = dict(combo)
        row.update({k: summary[k] for k in
                     ("sharpe", "annualized_return", "annualized_vol",
                      "max_dd", "total_return")})
        rows.append(row)
        rets_columns.append(sim["daily_returns"])
        col_names.append(
            f"lb{combo['lookback_days']}_skip{combo['skip_days']}"
            f"_vlb{combo['vol_lookback_days']}"
        )

    rets_df = pd.concat(rets_columns, axis=1).fillna(0.0)
    rets_df.columns = col_names
    rows_df = pd.DataFrame(rows)

    return {
        "rows": rows_df,
        "returns_matrix": rets_df,
        "n_strategies": len(rows),
    }


def analyze_sensitivity(res: dict, pbo_blocks: int = 8) -> dict:
    """Compute DSR (deflated for n_trials) and PBO on the sensitivity output."""
    rows = res["rows"]
    rets = res["returns_matrix"]
    n_trials = res["n_strategies"]

    # 1. Stability statistics across the grid
    sharpes = rows["sharpe"].values
    stability = {
        "n_configs": int(n_trials),
        "sharpe_p10": float(np.percentile(sharpes, 10)),
        "sharpe_p50": float(np.percentile(sharpes, 50)),
        "sharpe_p90": float(np.percentile(sharpes, 90)),
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_std": float(np.std(sharpes)),
        "frac_positive_sharpe": float(np.mean(sharpes > 0)),
        "best_sharpe": float(sharpes.max()),
        "worst_sharpe": float(sharpes.min()),
    }

    # 2. Deflated Sharpe Ratio at the BEST config, deflated for n_trials
    best_col_idx = int(np.argmax(sharpes))
    best_rets = rets.iloc[:, best_col_idx]
    # Drop pre-warmup zeros so the SR estimate is on the live period
    live = best_rets[best_rets != 0]
    if len(live) >= 30:
        dsr = deflated_sharpe_ratio(live, n_trials=n_trials, periods_per_year=252)
        stability.update({
            "dsr_at_best": float(dsr.deflated_sharpe),
            "dsr_observed_sharpe": float(dsr.observed_sharpe),
            "dsr_expected_max_sharpe": float(dsr.expected_max_sharpe),
            "dsr_n_observations": int(dsr.n_observations),
        })

    # 3. PBO via CSCV
    # Trim to live rows (drop the all-zero pre-warmup rows shared by all configs)
    nonzero_rows = (rets != 0).any(axis=1)
    rets_live = rets[nonzero_rows].values  # T × N
    if rets_live.shape[0] >= pbo_blocks and rets_live.shape[1] >= 2:
        pbo = probability_of_backtest_overfitting(rets_live, n_blocks=pbo_blocks)
        stability.update({
            "pbo": float(pbo.pbo),
            "pbo_n_splits": int(pbo.n_splits),
        })
    else:
        log.warning("insufficient data for PBO: shape=%s", rets_live.shape)

    return stability


def _print_topk(rows: pd.DataFrame, k: int = 5) -> None:
    keep = ["lookback_days", "skip_days", "vol_lookback_days",
            "sharpe", "annualized_return", "annualized_vol",
            "max_dd", "total_return"]
    keep = [c for c in keep if c in rows.columns]
    top = rows.sort_values("sharpe", ascending=False).head(k)
    print("\nTop-K by Sharpe:")
    print(top[keep].to_string(index=False,
                                formatters={
                                    "sharpe": "{:.3f}".format,
                                    "annualized_return": "{:.2%}".format,
                                    "annualized_vol": "{:.2%}".format,
                                    "max_dd": "{:.2%}".format,
                                    "total_return": "{:.2%}".format,
                                }))


def _print_verdict(stability: dict) -> None:
    dsr = stability.get("dsr_at_best", float("nan"))
    pbo = stability.get("pbo", float("nan"))
    frac_pos = stability.get("frac_positive_sharpe", 0.0)
    med_sharpe = stability.get("sharpe_p50", 0.0)
    worst_sharpe = stability.get("worst_sharpe", float("nan"))

    # ROBUST: a real edge as a *class* — almost every parameter setting works.
    # High PBO here just means parameter selection is noisy; use the literature
    # defaults rather than optimizing.
    if frac_pos >= 0.95 and (not np.isnan(dsr) and dsr >= 0.90) and med_sharpe > 0.4:
        v = "ROBUST"
        notes = (f"All (or nearly all) parameter configs are profitable; DSR "
                 f"survives deflation. The strategy class works. "
                 f"PBO={pbo:.2f} → don't try to optimize parameters; use the "
                 f"literature defaults and expect ~median Sharpe ({med_sharpe:.2f}) "
                 f"going forward, not the best-IS Sharpe ({stability['best_sharpe']:.2f}).")
    # BORDERLINE: edge exists but spread is wider; some configs lose
    elif frac_pos >= 0.70 and (not np.isnan(dsr) and dsr >= 0.80):
        v = "BORDERLINE"
        notes = (f"Most configs are profitable but {(1-frac_pos)*100:.0f}% lose money "
                 f"(worst Sharpe {worst_sharpe:.2f}). Edge is real but parameter "
                 f"choice matters more — don't deploy without longer OOS validation.")
    # OVERFIT: best is high but median is near zero — selection bias dominates
    elif (not np.isnan(pbo) and pbo > 0.50 and med_sharpe < 0.2 and frac_pos < 0.70):
        v = "OVERFIT"
        notes = ("Best Sharpe is likely an artifact of multiple comparisons; "
                 "most parameter choices don't reproduce the win.")
    else:
        v = "WEAK"
        notes = "Some edge survives but not enough to deploy."
    print(f"\nFinal verdict: {v}")
    print(f"  {notes}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="TSM parameter-sensitivity sweep with PBO + deflated DSR."
    )
    p.add_argument("--data", required=True, help="Path to OHLCV CSV")
    p.add_argument("--symbol", required=True)
    p.add_argument("--grid", help="Optional YAML file overriding the default grid")
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument("--target-vol", type=float, default=0.10)
    p.add_argument("--max-leverage", type=float, default=2.0)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--rebalance", default="monthly",
                    choices=["daily", "weekly", "monthly"])
    p.add_argument("--pbo-blocks", type=int, default=8)
    p.add_argument("--out", default="logs/tsm_sensitivity")
    args = p.parse_args()

    setup_logging("INFO", "logs/tradebot.log")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = DEFAULT_GRID
    if args.grid:
        with open(args.grid) as f:
            grid = yaml.safe_load(f)

    df = load_historical(args.data)
    df = _to_daily(df)
    if args.start_date:
        df = df[df.index >= pd.Timestamp(args.start_date, tz="UTC")]
    if args.end_date:
        df = df[df.index <= pd.Timestamp(args.end_date, tz="UTC")]
    close = df["close"].astype(float)

    base = TSMParams(
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
        cost_bps_per_turnover=args.cost_bps,
        rebalance=args.rebalance,
    )

    print(f"\n=== TSM sensitivity sweep: {args.symbol} ===")
    print(f"Data: {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} bars)")
    print(f"Grid: lookback={grid['lookback_days']} skip={grid['skip_days']} "
          f"vol_lb={grid['vol_lookback_days']}")
    print(f"      ({np.prod([len(v) for v in grid.values()])} combinations)\n")

    res = run_sensitivity(close, base, grid)
    stability = analyze_sensitivity(res, pbo_blocks=args.pbo_blocks)

    print("Stability across the grid:")
    print(f"  configs                {stability['n_configs']}")
    print(f"  best Sharpe            {stability['best_sharpe']:>6.3f}")
    print(f"  median Sharpe          {stability['sharpe_p50']:>6.3f}")
    print(f"  worst Sharpe           {stability['worst_sharpe']:>6.3f}")
    print(f"  p10 / p90              {stability['sharpe_p10']:>6.3f} / {stability['sharpe_p90']:.3f}")
    print(f"  frac with positive SR  {stability['frac_positive_sharpe']:>6.1%}")
    if "dsr_at_best" in stability:
        print(f"\nDeflated Sharpe (at best config, deflated for {stability['n_configs']} trials):")
        print(f"  observed Sharpe        {stability['dsr_observed_sharpe']:.3f}")
        print(f"  expected max under H0  {stability['dsr_expected_max_sharpe']:.3f}")
        print(f"  deflated DSR           {stability['dsr_at_best']:.3f}  "
              f"(P[true SR > 0])")
    if "pbo" in stability:
        verdict = ("HEALTHY"   if stability["pbo"] < 0.10
                   else "MODERATE" if stability["pbo"] < 0.30
                   else "SEVERE")
        print(f"\nProbability of Backtest Overfitting (CSCV):")
        print(f"  PBO                    {stability['pbo']:.3f}  → {verdict}")
        print(f"  splits evaluated       {stability['pbo_n_splits']}")

    _print_topk(res["rows"], k=8)
    _print_verdict(stability)

    res["rows"].to_csv(out_dir / f"{args.symbol.replace('/', '_')}_grid.csv",
                        index=False)
    pd.Series(stability).to_csv(out_dir / f"{args.symbol.replace('/', '_')}_stability.csv",
                                  header=["value"])
    print(f"\nDetails written to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
