"""Asset sweep: run the full backtest + robustness pipeline across many assets
and produce a single ranked comparison table.

The point: when the default strategies fail on one asset (e.g. BTC), this lets
you quickly check whether they have *any* edge somewhere else without manually
running run_backtest.py N times and copy-pasting numbers. Output is sorted by
Deflated Sharpe Ratio so the most promising asset (if any) floats to the top.

Usage:
    # Define assets via a YAML spec (recommended for repeatability)
    python run_asset_sweep.py --spec config/asset_sweep.yaml --out logs/sweep

    # Or via repeated --asset flags on the CLI:
    python run_asset_sweep.py \
        --asset data/raw/btc_1h.csv:BTC/USDT:crypto:1h:4h \
        --asset data/raw/spy_1d.csv:SPY:equity:1d:1d \
        --asset data/raw/eurusd_1h.csv:EUR/USD:forex:1h:4h \
        --out logs/sweep

Each --asset is colon-separated: <csv_path>:<symbol>:<asset_class>:<tf_low>:<tf_high>.
Use tf_high == tf_low when you only have one timeframe (e.g. daily data); the
HTF is then a copy of the LTF.

A spec YAML looks like:
    start_date: "2020-01-01"
    end_date: "2026-05-11"
    pbo_blocks: 8        # smaller if shorter histories
    sensitivity: true    # also run the parameter grid for PBO
    assets:
      - data: data/raw/btc_1h.csv
        symbol: BTC/USDT
        asset_class: crypto
        tf_low: 1h
        tf_high: 4h
      - data: data/raw/spy_1d.csv
        symbol: SPY
        asset_class: equity
        tf_low: 1d
        tf_high: 1d
"""
from __future__ import annotations

import argparse
import copy
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from backtest.engine import Backtester
from backtest.metrics import compute_metrics
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
from utils.logging import get_logger, setup_logging

log = get_logger(__name__)

_TF_TO_PANDAS = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D",
}


@dataclass
class AssetSpec:
    data: str
    symbol: str
    asset_class: str
    tf_low: str
    tf_high: str

    @classmethod
    def from_cli(cls, s: str) -> "AssetSpec":
        parts = s.split(":")
        if len(parts) != 5:
            raise ValueError(
                f"--asset must be 'path:symbol:class:tf_low:tf_high'; got {s!r}"
            )
        return cls(*parts)


def _resample_to_htf(ltf: pd.DataFrame, tf_high: str) -> pd.DataFrame:
    rule = _TF_TO_PANDAS.get(tf_high, tf_high)
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return ltf.resample(rule).agg(agg).dropna()


def evaluate_asset(
    spec: AssetSpec,
    base_cfg: dict,
    start_date: Optional[str],
    end_date: Optional[str],
    run_sensitivity_flag: bool,
    pbo_blocks: int,
    mc_runs: int,
) -> dict:
    """Run in-sample + walkforward + MC + (optional) sensitivity & PBO for one asset.

    Returns a flat dict of summary metrics so the caller can stack them into a
    comparison DataFrame.
    """
    cfg = copy.deepcopy(base_cfg)
    if start_date:
        cfg["backtest"]["start_date"] = start_date
    if end_date:
        cfg["backtest"]["end_date"] = end_date

    row: dict = {
        "symbol": spec.symbol,
        "asset_class": spec.asset_class,
        "tf_low": spec.tf_low,
        "tf_high": spec.tf_high,
        "data": spec.data,
    }

    try:
        ltf = load_historical(spec.data)
        if ltf.empty:
            row["error"] = "empty data file"
            return row
        if spec.tf_high == spec.tf_low:
            htf = ltf.copy()
        else:
            htf = _resample_to_htf(ltf, spec.tf_high)

        row["bars_ltf"] = len(ltf)
        row["bars_htf"] = len(htf)
        row["date_start"] = str(ltf.index[0].date())
        row["date_end"] = str(ltf.index[-1].date())

        bt = Backtester(cfg)
        trades, equity = bt.run(
            ltf, htf, spec.symbol, spec.asset_class, spec.tf_low,
            start_date=start_date, end_date=end_date,
        )
        metrics = compute_metrics(trades, cfg["backtest"]["initial_equity"])
        row.update({
            "trades": metrics["trades"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "sharpe": metrics["sharpe"],
            "sortino": metrics["sortino"],
            "expectancy_R": metrics["expectancy_R"],
            "total_return_pct": metrics["total_return_pct"],
            "max_dd_pct": metrics["max_drawdown_pct"],
        })

        # Walk-forward (only meaningful with a long enough window)
        try:
            wf = run_walkforward(cfg, ltf, htf, spec.symbol,
                                  spec.asset_class, spec.tf_low)
            agg = aggregate(wf)
            row["wf_sharpe_mean"] = agg.get("sharpe_mean", float("nan"))
            row["wf_pf_mean"] = agg.get("profit_factor_mean", float("nan"))
            row["wf_return_mean"] = agg.get("total_return_pct_mean", float("nan"))
            row["wf_windows_with_trades"] = agg.get("windows_with_trades", 0)
        except Exception as e:
            log.warning("[%s] walkforward failed: %s", spec.symbol, e)
            row["wf_sharpe_mean"] = float("nan")

        # Monte Carlo
        if trades:
            mc = run_monte_carlo(trades, starting_equity=cfg["backtest"]["initial_equity"],
                                 n_runs=mc_runs, seed=42)
            row["mc_p_profitable"] = mc.prob_profitable
            row["mc_p_ruin"] = mc.prob_ruin
            row["mc_dd_p95"] = mc.max_dd_p95

        # Sensitivity (needed for honest PBO + an honest DSR n_trials)
        sens = None
        if run_sensitivity_flag:
            grid = cfg.get("backtest", {}).get("sensitivity", {}).get("grid", {})
            if grid:
                sens = run_sensitivity(ltf, htf, cfg, grid, spec.symbol,
                                       spec.asset_class, spec.tf_low)

        # Deflated Sharpe Ratio
        if len(equity) >= 5:
            eq_daily = equity.resample("1D").last().ffill()
            rets = eq_daily.pct_change().dropna()
            n_trials = len(sens.rows) if sens is not None and sens.rows else 1
            dsr = deflated_sharpe_ratio(rets, n_trials=n_trials, periods_per_year=252)
            row["dsr"] = dsr.deflated_sharpe
            row["n_trials"] = dsr.n_trials

        # PBO requires sensitivity
        if sens is not None and len(sens.equity_curves) >= 2:
            mat = returns_matrix_from_sensitivity(sens.rows, sens.equity_curves, "1D")
            if mat.shape[0] >= pbo_blocks and mat.shape[1] >= 2:
                pbo = probability_of_backtest_overfitting(mat, n_blocks=pbo_blocks)
                row["pbo"] = pbo.pbo

    except Exception as e:
        log.error("[%s] evaluation failed: %s\n%s",
                  spec.symbol, e, traceback.format_exc())
        row["error"] = str(e)

    return row


def _verdict(row: dict) -> str:
    """One-word read on a single asset's robustness output."""
    if "error" in row:
        return "ERROR"
    dsr = row.get("dsr")
    pbo = row.get("pbo")
    wf = row.get("wf_sharpe_mean")
    if dsr is None:
        return "NO_DATA"
    # NO_EDGE flags first: a clear-cut failure overrides any high DSR.
    if dsr < 0.50 or (pbo is not None and pbo > 0.50) or (wf is not None and wf < 0):
        return "NO_EDGE"
    if dsr >= 0.95 and (pbo is None or pbo < 0.10):
        return "PROMISING"
    if dsr >= 0.80 and (pbo is None or pbo < 0.30):
        return "INVESTIGATE"
    return "INCONCLUSIVE"


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no rows)")
        return
    df = pd.DataFrame(rows)
    # Stable column order, with a verdict column up front for readability
    df["verdict"] = df.apply(lambda r: _verdict(r.to_dict()), axis=1)
    cols = ["verdict", "symbol", "tf_low", "trades",
            "sharpe", "wf_sharpe_mean", "mc_p_profitable",
            "dsr", "pbo", "max_dd_pct", "total_return_pct"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    # Format floats compactly
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].map(
                lambda v: "" if pd.isna(v) else (f"{v:.2%}" if c in {
                    "win_rate", "total_return_pct", "max_dd_pct",
                    "mc_p_profitable", "mc_p_ruin"
                } else f"{v:.3f}")
            )
    # Sort: PROMISING > INVESTIGATE > INCONCLUSIVE > NO_EDGE > ERROR
    order = {"PROMISING": 0, "INVESTIGATE": 1, "INCONCLUSIVE": 2,
             "NO_EDGE": 3, "NO_DATA": 4, "ERROR": 5}
    df = df.sort_values("verdict", key=lambda s: s.map(order)).reset_index(drop=True)
    print(df.to_string(index=False))


def main() -> int:
    p = argparse.ArgumentParser(description="Run the backtest pipeline across many assets.")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--spec", help="YAML file describing the sweep (see module docstring)")
    p.add_argument("--asset", action="append", default=[],
                   help="Inline asset spec 'path:symbol:class:tf_low:tf_high' (repeatable)")
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument("--sensitivity", action="store_true",
                   help="Run sensitivity grid per asset for honest DSR/PBO")
    p.add_argument("--pbo-blocks", type=int, default=8)
    p.add_argument("--mc-runs", type=int, default=1000)
    p.add_argument("--out", default="logs/asset_sweep")
    args = p.parse_args()

    base_cfg = load_config(args.config)
    setup_logging(base_cfg.get("logging", {}).get("level", "INFO"),
                  base_cfg.get("logging", {}).get("file"))

    # Build asset list from --spec and/or --asset flags
    specs: list[AssetSpec] = []
    start = args.start_date
    end = args.end_date
    sensitivity_flag = args.sensitivity
    pbo_blocks = args.pbo_blocks

    if args.spec:
        with open(args.spec) as f:
            doc = yaml.safe_load(f)
        start = start or doc.get("start_date")
        end = end or doc.get("end_date")
        if "sensitivity" in doc:
            sensitivity_flag = bool(doc["sensitivity"])
        if "pbo_blocks" in doc:
            pbo_blocks = int(doc["pbo_blocks"])
        for a in doc.get("assets", []):
            specs.append(AssetSpec(
                data=a["data"], symbol=a["symbol"], asset_class=a["asset_class"],
                tf_low=a["tf_low"], tf_high=a["tf_high"],
            ))
    for s in args.asset:
        specs.append(AssetSpec.from_cli(s))

    if not specs:
        print("no assets provided; use --spec or --asset", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i, spec in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}] {spec.symbol} ({spec.asset_class}) "
              f"{spec.tf_low}→{spec.tf_high}  ...")
        row = evaluate_asset(spec, base_cfg, start, end,
                              sensitivity_flag, pbo_blocks, args.mc_runs)
        rows.append(row)
        # Persist incrementally so a crash doesn't lose prior results
        pd.DataFrame(rows).to_csv(out_dir / "sweep.csv", index=False)

    print("\n=== Asset sweep results (sorted by verdict) ===")
    _print_table(rows)
    print(f"\nFull results: {out_dir / 'sweep.csv'}")
    print("\nVerdict legend:")
    print("  PROMISING    DSR>=0.95, PBO<0.10, OOS Sharpe positive — investigate further")
    print("  INVESTIGATE  DSR>=0.80, PBO<0.30 — borderline, worth a closer look")
    print("  INCONCLUSIVE neither clearly real nor clearly fake")
    print("  NO_EDGE      DSR<0.50 or PBO>0.50 or OOS Sharpe negative — don't trade")
    return 0


if __name__ == "__main__":
    sys.exit(main())
