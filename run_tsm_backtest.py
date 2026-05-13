"""Run time-series momentum across many daily price series.

Standalone runner — does NOT use the per-bar engine. The Strategy interface
doesn't fit TSM (long-hold, no stop/TP, signal-driven exits), so we simulate
directly via backtest.tsm.

Why TSM and not the default momentum/MR/breakout strategies:
  Moskowitz, Ooi & Pedersen (2012) document Sharpe ~1.0 across 58 instruments
  using exactly this 12-1 month momentum approach. It's the lowest-fitted,
  best-evidenced rule-based signal we can test. If the simple version doesn't
  show up here, it's time to consider whether the asset basket itself is the
  problem — not the strategy logic.

Usage:
    python run_tsm_backtest.py --spec config/tsm_sweep.yaml

    python run_tsm_backtest.py \
        --asset data/raw/btc_1d.csv:BTC/USDT \
        --asset data/raw/qqq_1d.csv:QQQ

Asset spec: 'csv_path:symbol' (no asset class needed — TSM has no per-class config).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from backtest.montecarlo import run_monte_carlo
from backtest.robustness import deflated_sharpe_ratio
from backtest.tsm import (
    TSMParams,
    simulate_tsm,
    simulate_tsm_portfolio,
    summarize,
)
from data.feed import load_historical
from utils.logging import get_logger, setup_logging
from utils.types import Side, Trade

log = get_logger(__name__)


@dataclass
class AssetSpec:
    data: str
    symbol: str

    @classmethod
    def from_cli(cls, s: str) -> "AssetSpec":
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"--asset must be 'path:symbol'; got {s!r}")
        return cls(parts[0], parts[1])


def _to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """If df is intraday, resample close to daily. Otherwise return as-is."""
    if df.empty:
        return df
    delta = df.index.to_series().diff().median()
    if delta < pd.Timedelta(days=1):
        return df.resample("1D").agg({"close": "last"}).dropna()
    return df


def evaluate_tsm(spec: AssetSpec, params: TSMParams,
                  start_date: Optional[str], end_date: Optional[str]) -> dict:
    """Run TSM on one asset and return a summary row."""
    row: dict = {"symbol": spec.symbol, "data": spec.data}
    try:
        if not Path(spec.data).exists():
            row["error"] = "missing data file"
            return row
        df = load_historical(spec.data)
        if df.empty:
            row["error"] = "empty data file"
            return row
        df = _to_daily(df)
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]
        if len(df) < params.lookback_days + params.skip_days + params.vol_lookback_days + 30:
            row["error"] = f"insufficient data: {len(df)} bars"
            return row

        close = df["close"].astype(float)
        sim = simulate_tsm(close, params)
        summary = summarize(sim)
        row.update({
            "bars": len(df),
            "date_start": str(df.index[0].date()),
            "date_end": str(df.index[-1].date()),
            "rebalances": summary["n_trades"],
            "sharpe": summary["sharpe"],
            "annualized_return": summary["annualized_return"],
            "annualized_vol": summary["annualized_vol"],
            "total_return": summary["total_return"],
            "max_dd": summary["max_dd"],
            "turnover_total": summary["turnover_total"],
        })

        # DSR with n_trials=1 (parameter-free strategy — nothing was optimized)
        rets = sim["daily_returns"]
        rets_for_dsr = rets[rets != 0]
        if len(rets_for_dsr) >= 30:
            dsr = deflated_sharpe_ratio(rets_for_dsr, n_trials=1, periods_per_year=252)
            row["dsr"] = dsr.deflated_sharpe

        # Monte Carlo bootstrap on rebalance-period returns (~monthly).
        # We synthesize Trade objects so we can reuse run_monte_carlo; each
        # rebalance period is a "trade" with the period's P&L in dollars.
        if sim["trades"]:
            mc_trades = []
            eq_path = sim["equity"]
            prev_eq = params.initial_equity
            for t in sim["trades"]:
                period_eq = float(eq_path.loc[t["rebalance_time"]])
                pnl = period_eq - prev_eq
                pnl_pct = pnl / prev_eq if prev_eq > 0 else 0.0
                mc_trades.append(Trade(
                    symbol=spec.symbol,
                    asset_class="multi",
                    side=Side.LONG if t["new_position"] > 0 else Side.SHORT,
                    strategy="tsm",
                    entry_time=t["rebalance_time"].to_pydatetime(),
                    exit_time=t["rebalance_time"].to_pydatetime(),
                    entry_price=0.0, exit_price=0.0, quantity=1.0,
                    pnl=pnl, pnl_pct=pnl_pct,
                    r_multiple=0.0, exit_reason="rebalance",
                ))
                prev_eq = period_eq
            if len(mc_trades) >= 10:
                mc = run_monte_carlo(mc_trades, starting_equity=params.initial_equity,
                                      n_runs=1000, seed=42)
                row["mc_p_profitable"] = mc.prob_profitable
                row["mc_p_ruin"] = mc.prob_ruin

    except Exception as e:
        log.error("[%s] TSM eval failed: %s\n%s", spec.symbol, e, traceback.format_exc())
        row["error"] = str(e)
    return row


def evaluate_portfolio(specs: list[AssetSpec], params: TSMParams,
                        start_date: Optional[str], end_date: Optional[str],
                        target_portfolio_vol: float | None,
                        weighting: str = "equal") -> dict:
    """Build closes dict, run the combined-portfolio simulation, and return a
    row with the same shape as evaluate_tsm()."""
    row: dict = {"symbol": "PORTFOLIO", "data": "(combined)"}
    try:
        closes: dict[str, pd.Series] = {}
        skipped_missing: list[str] = []
        for spec in specs:
            if not Path(spec.data).exists():
                skipped_missing.append(spec.symbol)
                continue
            df = load_historical(spec.data)
            if df.empty:
                continue
            df = _to_daily(df)
            if start_date:
                df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]
            if len(df) < params.lookback_days + params.skip_days + params.vol_lookback_days + 30:
                continue
            closes[spec.symbol] = df["close"].astype(float)
        if skipped_missing:
            log.info("[PORTFOLIO] skipped %d missing CSVs: %s",
                      len(skipped_missing), ", ".join(skipped_missing))
        if not closes:
            row["error"] = "no eligible assets after filtering"
            return row

        port = simulate_tsm_portfolio(closes, params,
                                       target_portfolio_vol=target_portfolio_vol,
                                       weighting=weighting)
        summary = summarize(port)
        row.update({
            "bars": int(len(port["daily_returns"])),
            "n_assets": len(closes),
            "avg_n_live": float(port["n_live"].mean()),
            "sharpe": summary["sharpe"],
            "annualized_return": summary["annualized_return"],
            "annualized_vol": summary["annualized_vol"],
            "total_return": summary["total_return"],
            "max_dd": summary["max_dd"],
        })

        rets = port["daily_returns"]
        rets_for_dsr = rets[rets != 0]
        if len(rets_for_dsr) >= 30:
            dsr = deflated_sharpe_ratio(rets_for_dsr, n_trials=1, periods_per_year=252)
            row["dsr"] = dsr.deflated_sharpe

        # MC bootstrap on monthly portfolio returns
        monthly = port["daily_returns"].resample("ME") \
            .apply(lambda s: (1 + s).prod() - 1).dropna()
        if len(monthly) >= 10:
            mc_trades = []
            for ts, r in monthly.items():
                pnl = params.initial_equity * float(r)
                mc_trades.append(Trade(
                    symbol="PORTFOLIO", asset_class="multi",
                    side=Side.LONG, strategy="tsm-portfolio",
                    entry_time=ts.to_pydatetime(), exit_time=ts.to_pydatetime(),
                    entry_price=0.0, exit_price=0.0, quantity=1.0,
                    pnl=pnl, pnl_pct=float(r), r_multiple=0.0,
                    exit_reason="rebalance",
                ))
            mc = run_monte_carlo(mc_trades, starting_equity=params.initial_equity,
                                  n_runs=1000, seed=42)
            row["mc_p_profitable"] = mc.prob_profitable
            row["mc_p_ruin"] = mc.prob_ruin

    except Exception as e:
        log.error("[PORTFOLIO] eval failed: %s\n%s", e, traceback.format_exc())
        row["error"] = str(e)
    return row


def _verdict(row: dict) -> str:
    err = row.get("error")
    if isinstance(err, str) and err:
        return "ERROR"
    sharpe = row.get("sharpe")
    if sharpe is None or (isinstance(sharpe, float) and pd.isna(sharpe)):
        return "NO_DATA"
    dsr = row.get("dsr", 0.0)
    if sharpe >= 0.7 and dsr >= 0.85:
        return "PROMISING"
    if sharpe >= 0.4 and dsr >= 0.70:
        return "INVESTIGATE"
    if sharpe < 0:
        return "NEGATIVE"
    return "WEAK"


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no rows)")
        return
    df = pd.DataFrame(rows)
    df["verdict"] = df.apply(lambda r: _verdict(r.to_dict()), axis=1)
    cols = ["verdict", "symbol", "bars", "rebalances", "sharpe",
            "annualized_return", "annualized_vol", "max_dd",
            "total_return", "dsr", "mc_p_profitable"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].map(
                lambda v: "" if pd.isna(v) else (
                    f"{v:.2%}" if c in {"annualized_return", "annualized_vol",
                                         "max_dd", "total_return", "mc_p_profitable"}
                    else f"{v:.3f}"
                )
            )
    order = {"PROMISING": 0, "INVESTIGATE": 1, "WEAK": 2, "NEGATIVE": 3,
             "NO_DATA": 4, "ERROR": 5}
    df = df.sort_values("verdict", key=lambda s: s.map(order)).reset_index(drop=True)
    print(df.to_string(index=False))


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep time-series momentum across assets.")
    p.add_argument("--spec", help="YAML spec with assets + TSM params")
    p.add_argument("--asset", action="append", default=[],
                   help="Inline 'path:symbol' (repeatable)")
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--skip", type=int, default=21)
    p.add_argument("--vol-lookback", type=int, default=60)
    p.add_argument("--target-vol", type=float, default=0.10)
    p.add_argument("--max-leverage", type=float, default=2.0)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--rebalance", default="monthly", choices=["daily", "weekly", "monthly"])
    p.add_argument("--portfolio", action="store_true",
                   help="Also evaluate the equal-weight portfolio of all assets")
    p.add_argument("--portfolio-vol-target", type=float, default=0.10,
                   help="Portfolio realized-vol target for the rescale (0 disables)")
    p.add_argument("--portfolio-weighting", default="equal",
                   choices=["equal", "inverse_vol", "sharpe"],
                   help="How to weight assets in the portfolio")
    p.add_argument("--multi-horizon",
                   help="Comma-separated lookback list (e.g. '63,126,252'). "
                        "When set, signal = avg of TSM signals at each lookback")
    p.add_argument("--derisk-dd-threshold", type=float, default=0.0,
                   help="Halve positions when running DD exceeds this (0 = off, "
                        "typical 0.10-0.15)")
    p.add_argument("--derisk-scale", type=float, default=0.5,
                   help="Multiply positions by this factor while derisked")
    p.add_argument("--derisk-recovery", type=float, default=0.05,
                   help="Restore full size when DD recovers under this level")
    p.add_argument("--out", default="logs/tsm_sweep")
    args = p.parse_args()

    setup_logging("INFO", "logs/tradebot.log")

    specs: list[AssetSpec] = []
    start = args.start_date
    end = args.end_date
    mh = ([int(x) for x in args.multi_horizon.split(",")]
            if args.multi_horizon else None)
    params = TSMParams(
        lookback_days=args.lookback,
        skip_days=args.skip,
        vol_lookback_days=args.vol_lookback,
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
        cost_bps_per_turnover=args.cost_bps,
        rebalance=args.rebalance,
        multi_horizon_lookbacks=mh,
        derisk_dd_threshold=args.derisk_dd_threshold,
        derisk_scale=args.derisk_scale,
        derisk_recovery_threshold=args.derisk_recovery,
    )

    portfolio_flag = args.portfolio
    portfolio_vol_target = args.portfolio_vol_target if args.portfolio_vol_target > 0 else None
    portfolio_weighting = args.portfolio_weighting

    if args.spec:
        with open(args.spec) as f:
            doc = yaml.safe_load(f)
        start = start or doc.get("start_date")
        end = end or doc.get("end_date")
        if "portfolio" in doc:
            portfolio_flag = bool(doc["portfolio"])
        if "portfolio_vol_target" in doc:
            v = doc["portfolio_vol_target"]
            portfolio_vol_target = float(v) if v and v > 0 else None
        if "portfolio_weighting" in doc:
            portfolio_weighting = doc["portfolio_weighting"]
        tsm_overrides = doc.get("tsm", {}) or {}
        params = TSMParams(
            lookback_days=tsm_overrides.get("lookback_days", params.lookback_days),
            skip_days=tsm_overrides.get("skip_days", params.skip_days),
            vol_lookback_days=tsm_overrides.get("vol_lookback_days", params.vol_lookback_days),
            target_vol=tsm_overrides.get("target_vol", params.target_vol),
            max_leverage=tsm_overrides.get("max_leverage", params.max_leverage),
            cost_bps_per_turnover=tsm_overrides.get("cost_bps_per_turnover", params.cost_bps_per_turnover),
            rebalance=tsm_overrides.get("rebalance", params.rebalance),
            multi_horizon_lookbacks=tsm_overrides.get("multi_horizon_lookbacks",
                                                       params.multi_horizon_lookbacks),
            derisk_dd_threshold=tsm_overrides.get("derisk_dd_threshold",
                                                    params.derisk_dd_threshold),
            derisk_scale=tsm_overrides.get("derisk_scale", params.derisk_scale),
            derisk_recovery_threshold=tsm_overrides.get(
                "derisk_recovery_threshold", params.derisk_recovery_threshold),
        )
        for a in doc.get("assets", []):
            specs.append(AssetSpec(data=a["data"], symbol=a["symbol"]))
    for s in args.asset:
        specs.append(AssetSpec.from_cli(s))

    if not specs:
        print("no assets provided; use --spec or --asset", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sig_desc = (f"multi-horizon {params.multi_horizon_lookbacks}"
                  if params.multi_horizon_lookbacks
                  else f"single lookback={params.lookback_days}")
    print(f"\nTSM params: {sig_desc} skip={params.skip_days} "
          f"vol_lb={params.vol_lookback_days} target_vol={params.target_vol:.0%} "
          f"max_lev={params.max_leverage} costs={params.cost_bps_per_turnover}bps "
          f"rebal={params.rebalance}")
    if params.derisk_dd_threshold > 0:
        print(f"DD-derisk: threshold={params.derisk_dd_threshold:.0%}, "
              f"scale={params.derisk_scale}, "
              f"recovery={params.derisk_recovery_threshold:.0%}")
    if portfolio_flag:
        print(f"Portfolio: weighting={portfolio_weighting} "
              f"target_vol={portfolio_vol_target}")
    print()

    rows: list[dict] = []
    for i, spec in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] {spec.symbol} ...")
        row = evaluate_tsm(spec, params, start, end)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "tsm_sweep.csv", index=False)

    if portfolio_flag and len(specs) >= 2:
        print(f"\n[PORTFOLIO] combining {len(specs)} assets "
              f"(weighting={portfolio_weighting}, "
              f"target vol={portfolio_vol_target}) ...")
        port_row = evaluate_portfolio(specs, params, start, end,
                                        portfolio_vol_target,
                                        weighting=portfolio_weighting)
        rows.append(port_row)
        pd.DataFrame(rows).to_csv(out_dir / "tsm_sweep.csv", index=False)

    print("\n=== Time-series momentum results (sorted by verdict) ===")
    _print_table(rows)
    print(f"\nFull results: {out_dir / 'tsm_sweep.csv'}")
    print("\nVerdict legend:")
    print("  PROMISING    Sharpe>=0.7, DSR>=0.85 — real edge worth pursuing")
    print("  INVESTIGATE  Sharpe>=0.4, DSR>=0.70 — borderline")
    print("  WEAK         Positive but uncompelling")
    print("  NEGATIVE     Sharpe < 0 — TSM doesn't work here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
