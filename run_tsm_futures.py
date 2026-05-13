"""TSM-on-futures runner: same validated strategy at multiple leverage levels.

The point of this runner is to answer "what leverage can I actually trade
at?". Higher leverage → more profit in theory, but each leverage level has
to *survive the historical drawdown* without hitting margin call (= forced
liquidation). The output shows where the strategy blows up.

Usage:
    python run_tsm_futures.py \
        --data data/raw/gold_1d.csv --contract MGC \
        --leverage 1.0,2.0,3.0,5.0,10.0 \
        --equity 50000

    python run_tsm_futures.py --spec config/tsm_futures.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from backtest.tsm import TSMParams
from backtest.tsm_futures import (
    FuturesContract,
    simulate_tsm_futures,
    summarize_futures,
)
from data.feed import load_historical
from utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def _to_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    delta = df.index.to_series().diff().median()
    if delta < pd.Timedelta(days=1):
        return df.resample("1D").agg({"close": "last"}).dropna()
    return df


# ETF proxy → futures price ratios. ETFs trade at a fraction of the actual
# index/spot price (often by design, to keep the share price retail-friendly).
# The strategy P&L is scale-invariant, but contract sizing and margin
# tracking require the *true* futures price level. These defaults are
# rough and stable enough for TSM granularity calculations.
_DEFAULT_PROXY_SCALES = {
    "GC": 10.0,  "MGC": 10.0,    # GLD ≈ gold_spot / 10
    "NQ": 40.0,  "MNQ": 40.0,    # QQQ ≈ Nasdaq-100 / 40
    "ES": 10.0,  "MES": 10.0,    # SPY ≈ S&P 500 / 10
    "ZB": 1.0,   "ZN": 1.0,      # TLT/IEF roughly track underlying price
}


def load_contracts(path: str) -> dict[str, FuturesContract]:
    with open(path) as f:
        doc = yaml.safe_load(f)
    out: dict[str, FuturesContract] = {}
    for sym, spec in doc.get("contracts", {}).items():
        # Prefer YAML-specified scale; fall back to symbol-based default; final
        # fallback 1.0 means "proxy IS the underlying".
        scale = spec.get("proxy_price_scale")
        if scale is None:
            scale = _DEFAULT_PROXY_SCALES.get(sym, 1.0)
        out[sym] = FuturesContract(
            symbol=sym,
            multiplier=float(spec["multiplier"]),
            initial_margin=float(spec["initial_margin"]),
            commission_per_side=float(spec.get("commission_per_side", 2.5)),
            tick_size=float(spec.get("tick_size", 0.01)),
            tick_value=float(spec.get("tick_value", 0.0)),
            name=str(spec.get("name", "")),
            proxy_price_scale=float(scale),
        )
    return out


def run_leverage_sweep(close: pd.Series, contract: FuturesContract,
                        params: TSMParams, leverages: list[float],
                        initial_equity: float) -> pd.DataFrame:
    rows = []
    for lev in leverages:
        res = simulate_tsm_futures(close, contract, params, leverage=lev,
                                     initial_equity=initial_equity)
        s = summarize_futures(res)
        rows.append({
            "leverage": lev,
            "sharpe": s["sharpe"],
            "ann_return": s["annualized_return"],
            "ann_vol": s["annualized_vol"],
            "max_dd": s["max_dd"],
            "total_return": s["total_return"],
            "wiped_out": s["wiped_out"],
            "wipeout_date": s["wipeout_date"],
            "max_margin_pct": s["max_margin_pct"],
            "margin_breaches": s["margin_breaches"],
            "commissions": s["commissions_paid"],
        })
    return pd.DataFrame(rows)


def _print_sweep(df: pd.DataFrame, equity: float, contract_name: str) -> None:
    print(f"\nFutures contract: {contract_name}  |  Initial equity: ${equity:,.0f}")
    print("-" * 110)
    fmt = df.copy()
    for c in fmt.columns:
        if fmt[c].dtype.kind == "f":
            if c in {"ann_return", "ann_vol", "max_dd", "total_return", "max_margin_pct"}:
                fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
            elif c in {"sharpe"}:
                fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
            elif c in {"commissions"}:
                fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"${v:,.0f}")
            else:
                fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        elif fmt[c].dtype == bool:
            fmt[c] = fmt[c].map(lambda v: "WIPEOUT" if v else "")
    cols = ["leverage", "sharpe", "ann_return", "ann_vol", "max_dd",
            "total_return", "wiped_out", "max_margin_pct", "margin_breaches",
            "commissions"]
    print(fmt[cols].to_string(index=False))


def _print_advice(df: pd.DataFrame, contract: FuturesContract,
                    equity: float) -> None:
    # 'Practically trade-able' criteria — much stricter than just "no margin call":
    #   - No wipeouts
    #   - Zero margin breaches (any breach = forced liquidation in reality)
    #   - Max margin utilization < 50% (50%+ leaves no room for the *next* DD)
    #   - Max historical DD < 50% (deeper DDs are psychologically untradable
    #     for most retail and many institutional investors)
    practical = df[
        (df["wiped_out"] == False) &
        (df["margin_breaches"] == 0) &
        (df["max_margin_pct"] < 0.5) &
        (df["max_dd"] > -0.50)
    ]
    if practical.empty:
        print(f"\n⚠️  No leverage level meets the practical-deployment criteria at "
              f"${equity:,.0f}.")
        print(f"   Try a larger account, target_vol < 0.10, or 'survival' criteria")
        print(f"   (no margin call but accepting deeper DDs).")
        return

    best = practical.iloc[practical["sharpe"].idxmax()]
    # Recommended leverage = the practical max; production-recommended is half of that
    max_practical = practical.iloc[-1]
    half_lev = max_practical["leverage"] / 2.0
    print(f"\nRecommended leverage on {contract.symbol} at ${equity:,.0f}: "
          f"{max_practical['leverage']:.1f}× (practical max with DD < 50% and "
          f"no margin breaches)")
    print(f"  → Expected annual return: {max_practical['ann_return']:.1%}")
    print(f"  → Realized vol:           {max_practical['ann_vol']:.1%}")
    print(f"  → Historical max DD:      {max_practical['max_dd']:.1%}")
    print(f"  → Max margin utilization: {max_practical['max_margin_pct']:.1%}")
    print(f"\n  Conservative deploy: half of recommended ({half_lev:.1f}×). Headroom")
    print(f"  for the *next* drawdown to be larger than the historical worst case.")


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep leverage levels for TSM on futures.")
    p.add_argument("--data", help="Path to OHLCV proxy CSV (ETF or futures)")
    p.add_argument("--contract", help="Contract symbol from config/futures_contracts.yaml")
    p.add_argument("--symbol-label", help="Display label for the asset (e.g., 'Gold')")
    p.add_argument("--leverage", default="1.0,2.0,3.0,5.0,10.0",
                    help="Comma-separated leverage values to sweep")
    p.add_argument("--equity", type=float, default=100_000)
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument("--target-vol", type=float, default=0.10)
    p.add_argument("--max-leverage", type=float, default=2.0,
                    help="Cap on the vol-scaling alone (leverage stacks on top)")
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--skip", type=int, default=21)
    p.add_argument("--vol-lookback", type=int, default=60)
    p.add_argument("--rebalance", default="monthly",
                    choices=["daily", "weekly", "monthly"])
    p.add_argument("--multi-horizon",
                    help="Comma-separated lookback list (e.g. '63,126,252'). "
                         "When set, signal = avg of TSM signals at each lookback")
    p.add_argument("--contracts-file", default="config/futures_contracts.yaml")
    p.add_argument("--out", default="logs/tsm_futures")
    args = p.parse_args()

    setup_logging("INFO", "logs/tradebot.log")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.data or not args.contract:
        p.error("--data and --contract are required")

    contracts = load_contracts(args.contracts_file)
    if args.contract not in contracts:
        print(f"unknown contract {args.contract!r}; available: "
              f"{list(contracts)}", file=sys.stderr)
        return 1
    contract = contracts[args.contract]
    leverages = [float(x) for x in args.leverage.split(",")]

    df = load_historical(args.data)
    df = _to_daily(df)
    if args.start_date:
        df = df[df.index >= pd.Timestamp(args.start_date, tz="UTC")]
    if args.end_date:
        df = df[df.index <= pd.Timestamp(args.end_date, tz="UTC")]
    close = df["close"].astype(float)
    if len(close) == 0:
        print("no data after filtering", file=sys.stderr)
        return 1

    mh = ([int(x) for x in args.multi_horizon.split(",")]
            if args.multi_horizon else None)
    params = TSMParams(
        lookback_days=args.lookback,
        skip_days=args.skip,
        vol_lookback_days=args.vol_lookback,
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
        cost_bps_per_turnover=0.0,  # not used in futures sim
        rebalance=args.rebalance,
        multi_horizon_lookbacks=mh,
    )

    label = args.symbol_label or args.contract
    print(f"\n=== TSM-on-futures leverage sweep: {label} ===")
    print(f"Contract: {contract.symbol} ({contract.name}) "
          f"multiplier=${contract.multiplier:.0f}/pt "
          f"margin=${contract.initial_margin:,.0f}/contract")
    print(f"Data:     {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} daily bars)")
    sig_desc = (f"multi-horizon {params.multi_horizon_lookbacks}"
                  if params.multi_horizon_lookbacks
                  else f"single lookback={params.lookback_days}d")
    print(f"Params:   target_vol={params.target_vol:.0%}, signal={sig_desc}, "
          f"skip={params.skip_days}d, rebalance={params.rebalance}")

    sweep = run_leverage_sweep(close, contract, params, leverages, args.equity)
    sweep.to_csv(out_dir / f"{args.contract}_leverage_sweep.csv", index=False)

    _print_sweep(sweep, args.equity, contract.name or contract.symbol)
    _print_advice(sweep, contract, args.equity)

    print(f"\nSaved: {out_dir / f'{args.contract}_leverage_sweep.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
