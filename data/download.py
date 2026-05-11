"""Real OHLCV data downloader.

Two backends:
  - CCXT for crypto (binance, bybit, coinbase, kraken, ...). Paginates through
    the exchange's history endpoint until the requested range is filled.
  - yfinance for equities/ETFs/FX/commodities. Limited to the lookback windows
    yfinance allows (e.g. 1m: 7 days, 5m: 60 days, 1h: 730 days, 1d: years).

The output schema matches what `data.feed.load_historical` expects:
  index = UTC DatetimeIndex named 'timestamp'
  columns = open, high, low, close, volume

Usage:
    python -m data.download crypto BTC/USDT 5m 2023-01-01 2024-12-31 -o data/btc_5m.csv
    python -m data.download equity SPY 1h 2023-01-01 2024-12-31 -o data/spy_1h.csv

The downloader is intentionally rate-limit-aware and resumable: if you pass
--resume and the output file exists, only the missing tail is fetched.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


_REQUIRED_COLS = ("open", "high", "low", "close", "volume")


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


def _ensure_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("expected DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    for c in _REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"missing column {c!r} in downloaded data")
    df = df[list(_REQUIRED_COLS)].dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def download_crypto(symbol: str, timeframe: str, start: str | pd.Timestamp,
                    end: str | pd.Timestamp, exchange_id: str = "binance",
                    sleep_ms: int = 250) -> pd.DataFrame:
    """Fetch crypto OHLCV via CCXT, paginating to cover [start, end]."""
    import ccxt

    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"unknown exchange: {exchange_id}")
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    if timeframe not in ex.timeframes:
        raise ValueError(f"{exchange_id} does not support timeframe {timeframe}; "
                         f"options: {sorted(ex.timeframes)}")

    start_ts = _ensure_utc(start)
    end_ts = _ensure_utc(end)
    since = _ms(start_ts)
    end_ms = _ms(end_ts)

    rows: list[list] = []
    last_ts: Optional[int] = None
    page_limit = 1000  # most exchanges cap around here

    while since < end_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe,
                               since=since, limit=page_limit)
        if not batch:
            break
        rows.extend(batch)
        new_last = batch[-1][0]
        # Detect no-progress (exchange returned a stale page) to avoid infinite loop
        if last_ts is not None and new_last <= last_ts:
            break
        last_ts = new_last
        # Advance past this bar to avoid duplicate on the next page
        since = new_last + 1
        time.sleep(sleep_ms / 1000.0)

    if not rows:
        return pd.DataFrame(columns=list(_REQUIRED_COLS),
                            index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))

    df = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.drop(columns=["ts_ms"]).set_index("timestamp")
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]
    return _normalize_frame(df)


_YF_TF_MAP = {
    "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m", "30m": "30m",
    "60m": "60m", "1h": "60m", "90m": "90m",
    "1d": "1d", "5d": "5d", "1wk": "1wk",
}


def download_equity(symbol: str, timeframe: str, start: str | pd.Timestamp,
                    end: str | pd.Timestamp) -> pd.DataFrame:
    """Fetch equity/ETF/FX/commodity OHLCV via yfinance.

    yfinance imposes lookback caps per interval — caller is responsible for
    keeping the requested window within those limits (1m=7d, 5m=60d, 1h=730d).
    """
    import yfinance as yf

    yf_tf = _YF_TF_MAP.get(timeframe)
    if yf_tf is None:
        raise ValueError(f"yfinance does not support timeframe {timeframe}; "
                         f"options: {sorted(_YF_TF_MAP)}")
    start_ts = _ensure_utc(start)
    end_ts = _ensure_utc(end)

    df = yf.download(symbol, start=start_ts.tz_convert(None),
                     end=end_ts.tz_convert(None), interval=yf_tf,
                     auto_adjust=False, progress=False, threads=False)
    if df.empty:
        return pd.DataFrame(columns=list(_REQUIRED_COLS),
                            index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))
    # Recent yfinance returns MultiIndex columns when threads=False on some
    # versions — flatten if so.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    if "adj close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj close": "close"})
    return _normalize_frame(df)


def _resume_window(out_path: Path, start: pd.Timestamp,
                   end: pd.Timestamp) -> tuple[pd.Timestamp, pd.DataFrame]:
    """If out_path exists, return (new_start, existing_df). Else (start, empty)."""
    if not out_path.exists():
        return start, pd.DataFrame()
    existing = pd.read_csv(out_path)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    existing = existing.set_index("timestamp").sort_index()
    if existing.empty:
        return start, existing
    last = existing.index[-1]
    if last >= end:
        return end, existing  # nothing to do
    return last + pd.Timedelta(seconds=1), existing


def main() -> int:
    p = argparse.ArgumentParser(description="Download real OHLCV data.")
    p.add_argument("kind", choices=["crypto", "equity"],
                   help="crypto → CCXT exchange; equity → yfinance")
    p.add_argument("symbol", help="e.g. BTC/USDT, ETH/USDT, SPY, EURUSD=X")
    p.add_argument("timeframe", help="e.g. 1m 5m 15m 1h 1d")
    p.add_argument("start", help="UTC start date YYYY-MM-DD")
    p.add_argument("end", help="UTC end date YYYY-MM-DD (exclusive of next day)")
    p.add_argument("-o", "--out", required=True, help="Output CSV path")
    p.add_argument("--exchange", default="binance", help="CCXT exchange id (crypto only)")
    p.add_argument("--resume", action="store_true",
                   help="Append from where the existing file left off")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_ts = _ensure_utc(args.start)
    end_ts = _ensure_utc(args.end)

    if args.resume:
        start_ts, existing = _resume_window(out_path, start_ts, end_ts)
        if start_ts >= end_ts:
            print(f"already up to date: {out_path} (last bar ≥ {args.end})")
            return 0
        print(f"resuming from {start_ts.isoformat()}")
    else:
        existing = pd.DataFrame()

    print(f"downloading {args.kind} {args.symbol} {args.timeframe} "
          f"{start_ts.date()} → {end_ts.date()}")
    if args.kind == "crypto":
        new = download_crypto(args.symbol, args.timeframe, start_ts, end_ts,
                              exchange_id=args.exchange)
    else:
        new = download_equity(args.symbol, args.timeframe, start_ts, end_ts)

    if new.empty:
        print("no new bars returned")
        if existing.empty:
            return 1
        df = existing
    elif not existing.empty:
        df = pd.concat([existing, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = new

    df.to_csv(out_path, index_label="timestamp")
    print(f"wrote {len(df):,} bars to {out_path}  "
          f"({df.index[0]} → {df.index[-1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
