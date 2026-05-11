"""Vectorized technical indicators. All functions accept a pandas Series/DataFrame
and return a Series/DataFrame aligned to the input index. Designed to be fast
enough for both live tick-by-tick and full backtests."""
from __future__ import annotations
import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/period
    avg_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_dn = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ADX. Returns a Series. NaN until min_periods met."""
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=df.index)
    tr = true_range(df)
    atr_val = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_val)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Returns DataFrame with mid/upper/lower/bandwidth columns."""
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    bw = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "bandwidth": bw})


def volume_surge(volume: pd.Series, period: int = 20) -> pd.Series:
    """volume / SMA(volume, period). > 1 means above average."""
    avg = sma(volume, period)
    return volume / avg.replace(0, np.nan)


def support_resistance(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Rolling support (low min) / resistance (high max). Excludes current bar
    so a breakout is detectable on the close of the current bar."""
    res = df["high"].shift(1).rolling(lookback, min_periods=lookback).max()
    sup = df["low"].shift(1).rolling(lookback, min_periods=lookback).min()
    return pd.DataFrame({"support": sup, "resistance": res})


def add_all_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute the full indicator set used by every strategy. Mutates a copy
    and returns it. cfg comes from the loaded YAML."""
    out = df.copy()
    out["ema_fast"] = ema(out["close"], cfg["strategy"]["momentum"]["ema_fast"])
    out["ema_slow"] = ema(out["close"], cfg["strategy"]["momentum"]["ema_slow"])
    out["rsi"] = rsi(out["close"], cfg["strategy"]["momentum"]["rsi_period"])
    out["atr"] = atr(out, cfg["strategy"]["regime"]["atr_period"])
    out["adx"] = adx(out, cfg["strategy"]["regime"]["adx_period"])

    bb = bollinger(out["close"],
                   cfg["strategy"]["mean_reversion"]["bb_period"],
                   cfg["strategy"]["mean_reversion"]["bb_std"])
    out["bb_mid"] = bb["mid"]
    out["bb_upper"] = bb["upper"]
    out["bb_lower"] = bb["lower"]
    out["bb_width"] = bb["bandwidth"]

    sr = support_resistance(out, cfg["strategy"]["breakout"]["lookback_bars"])
    out["support"] = sr["support"]
    out["resistance"] = sr["resistance"]

    out["vol_surge"] = volume_surge(out["volume"], 20)

    # ATR as % of price — used by regime classifier and breakout consolidation check
    out["atr_pct"] = (out["atr"] / out["close"]) * 100.0

    return out
