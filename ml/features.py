"""Feature engineering for the XGBoost trade-setup scorer.

Each feature row corresponds to a *signal* — a candidate trade — and the model
predicts P(profitable trade). Features are computed from the LTF DataFrame at
the bar of the signal.

Features (32):
  - Returns: last 1, 3, 5, 10, 20-bar log returns
  - Volatility: ATR/price (current + 5-bar avg + 20-bar avg)
  - Momentum: RSI, RSI delta, EMA fast/slow ratio, distance to ema_slow
  - Volume: vol_surge (current + 5-bar avg), vol z-score
  - Range/structure: distance to BB upper/lower (in ATRs), BB width,
    distance to 20-bar high/low (in ATRs)
  - Regime: ADX, ATR%, ADX delta
  - Setup: side (1/-1), strategy one-hot (3), conviction
  - HTF: HTF return last 5 bars, HTF EMA50 slope sign
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

from utils.types import Signal, Side


FEATURE_NAMES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "atr_pct", "atr_pct_5avg", "atr_pct_20avg",
    "rsi", "rsi_delta", "ema_ratio", "dist_ema_slow_atr",
    "vol_surge", "vol_surge_5avg", "vol_z",
    "dist_bb_upper_atr", "dist_bb_lower_atr", "bb_width",
    "dist_20h_atr", "dist_20l_atr",
    "adx", "adx_delta",
    "side", "is_momentum", "is_meanrev", "is_breakout", "conviction",
    "htf_ret_5", "htf_slope_sign",
    "hour_sin", "hour_cos", "dow",
]


def build_features(signal: Signal, ltf: pd.DataFrame, htf: pd.DataFrame,
                   lookback: int = 50) -> Optional[np.ndarray]:
    """Build a feature vector for one signal. Returns None if data insufficient."""
    if len(ltf) < lookback or len(htf) < 60:
        return None
    if "ema_fast" not in ltf.columns or "atr" not in ltf.columns:
        return None

    last = ltf.iloc[-1]
    close = ltf["close"]
    atr_val = float(last["atr"])
    if atr_val <= 0 or pd.isna(atr_val):
        return None

    # Log returns
    log_ret = np.log(close / close.shift(1))
    ret_1 = log_ret.iloc[-1] if len(log_ret) >= 1 else 0.0
    ret_3 = log_ret.iloc[-3:].sum() if len(log_ret) >= 3 else 0.0
    ret_5 = log_ret.iloc[-5:].sum() if len(log_ret) >= 5 else 0.0
    ret_10 = log_ret.iloc[-10:].sum() if len(log_ret) >= 10 else 0.0
    ret_20 = log_ret.iloc[-20:].sum() if len(log_ret) >= 20 else 0.0

    atr_pct_5avg = ltf["atr_pct"].iloc[-5:].mean()
    atr_pct_20avg = ltf["atr_pct"].iloc[-20:].mean()

    rsi_delta = ltf["rsi"].iloc[-1] - ltf["rsi"].iloc[-2] if len(ltf) >= 2 else 0.0
    ema_ratio = last["ema_fast"] / last["ema_slow"] if last["ema_slow"] else 1.0
    dist_ema_slow = (last["close"] - last["ema_slow"]) / atr_val

    vol_surge_5avg = ltf["vol_surge"].iloc[-5:].mean()
    vol_mean = ltf["volume"].iloc[-20:].mean()
    vol_std = ltf["volume"].iloc[-20:].std(ddof=0)
    vol_z = (last["volume"] - vol_mean) / vol_std if vol_std and not pd.isna(vol_std) else 0.0

    dist_bb_upper = (last["bb_upper"] - last["close"]) / atr_val
    dist_bb_lower = (last["close"] - last["bb_lower"]) / atr_val
    bb_width = last["bb_width"] if not pd.isna(last["bb_width"]) else 0.0

    high_20 = ltf["high"].iloc[-20:].max()
    low_20 = ltf["low"].iloc[-20:].min()
    dist_20h = (high_20 - last["close"]) / atr_val
    dist_20l = (last["close"] - low_20) / atr_val

    adx_val = float(last["adx"]) if not pd.isna(last["adx"]) else 20.0
    adx_delta = (last["adx"] - ltf["adx"].iloc[-2]) if len(ltf) >= 2 and not pd.isna(ltf["adx"].iloc[-2]) else 0.0

    side_num = 1.0 if signal.side == Side.LONG else -1.0
    is_mom = 1.0 if signal.strategy == "momentum" else 0.0
    is_mr = 1.0 if signal.strategy == "mean_reversion" else 0.0
    is_bo = 1.0 if signal.strategy == "breakout" else 0.0

    htf_close = htf["close"]
    htf_ret_5 = np.log(htf_close.iloc[-1] / htf_close.iloc[-6]) if len(htf_close) >= 6 else 0.0
    htf_ema = htf_close.ewm(span=50, adjust=False, min_periods=50).mean()
    htf_slope = htf_ema.iloc[-1] - htf_ema.iloc[-6] if len(htf_ema) >= 6 and not pd.isna(htf_ema.iloc[-6]) else 0.0
    htf_slope_sign = float(np.sign(htf_slope))

    ts = last.name if hasattr(last, "name") else None
    if isinstance(ts, pd.Timestamp):
        hour = ts.hour
        dow = ts.dayofweek
    else:
        hour, dow = 0, 0
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    feats = [
        ret_1, ret_3, ret_5, ret_10, ret_20,
        last["atr_pct"], atr_pct_5avg, atr_pct_20avg,
        last["rsi"], rsi_delta, ema_ratio, dist_ema_slow,
        last["vol_surge"], vol_surge_5avg, vol_z,
        dist_bb_upper, dist_bb_lower, bb_width,
        dist_20h, dist_20l,
        adx_val, adx_delta,
        side_num, is_mom, is_mr, is_bo, signal.conviction,
        htf_ret_5, htf_slope_sign,
        hour_sin, hour_cos, dow,
    ]
    arr = np.asarray(feats, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None
    return arr
