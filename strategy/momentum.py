"""Momentum: EMA cross + RSI confirmation + volume surge, gated by HTF trend.

Long entry:
  - LTF: ema_fast just crossed above ema_slow (current bar)
  - LTF: RSI > rsi_min_long
  - LTF: volume > volume_surge_mult × SMA(volume, 20)
  - HTF: close > ema(htf_ema_period)  (uptrend bias)

Short entry: mirror image.

Conviction = blend of (RSI deviation from 50) and (volume surge magnitude),
both clamped to [0, 1].
"""
from __future__ import annotations
from typing import Optional
import pandas as pd

from strategy.base import Strategy
from utils.types import Signal, Side, Regime
from utils.logging import get_logger

log = get_logger(__name__)


class MomentumStrategy(Strategy):
    name = "momentum"

    def evaluate(self, htf: pd.DataFrame, ltf: pd.DataFrame, symbol: str,
                 asset_class: str, regime: Regime, timeframe_low: str) -> Optional[Signal]:
        p = self.params
        if len(ltf) < max(p["ema_slow"], p["rsi_period"], 20) + 2:
            return None
        if len(htf) < p["htf_ema_period"] + 1:
            return None

        last = ltf.iloc[-1]
        prev = ltf.iloc[-2]

        # HTF trend filter (compute fresh — htf doesn't carry this column by default)
        htf_ema = htf["close"].ewm(span=p["htf_ema_period"], adjust=False,
                                    min_periods=p["htf_ema_period"]).mean().iloc[-1]
        if pd.isna(htf_ema):
            return None
        uptrend = htf["close"].iloc[-1] > htf_ema
        downtrend = htf["close"].iloc[-1] < htf_ema

        cross_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
        cross_dn = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

        rsi_ok_long = last["rsi"] > p["rsi_min_long"]
        rsi_ok_short = last["rsi"] < p["rsi_max_short"]
        vol_ok = last["vol_surge"] >= p["volume_surge_mult"]

        side: Optional[Side] = None
        if cross_up and rsi_ok_long and vol_ok and uptrend:
            side = Side.LONG
        elif cross_dn and rsi_ok_short and vol_ok and downtrend:
            side = Side.SHORT
        else:
            return None

        atr_val = last["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            return None

        entry = float(last["close"])
        # Stop is set later by risk/position_sizer (uses ATR mult), but strategies
        # propose an initial stop & TP for the risk manager's reward:risk check.
        stop_mult = 1.75  # midpoint of risk.atr_stop_mult range
        if side == Side.LONG:
            stop = entry - stop_mult * atr_val
            take_profit = entry + 3.0 * (entry - stop)  # ~3R target
        else:
            stop = entry + stop_mult * atr_val
            take_profit = entry - 3.0 * (stop - entry)

        # Conviction: RSI distance from 50 (max 30) + vol surge above threshold
        rsi_strength = min(abs(last["rsi"] - 50) / 30.0, 1.0)
        vol_strength = min((last["vol_surge"] - p["volume_surge_mult"]) / p["volume_surge_mult"], 1.0)
        vol_strength = max(vol_strength, 0.0)
        conviction = 0.5 * rsi_strength + 0.5 * vol_strength
        conviction = max(0.1, min(conviction, 1.0))

        return Signal(
            symbol=symbol, asset_class=asset_class, side=side, strategy=self.name,
            entry=entry, stop=stop, take_profit=take_profit, conviction=conviction,
            timestamp=last.name.to_pydatetime() if hasattr(last.name, "to_pydatetime") else last.name,
            timeframe=timeframe_low, regime=regime,
            metadata={"rsi": float(last["rsi"]), "vol_surge": float(last["vol_surge"]),
                      "atr": float(atr_val)},
        )
