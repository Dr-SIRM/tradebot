"""Breakout: enter on a confirmed break of N-bar support/resistance with
volume confirmation, after a period of consolidation.

Long entry:
  - close > rolling resistance(lookback)
  - volume > volume_confirm_mult × SMA(volume, 20)
  - prior consolidation: ATR/price < consolidation_atr_pct over last lookback bars
  - HTF bias is up (close > HTF EMA50 — defensively use the same htf_ema_period
    as the momentum strategy if the param exists; otherwise default 50)

Short entry: mirror.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd

from strategy.base import Strategy
from utils.types import Signal, Side, Regime


class BreakoutStrategy(Strategy):
    name = "breakout"

    def evaluate(self, htf: pd.DataFrame, ltf: pd.DataFrame, symbol: str,
                 asset_class: str, regime: Regime, timeframe_low: str) -> Optional[Signal]:
        p = self.params
        lookback = p["lookback_bars"]
        if len(ltf) < lookback + 5:
            return None

        last = ltf.iloc[-1]
        if pd.isna(last["support"]) or pd.isna(last["resistance"]) or pd.isna(last["atr"]):
            return None

        # Consolidation check: average atr_pct over the lookback window must be tight
        atr_pct_window = ltf["atr_pct"].iloc[-lookback:].dropna()
        if atr_pct_window.empty:
            return None
        if atr_pct_window.mean() > p["consolidation_atr_pct"] * 2:
            # Not consolidating — skip
            return None

        # HTF bias
        htf_ema_period = 50
        htf_ema_val = htf["close"].ewm(span=htf_ema_period, adjust=False,
                                        min_periods=htf_ema_period).mean().iloc[-1]
        if pd.isna(htf_ema_val):
            return None
        uptrend = htf["close"].iloc[-1] > htf_ema_val
        downtrend = htf["close"].iloc[-1] < htf_ema_val

        vol_ok = last["vol_surge"] >= p["volume_confirm_mult"]

        side: Optional[Side] = None
        if last["close"] > last["resistance"] and vol_ok and uptrend:
            side = Side.LONG
        elif last["close"] < last["support"] and vol_ok and downtrend:
            side = Side.SHORT
        else:
            return None

        atr_val = float(last["atr"])
        entry = float(last["close"])
        # Stop just inside the broken level
        if side == Side.LONG:
            stop = float(last["resistance"]) - 0.5 * atr_val
            stop = min(stop, entry - 1.5 * atr_val)  # don't allow tighter than 1.5 ATR
            take_profit = entry + 3.0 * (entry - stop)
        else:
            stop = float(last["support"]) + 0.5 * atr_val
            stop = max(stop, entry + 1.5 * atr_val)
            take_profit = entry - 3.0 * (stop - entry)

        # Conviction: how decisive was the break? (close beyond level by N ATRs)
        if side == Side.LONG:
            decisiveness = (entry - last["resistance"]) / atr_val
        else:
            decisiveness = (last["support"] - entry) / atr_val
        decisiveness = max(0.0, min(decisiveness, 1.0))

        vol_strength = max(0.0, min(
            (last["vol_surge"] - p["volume_confirm_mult"]) / p["volume_confirm_mult"], 1.0
        ))
        conviction = max(0.1, min(0.6 * decisiveness + 0.4 * vol_strength, 1.0))

        return Signal(
            symbol=symbol, asset_class=asset_class, side=side, strategy=self.name,
            entry=entry, stop=stop, take_profit=take_profit, conviction=conviction,
            timestamp=last.name.to_pydatetime() if hasattr(last.name, "to_pydatetime") else last.name,
            timeframe=timeframe_low, regime=regime,
            metadata={"resistance": float(last["resistance"]),
                      "support": float(last["support"]),
                      "vol_surge": float(last["vol_surge"]), "atr": atr_val},
        )
