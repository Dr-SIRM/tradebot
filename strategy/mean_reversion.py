"""Mean Reversion: fade Bollinger Band touches when RSI is at extremes.

Long entry:
  - close pierces lower BB (close <= bb_lower)
  - RSI < rsi_oversold
  - optionally require recent BB squeeze (bb_width < bb_squeeze_pct)

Short entry: mirror.

This strategy is *contra-trend* — it deliberately does NOT use the HTF filter,
because mean-reversion works best in ranging markets where the HTF lacks a
strong trend. The regime selector ensures it only runs in ranging regimes.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd

from strategy.base import Strategy
from utils.types import Signal, Side, Regime


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def evaluate(self, htf: pd.DataFrame, ltf: pd.DataFrame, symbol: str,
                 asset_class: str, regime: Regime, timeframe_low: str) -> Optional[Signal]:
        p = self.params
        if len(ltf) < max(p["bb_period"], p["rsi_period"]) + 5:
            return None

        last = ltf.iloc[-1]
        if pd.isna(last["bb_lower"]) or pd.isna(last["bb_upper"]) or pd.isna(last["atr"]):
            return None

        # Optional squeeze filter — if required, last 5 bars must contain a squeeze
        if p.get("require_squeeze", False):
            recent_bw = ltf["bb_width"].iloc[-5:].dropna()
            if recent_bw.empty or recent_bw.min() > p["bb_squeeze_pct"]:
                return None

        side: Optional[Side] = None
        if last["close"] <= last["bb_lower"] and last["rsi"] < p["rsi_oversold"]:
            side = Side.LONG
        elif last["close"] >= last["bb_upper"] and last["rsi"] > p["rsi_overbought"]:
            side = Side.SHORT
        else:
            return None

        atr_val = float(last["atr"])
        entry = float(last["close"])
        # Tighter stop than momentum (mean-rev expects quick reversals)
        stop_mult = 1.5
        if side == Side.LONG:
            stop = entry - stop_mult * atr_val
            take_profit = float(last["bb_mid"])  # target the mean
            # Sanity: TP must improve on entry
            if take_profit <= entry:
                take_profit = entry + 2.5 * (entry - stop)
        else:
            stop = entry + stop_mult * atr_val
            take_profit = float(last["bb_mid"])
            if take_profit >= entry:
                take_profit = entry - 2.5 * (stop - entry)

        # Conviction: how extreme is RSI? how far outside the band?
        if side == Side.LONG:
            rsi_extremity = (p["rsi_oversold"] - last["rsi"]) / p["rsi_oversold"]
            band_extension = (last["bb_lower"] - last["close"]) / atr_val
        else:
            rsi_extremity = (last["rsi"] - p["rsi_overbought"]) / (100 - p["rsi_overbought"])
            band_extension = (last["close"] - last["bb_upper"]) / atr_val

        rsi_extremity = max(0.0, min(rsi_extremity, 1.0))
        band_extension = max(0.0, min(band_extension, 1.0))
        conviction = max(0.1, min(0.5 * rsi_extremity + 0.5 * band_extension, 1.0))

        return Signal(
            symbol=symbol, asset_class=asset_class, side=side, strategy=self.name,
            entry=entry, stop=stop, take_profit=take_profit, conviction=conviction,
            timestamp=last.name.to_pydatetime() if hasattr(last.name, "to_pydatetime") else last.name,
            timeframe=timeframe_low, regime=regime,
            metadata={"rsi": float(last["rsi"]), "bb_width": float(last["bb_width"]),
                      "atr": atr_val},
        )
