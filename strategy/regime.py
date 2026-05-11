"""Market regime classifier — ADX + ATR-based.

Outputs one of {TRENDING, RANGING, VOLATILE, UNKNOWN}.

Rules:
  - VOLATILE if atr_pct > atr_high_vol_pct (overrides everything else)
  - TRENDING if adx > adx_trending
  - RANGING if adx < adx_ranging
  - else UNKNOWN (transitional zone — don't take new trades)
"""
from __future__ import annotations
import pandas as pd

from utils.types import Regime


class RegimeDetector:
    def __init__(self, params: dict):
        self.p = params

    def classify(self, df: pd.DataFrame) -> Regime:
        if df.empty:
            return Regime.UNKNOWN
        last = df.iloc[-1]
        adx_val = last.get("adx")
        atr_pct = last.get("atr_pct")
        if pd.isna(adx_val) or pd.isna(atr_pct):
            return Regime.UNKNOWN

        if atr_pct > self.p["atr_high_vol_pct"]:
            return Regime.VOLATILE
        if adx_val > self.p["adx_trending"]:
            return Regime.TRENDING
        if adx_val < self.p["adx_ranging"]:
            return Regime.RANGING
        return Regime.UNKNOWN
