"""Market regime classifier — ADX + ATR-based.

Outputs one of {TRENDING, RANGING, VOLATILE, UNKNOWN}.

Rules:
  - VOLATILE if atr_pct > atr_high_vol_pct (overrides everything else)
  - TRENDING if adx > adx_trending
  - RANGING if adx < adx_ranging
  - else UNKNOWN (transitional zone — don't take new trades)

Asset classes have very different baseline volatility — daily BTC routinely
has ATR/price ~4% which is *normal*, not "volatile". Per-class overrides in
config.strategy.regime.asset_class_overrides resolve that at classify time.
"""
from __future__ import annotations
from typing import Optional

import pandas as pd

from utils.types import Regime


class RegimeDetector:
    def __init__(self, params: dict):
        self.p = params
        self.overrides: dict[str, dict] = params.get("asset_class_overrides", {}) or {}

    def _params_for(self, asset_class: Optional[str]) -> dict:
        if asset_class and asset_class in self.overrides:
            merged = dict(self.p)
            merged.update(self.overrides[asset_class])
            return merged
        return self.p

    def classify(self, df: pd.DataFrame, asset_class: Optional[str] = None) -> Regime:
        if df.empty:
            return Regime.UNKNOWN
        last = df.iloc[-1]
        adx_val = last.get("adx")
        atr_pct = last.get("atr_pct")
        if pd.isna(adx_val) or pd.isna(atr_pct):
            return Regime.UNKNOWN

        p = self._params_for(asset_class)
        if atr_pct > p["atr_high_vol_pct"]:
            return Regime.VOLATILE
        if adx_val > p["adx_trending"]:
            return Regime.TRENDING
        if adx_val < p["adx_ranging"]:
            return Regime.RANGING
        return Regime.UNKNOWN
