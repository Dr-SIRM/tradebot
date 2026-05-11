"""Strategy selector — picks the right strategy for the current regime, runs
it, and returns a signal (or None).

Mapping (default):
  - TRENDING  → momentum, breakout (run both, take higher conviction)
  - RANGING   → mean_reversion
  - VOLATILE  → no trade (sit out)
  - UNKNOWN   → no trade
"""
from __future__ import annotations
from typing import Optional
import pandas as pd

from strategy.base import Strategy
from strategy.regime import RegimeDetector
from strategy.momentum import MomentumStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.breakout import BreakoutStrategy
from utils.types import Signal, Regime
from utils.logging import get_logger

log = get_logger(__name__)


_STRATEGY_CLASSES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
}


class StrategySelector:
    def __init__(self, cfg: dict):
        enabled = set(cfg["strategy"]["enabled"])
        self.strategies: dict[str, Strategy] = {}
        for name, cls in _STRATEGY_CLASSES.items():
            if name in enabled:
                self.strategies[name] = cls(cfg["strategy"][name])
        self.regime = RegimeDetector(cfg["strategy"]["regime"])

        # Regime → list of strategy names eligible to run
        self.regime_map = {
            Regime.TRENDING: [n for n in ["momentum", "breakout"] if n in self.strategies],
            Regime.RANGING: [n for n in ["mean_reversion"] if n in self.strategies],
            Regime.VOLATILE: [],
            Regime.UNKNOWN: [],
        }

    def select(self, htf: pd.DataFrame, ltf: pd.DataFrame, symbol: str,
               asset_class: str, timeframe_low: str) -> tuple[Regime, Optional[Signal]]:
        regime = self.regime.classify(ltf)
        eligible = self.regime_map.get(regime, [])
        if not eligible:
            return regime, None

        best: Optional[Signal] = None
        for name in eligible:
            strat = self.strategies[name]
            sig = strat.evaluate(htf, ltf, symbol, asset_class, regime, timeframe_low)
            if sig is None:
                continue
            if best is None or sig.conviction > best.conviction:
                best = sig
        return regime, best
