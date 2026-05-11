"""Abstract Strategy interface.

A strategy receives a higher-timeframe DataFrame (for trend bias) and a lower-
timeframe DataFrame (for entries), both pre-loaded with indicators. It returns
an Optional[Signal] for the latest bar — None means "no setup right now".

Strategies are stateless. State (open positions, P&L, etc.) lives in the risk
manager. This makes strategies trivially backtestable.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

from utils.types import Signal, Regime


class Strategy(ABC):
    """All strategies subclass this."""

    name: str = "base"

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def evaluate(self, htf: pd.DataFrame, ltf: pd.DataFrame, symbol: str,
                 asset_class: str, regime: Regime, timeframe_low: str) -> Optional[Signal]:
        """Run on the latest bar. Return a Signal if entry conditions met,
        else None. Inputs are guaranteed to have at least `warmup_bars` rows."""

    @staticmethod
    def _last_complete(df: pd.DataFrame) -> pd.Series:
        """Last bar (which by feed contract is closed)."""
        return df.iloc[-1]

    @staticmethod
    def _prev(df: pd.DataFrame, n: int = 1) -> pd.Series:
        return df.iloc[-1 - n]
