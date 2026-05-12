"""Tests for the market regime classifier."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.indicators import add_all_indicators
from strategy.regime import RegimeDetector
from utils.types import Regime


def _full_cfg():
    return {
        "strategy": {
            "regime": {
                "adx_trending": 25,
                "adx_ranging": 20,
                "atr_high_vol_pct": 4.0,  # atr_pct is in % units
                "atr_period": 14,
                "adx_period": 14,
            },
            "momentum": {
                "ema_fast": 9, "ema_slow": 21,
                "rsi_period": 14, "rsi_buy": 55, "rsi_sell": 45,
                "volume_surge_threshold": 1.3,
            },
            "mean_reversion": {
                "bb_period": 20, "bb_std": 2.0, "rsi_period": 14,
                "rsi_overbought": 70, "rsi_oversold": 30,
                "squeeze_bandwidth_pct": 0.05,
            },
            "breakout": {
                "lookback_bars": 20, "volume_surge_threshold": 1.5,
                "consolidation_atr_pct_max": 3.0,
            },
        }
    }


def _trending_bars(n=200):
    t = np.linspace(100, 150, n)
    return pd.DataFrame({
        "open": t, "high": t + 0.5, "low": t - 0.5, "close": t,
        "volume": np.full(n, 1000.0),
    }, index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"))


def _ranging_bars(n=200, seed=1):
    rng = np.random.default_rng(seed)
    # Tight range around 100
    close = 100 + np.sin(np.linspace(0, 8 * np.pi, n)) * 0.5 + rng.standard_normal(n) * 0.05
    return pd.DataFrame({
        "open": close, "high": close + 0.1, "low": close - 0.1, "close": close,
        "volume": np.full(n, 1000.0),
    }, index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"))


def _volatile_bars(n=200, seed=2):
    rng = np.random.default_rng(seed)
    # Big spikes
    close = 100 + np.cumsum(rng.standard_normal(n) * 5)
    high = close + np.abs(rng.standard_normal(n)) * 5
    low = close - np.abs(rng.standard_normal(n)) * 5
    return pd.DataFrame({
        "open": close, "high": high, "low": low, "close": close,
        "volume": np.full(n, 1000.0),
    }, index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"))


class TestRegime:
    def test_trending_detected(self):
        cfg = _full_cfg()
        df = add_all_indicators(_trending_bars(), cfg)
        det = RegimeDetector(cfg["strategy"]["regime"])
        assert det.classify(df) == Regime.TRENDING

    def test_ranging_detected(self):
        cfg = _full_cfg()
        df = add_all_indicators(_ranging_bars(), cfg)
        det = RegimeDetector(cfg["strategy"]["regime"])
        # Expect either RANGING or UNKNOWN — both are non-trending and the
        # classifier may sit in the gap; key is that it's NOT TRENDING.
        regime = det.classify(df)
        assert regime in (Regime.RANGING, Regime.UNKNOWN)

    def test_volatile_detected(self):
        cfg = _full_cfg()
        df = add_all_indicators(_volatile_bars(), cfg)
        det = RegimeDetector(cfg["strategy"]["regime"])
        # With wide ATR, classification should not return TRENDING
        regime = det.classify(df)
        assert regime in (Regime.VOLATILE, Regime.RANGING, Regime.UNKNOWN)

    def test_asset_class_override_applied(self):
        # Same bars, but different per-class thresholds change the verdict.
        # Volatile bars typically score ATR~3-5%. With base threshold 2.0
        # they're VOLATILE; the 'crypto' override of 100% disables the
        # volatile classification entirely so we fall through to ADX rules.
        cfg = _full_cfg()
        cfg["strategy"]["regime"]["atr_high_vol_pct"] = 2.0
        cfg["strategy"]["regime"]["asset_class_overrides"] = {
            "crypto": {"atr_high_vol_pct": 100.0},
        }
        df = add_all_indicators(_volatile_bars(), cfg)
        det = RegimeDetector(cfg["strategy"]["regime"])
        assert det.classify(df, asset_class="equity") == Regime.VOLATILE
        assert det.classify(df, asset_class="crypto") != Regime.VOLATILE

    def test_asset_class_override_falls_back_to_default(self):
        cfg = _full_cfg()
        cfg["strategy"]["regime"]["asset_class_overrides"] = {
            "crypto": {"atr_high_vol_pct": 100.0},
        }
        df = add_all_indicators(_volatile_bars(), cfg)
        det = RegimeDetector(cfg["strategy"]["regime"])
        # No override for 'forex' → uses base 4.0 threshold from _full_cfg
        regime_forex = det.classify(df, asset_class="forex")
        regime_none = det.classify(df, asset_class=None)
        assert regime_forex == regime_none
