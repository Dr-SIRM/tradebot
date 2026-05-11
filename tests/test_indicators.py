"""Tests for technical indicators.

Sanity checks rather than reference-implementation tests; the goal is to
catch regressions in EMA/RSI/ATR/ADX behavior and edge cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.indicators import (
    ema, sma, rsi, true_range, atr, adx, bollinger,
    volume_surge, support_resistance,
)


def _make_bars(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + np.abs(rng.standard_normal(n) * 0.3)
    low = close - np.abs(rng.standard_normal(n) * 0.3)
    open_ = close + rng.standard_normal(n) * 0.1
    volume = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume})


class TestSMA:
    def test_sma_simple(self):
        s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        out = sma(s, 3)
        assert np.isnan(out.iloc[0])
        assert np.isnan(out.iloc[1])
        assert out.iloc[2] == pytest.approx(2.0)
        assert out.iloc[3] == pytest.approx(3.0)
        assert out.iloc[4] == pytest.approx(4.0)


class TestEMA:
    def test_ema_constant_series(self):
        s = pd.Series([5.0] * 50)
        out = ema(s, 10)
        assert out.iloc[-1] == pytest.approx(5.0)

    def test_ema_responsiveness(self):
        s = pd.Series([1.0] * 20 + [10.0] * 20)
        out = ema(s, 5)
        assert out.iloc[19] == pytest.approx(1.0, abs=0.01)
        assert out.iloc[39] > 8.0


class TestRSI:
    def test_rsi_bounded(self):
        df = _make_bars(100)
        out = rsi(df["close"], 14)
        assert out.between(0, 100).all()

    def test_rsi_uptrend_elevated(self):
        rng = np.random.default_rng(0)
        # Drift up but with enough variance to include some down moves
        returns = rng.normal(0.3, 1.0, 200)
        prices = 100 + np.cumsum(returns)
        out = rsi(pd.Series(prices), 14).dropna()
        # Average RSI over the series should be elevated (>50)
        assert out.mean() > 50

    def test_rsi_downtrend_depressed(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(-0.3, 1.0, 200)
        prices = 100 + np.cumsum(returns)
        out = rsi(pd.Series(prices), 14).dropna()
        assert out.mean() < 50


class TestTrueRangeATR:
    def test_true_range_basic(self):
        df = pd.DataFrame({
            "high": [10, 12, 11],
            "low":  [9, 10, 9.5],
            "close": [9.5, 11, 10],
        })
        tr = true_range(df)
        # Bar 0: prev_close NaN, max([1, NaN, NaN]) = 1.0
        assert tr.iloc[0] == pytest.approx(1.0)
        # Bar 1: max(12-10=2, |12-9.5|=2.5, |10-9.5|=0.5) = 2.5
        assert tr.iloc[1] == pytest.approx(2.5)
        # Bar 2: max(11-9.5=1.5, |11-11|=0, |9.5-11|=1.5) = 1.5
        assert tr.iloc[2] == pytest.approx(1.5)

    def test_atr_positive(self):
        df = _make_bars(100)
        out = atr(df, 14).dropna()
        assert (out > 0).all()


class TestADX:
    def test_adx_bounded(self):
        df = _make_bars(200)
        out = adx(df, 14).dropna()
        assert out.between(0, 100).all()

    def test_adx_higher_in_trend(self):
        n = 200
        trend = pd.Series(np.linspace(100, 150, n))
        df_trend = pd.DataFrame({
            "high": trend + 0.5, "low": trend - 0.5, "close": trend,
        })
        df_rand = _make_bars(n)
        adx_trend = adx(df_trend, 14).dropna().mean()
        adx_rand = adx(df_rand, 14).dropna().mean()
        assert adx_trend > adx_rand


class TestBollinger:
    def test_bollinger_ordering(self):
        df = _make_bars(100)
        bb = bollinger(df["close"], 20, 2.0).dropna()
        assert (bb["upper"] >= bb["mid"]).all()
        assert (bb["mid"] >= bb["lower"]).all()

    def test_bollinger_width_positive(self):
        df = _make_bars(100)
        bb = bollinger(df["close"], 20, 2.0).dropna()
        assert (bb["bandwidth"] >= 0).all()


class TestVolumeSurge:
    def test_surge_ratio(self):
        v = pd.Series([100.0] * 30 + [500.0])
        out = volume_surge(v, 20)
        # 20-bar mean includes the surge bar: (19*100 + 500) / 20 = 120
        # So ratio = 500 / 120 = 4.17
        assert out.iloc[-1] == pytest.approx(4.17, abs=0.1)


class TestSupportResistance:
    def test_excludes_current_bar(self):
        h = pd.Series([10.0, 11.0, 12.0, 13.0, 100.0])
        df = pd.DataFrame({"high": h, "low": h, "close": h})
        sr = support_resistance(df, lookback=4)
        assert sr["resistance"].iloc[-1] == pytest.approx(13.0)
        assert sr["support"].iloc[-1] == pytest.approx(10.0)
