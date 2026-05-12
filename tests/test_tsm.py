"""Tests for time-series momentum simulation (backtest/tsm.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.tsm import (
    TSMParams,
    _rebalance_mask,
    simulate_tsm,
    summarize,
    tsm_signal,
    vol_scaled_weight,
)


def _daily_index(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def test_tsm_signal_long_when_uptrend():
    # Steady uptrend: cumulative return is positive → signal +1
    idx = _daily_index(400)
    close = pd.Series(np.linspace(100, 200, 400), index=idx)
    sig = tsm_signal(close, lookback_days=252, skip_days=21)
    # First valid signal is at index 252+21 = 273
    assert sig.iloc[273] == 1.0
    assert sig.iloc[-1] == 1.0


def test_tsm_signal_short_when_downtrend():
    idx = _daily_index(400)
    close = pd.Series(np.linspace(200, 100, 400), index=idx)
    sig = tsm_signal(close, lookback_days=252, skip_days=21)
    assert sig.iloc[273] == -1.0
    assert sig.iloc[-1] == -1.0


def test_tsm_signal_zero_during_warmup():
    idx = _daily_index(100)
    close = pd.Series(np.linspace(100, 200, 100), index=idx)
    sig = tsm_signal(close, lookback_days=252, skip_days=21)
    # 100 bars is less than 252+21=273, so signal is 0 everywhere
    assert (sig == 0.0).all()


def test_vol_scaled_weight_scales_inverse_to_volatility():
    idx = _daily_index(200)
    # Construct two GBM series with materially different daily vols and
    # very high leverage cap so the clip doesn't mask the comparison.
    rng = np.random.RandomState(0)
    low_vol = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 200)))
    high_vol = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.05, 200)))
    w_low = vol_scaled_weight(pd.Series(low_vol, index=idx), 60, 0.10,
                                max_leverage=1000.0).iloc[-1]
    w_high = vol_scaled_weight(pd.Series(high_vol, index=idx), 60, 0.10,
                                max_leverage=1000.0).iloc[-1]
    assert w_low > w_high


def test_vol_scaled_weight_respects_max_leverage():
    idx = _daily_index(100)
    # Near-constant series → tiny realized vol → weight would blow up without cap
    close = pd.Series(100 + np.zeros(100) + np.linspace(0, 0.01, 100), index=idx)
    w = vol_scaled_weight(close, 60, 0.10, max_leverage=2.0)
    assert w.max() <= 2.0 + 1e-9


def test_rebalance_mask_monthly_marks_first_trading_day_of_month():
    idx = _daily_index(60, start="2020-01-01")
    mask = _rebalance_mask(idx, "monthly")
    # First bar of Jan and first bar of Feb should be True
    first_jan = mask.index[0]
    feb_starts = [t for t in mask.index if t.month == 2]
    assert mask.loc[first_jan]
    assert mask.loc[feb_starts[0]]
    # A mid-Jan day should be False
    mid_jan = [t for t in mask.index if t.month == 1 and t.day > 10][0]
    assert not mask.loc[mid_jan]


def test_simulate_tsm_uptrend_makes_money():
    # Smooth uptrend, no noise → positive return after warmup
    idx = _daily_index(600)
    close = pd.Series(100 * np.exp(np.linspace(0, 0.5, 600)), index=idx)
    p = TSMParams(lookback_days=120, skip_days=10, vol_lookback_days=30,
                  target_vol=0.10, max_leverage=2.0, cost_bps_per_turnover=0.0,
                  rebalance="monthly")
    sim = simulate_tsm(close, p)
    assert sim["equity"].iloc[-1] > p.initial_equity
    assert (sim["position"] > 0).any()


def test_simulate_tsm_downtrend_profits_via_short():
    idx = _daily_index(600)
    close = pd.Series(200 * np.exp(np.linspace(0, -0.5, 600)), index=idx)
    p = TSMParams(lookback_days=120, skip_days=10, vol_lookback_days=30,
                  cost_bps_per_turnover=0.0)
    sim = simulate_tsm(close, p)
    assert sim["equity"].iloc[-1] > p.initial_equity
    assert (sim["position"] < 0).any()


def test_simulate_tsm_costs_reduce_returns():
    idx = _daily_index(600)
    # Zigzag trend forces many position flips → high turnover
    base = np.concatenate([np.linspace(0, 0.3, 200), np.linspace(0.3, -0.2, 200),
                            np.linspace(-0.2, 0.4, 200)])
    close = pd.Series(100 * np.exp(base), index=idx)
    p_no_cost = TSMParams(lookback_days=60, skip_days=5, vol_lookback_days=20,
                           cost_bps_per_turnover=0.0)
    p_cost = TSMParams(lookback_days=60, skip_days=5, vol_lookback_days=20,
                        cost_bps_per_turnover=20.0)
    eq_no = simulate_tsm(close, p_no_cost)["equity"].iloc[-1]
    eq_cost = simulate_tsm(close, p_cost)["equity"].iloc[-1]
    assert eq_cost < eq_no


def test_simulate_tsm_flat_series_no_pnl():
    idx = _daily_index(400)
    close = pd.Series(100.0, index=idx)
    p = TSMParams(lookback_days=120, skip_days=10, vol_lookback_days=30)
    sim = simulate_tsm(close, p)
    # Flat → no signal → no P&L
    assert abs(sim["equity"].iloc[-1] - p.initial_equity) < 1e-6


def test_simulate_tsm_warmup_blocks_early_trades():
    idx = _daily_index(400)
    close = pd.Series(np.linspace(100, 200, 400), index=idx)
    p = TSMParams(lookback_days=120, skip_days=10, vol_lookback_days=30)
    sim = simulate_tsm(close, p)
    warmup = p.lookback_days + p.skip_days + p.vol_lookback_days
    # During warmup, position should be zero
    assert (sim["position"].iloc[:warmup] == 0).all()


def test_summarize_shape():
    idx = _daily_index(400)
    close = pd.Series(np.linspace(100, 200, 400), index=idx)
    p = TSMParams(lookback_days=120, skip_days=10, vol_lookback_days=30,
                  cost_bps_per_turnover=0.0)
    s = summarize(simulate_tsm(close, p))
    for k in ("sharpe", "annualized_return", "annualized_vol", "total_return",
              "max_dd", "n_trades"):
        assert k in s
    assert s["sharpe"] >= 0  # uptrend should yield non-negative Sharpe


def test_simulate_tsm_requires_datetime_index():
    close = pd.Series([100, 101, 102])
    with pytest.raises(TypeError):
        simulate_tsm(close, TSMParams())
