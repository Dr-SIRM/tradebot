"""Tests for run_tsm_sensitivity — grid expansion + stability analysis."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.tsm import TSMParams
from run_tsm_sensitivity import _grid_combos, analyze_sensitivity, run_sensitivity


def _trending_series(n: int = 1500) -> pd.Series:
    rng = np.random.RandomState(0)
    drift = np.linspace(0, 0.8, n)
    noise = rng.normal(0, 0.01, n).cumsum()
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(100 * np.exp(drift + noise), index=idx)


def test_grid_combos_cartesian():
    grid = {"a": [1, 2], "b": [10, 20, 30]}
    combos = _grid_combos(grid)
    assert len(combos) == 6
    # Every combo should have both keys
    for c in combos:
        assert set(c.keys()) == {"a", "b"}


def test_grid_combos_single_value():
    grid = {"a": [1]}
    assert _grid_combos(grid) == [{"a": 1}]


def test_run_sensitivity_produces_one_row_per_combo():
    close = _trending_series()
    grid = {"lookback_days": [120, 252],
             "skip_days": [10, 21],
             "vol_lookback_days": [60]}  # 2×2×1 = 4 combos
    res = run_sensitivity(close, TSMParams(cost_bps_per_turnover=0.0), grid)
    assert res["n_strategies"] == 4
    assert len(res["rows"]) == 4
    assert res["returns_matrix"].shape[1] == 4
    # Each row has Sharpe etc.
    for col in ("sharpe", "annualized_return", "max_dd"):
        assert col in res["rows"].columns


def test_analyze_sensitivity_returns_stability_fields():
    close = _trending_series()
    grid = {"lookback_days": [120, 252],
             "skip_days": [10, 21],
             "vol_lookback_days": [60]}
    res = run_sensitivity(close, TSMParams(cost_bps_per_turnover=0.0), grid)
    stability = analyze_sensitivity(res, pbo_blocks=4)
    expected = {"n_configs", "sharpe_p10", "sharpe_p50", "sharpe_p90",
                 "sharpe_mean", "sharpe_std", "frac_positive_sharpe",
                 "best_sharpe", "worst_sharpe"}
    assert expected.issubset(stability.keys())
    # Should also have DSR fields (>= 30 live points expected)
    assert "dsr_at_best" in stability
    # PBO is optional depending on data length / blocks
    if "pbo" in stability:
        assert 0.0 <= stability["pbo"] <= 1.0


def test_strong_uptrend_has_high_fraction_positive():
    # A clean uptrend should give positive Sharpe across most TSM configs.
    close = _trending_series(n=2000)
    grid = {"lookback_days": [120, 252], "skip_days": [10, 21],
             "vol_lookback_days": [60]}
    res = run_sensitivity(close, TSMParams(cost_bps_per_turnover=0.0), grid)
    stability = analyze_sensitivity(res, pbo_blocks=4)
    assert stability["frac_positive_sharpe"] >= 0.75
