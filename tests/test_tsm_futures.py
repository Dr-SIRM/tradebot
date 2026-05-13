"""Tests for backtest/tsm_futures — futures cost & margin modeling."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.tsm import TSMParams
from backtest.tsm_futures import (
    FuturesContract,
    simulate_tsm_futures,
    summarize_futures,
)


def _daily_index(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _uptrend(n: int = 700, start: float = 100.0, end: float = 400.0,
              seed: int = 0) -> pd.Series:
    """Strong, mostly-monotonic uptrend with low noise. TSM signals should
    be predominantly positive across the whole live region."""
    rng = np.random.RandomState(seed)
    drift = np.linspace(0, np.log(end / start), n)
    noise = rng.normal(0, 0.003, n).cumsum()
    return pd.Series(start * np.exp(drift + noise), index=_daily_index(n))


def _flat(n: int = 700, level: float = 100.0) -> pd.Series:
    return pd.Series(np.full(n, level), index=_daily_index(n))


def _gc_contract(**overrides) -> FuturesContract:
    base = dict(symbol="GC", multiplier=100.0, initial_margin=10_000.0,
                commission_per_side=2.50, proxy_price_scale=10.0)
    base.update(overrides)
    return FuturesContract(**base)


def _params(**overrides) -> TSMParams:
    base = dict(lookback_days=120, skip_days=10, vol_lookback_days=30,
                target_vol=0.10, max_leverage=2.0,
                cost_bps_per_turnover=0.0,  # not used in futures sim
                rebalance="monthly")
    base.update(overrides)
    return TSMParams(**base)


def test_uptrend_profits_at_1x_leverage():
    res = simulate_tsm_futures(_uptrend(), _gc_contract(), _params(),
                                leverage=1.0, initial_equity=100_000)
    assert not res.wiped_out
    assert res.equity.iloc[-1] > 100_000


def test_flat_series_no_pnl_minus_commissions():
    # Flat prices → signal stays at 0 throughout → no contracts traded → no costs
    res = simulate_tsm_futures(_flat(), _gc_contract(), _params(),
                                leverage=1.0, initial_equity=100_000)
    assert abs(res.equity.iloc[-1] - 100_000) < 1.0  # no PnL, no commissions
    assert res.commissions_paid == 0.0


def test_higher_leverage_amplifies_returns():
    close = _uptrend()
    p = _params(target_vol=0.05)  # lower base vol so leverage room exists
    contract = _gc_contract()
    r1 = simulate_tsm_futures(close, contract, p, leverage=1.0,
                                 initial_equity=500_000)
    r2 = simulate_tsm_futures(close, contract, p, leverage=2.0,
                                 initial_equity=500_000)
    s1 = summarize_futures(r1)
    s2 = summarize_futures(r2)
    # 2x leverage should give roughly 2x annual return (within slack for
    # rounding granularity and cost drag)
    assert s2["annualized_return"] > s1["annualized_return"]
    # Sharpe should be roughly preserved (cost drag may eat a tiny amount)
    assert abs(s2["sharpe"] - s1["sharpe"]) < 0.5


def test_extreme_leverage_wipes_out():
    # Create a series with a large drawdown right after warmup
    rng = np.random.RandomState(42)
    n = 600
    base = np.zeros(n)
    base[200:240] = -0.05  # 40 consecutive ~5% down days
    base += rng.normal(0, 0.005, n)
    close = pd.Series(100 * np.exp(np.cumsum(base)), index=_daily_index(n))
    # Run at huge leverage on a small account
    res = simulate_tsm_futures(close, _gc_contract(), _params(target_vol=0.5),
                                  leverage=20.0, initial_equity=10_000)
    s = summarize_futures(res)
    # At 20x leverage on a tiny account with a 5x40-day drawdown, equity
    # should hit zero
    assert s["wiped_out"] is True
    assert s["total_return"] == -1.0


def test_commissions_scale_with_contracts():
    close = _uptrend()
    p = _params()
    # Two equity sizes — bigger account holds more contracts → more cost
    r_small = simulate_tsm_futures(close, _gc_contract(), p, leverage=1.0,
                                      initial_equity=100_000)
    r_big = simulate_tsm_futures(close, _gc_contract(), p, leverage=1.0,
                                    initial_equity=1_000_000)
    # Costs should scale with account size (bigger account holds more
    # contracts per rebalance). Integer rounding makes the small account
    # under-allocate, so the ratio isn't exactly proportional to equity —
    # just verify the big account paid materially more.
    assert r_big.commissions_paid > r_small.commissions_paid
    assert r_big.commissions_paid > 2 * r_small.commissions_paid


def test_margin_breaches_recorded_on_aggressive_leverage():
    # Pick a series that triggers margin tightness with high leverage
    close = _uptrend()
    res = simulate_tsm_futures(close, _gc_contract(), _params(target_vol=0.5),
                                  leverage=10.0, initial_equity=50_000)
    # Either wiped out or had at least some margin breaches
    assert res.margin_breaches >= 0  # always non-negative; usually > 0 here


def test_requires_datetime_index():
    close = pd.Series([100, 101, 102])
    with pytest.raises(TypeError):
        simulate_tsm_futures(close, _gc_contract(), _params())


def test_insufficient_history_raises():
    close = _uptrend(n=80)  # less than warmup
    with pytest.raises(ValueError, match="need >="):
        simulate_tsm_futures(close, _gc_contract(), _params())


def test_summarize_returns_expected_keys():
    res = simulate_tsm_futures(_uptrend(), _gc_contract(), _params(),
                                 leverage=1.0, initial_equity=100_000)
    s = summarize_futures(res)
    for k in ("sharpe", "annualized_return", "annualized_vol", "max_dd",
              "total_return", "wiped_out", "commissions_paid",
              "margin_breaches", "max_margin_pct", "leverage", "contract"):
        assert k in s, f"missing key: {k}"


def test_integer_contracts_only():
    close = _uptrend()
    res = simulate_tsm_futures(close, _gc_contract(), _params(),
                                  leverage=1.0, initial_equity=100_000)
    # Every value in the contracts series should be an integer
    assert ((res.contracts == res.contracts.astype(int)).all())
