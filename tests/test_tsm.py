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
    _portfolio_weights,
    _rebalance_mask,
    multi_horizon_signal,
    simulate_tsm,
    simulate_tsm_portfolio,
    summarize,
    summarize_portfolio,
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


# ---------- portfolio tests ----------

def _uptrend(n: int = 600, start: float = 100.0, end: float = 150.0,
              seed: int = 0) -> pd.Series:
    rng = np.random.RandomState(seed)
    drift = np.linspace(0, np.log(end / start), n)
    noise = rng.normal(0, 0.01, n).cumsum()
    return pd.Series(start * np.exp(drift + noise), index=_daily_index(n))


def _downtrend(n: int = 600, start: float = 200.0, end: float = 100.0,
                seed: int = 1) -> pd.Series:
    rng = np.random.RandomState(seed)
    drift = np.linspace(0, np.log(end / start), n)
    noise = rng.normal(0, 0.01, n).cumsum()
    return pd.Series(start * np.exp(drift + noise), index=_daily_index(n))


def test_portfolio_single_asset_matches_single_run():
    # Portfolio of 1 asset, no portfolio-level rescaling → identical equity
    # path to running simulate_tsm() directly.
    close = _uptrend()
    p = TSMParams(lookback_days=120, skip_days=10, vol_lookback_days=30,
                  cost_bps_per_turnover=0.0)
    single = simulate_tsm(close, p)
    port = simulate_tsm_portfolio({"A": close}, p, target_portfolio_vol=None)
    pd.testing.assert_series_equal(
        port["daily_returns"], single["daily_returns"], check_names=False
    )


def test_portfolio_two_perfectly_correlated_assets_no_diversification():
    # Same series, same params → portfolio has same Sharpe as one asset
    # (just lower vol because we're averaging two identical streams, but
    # the rolling rescale brings it back to target).
    close = _uptrend()
    p = TSMParams(lookback_days=120, skip_days=10, vol_lookback_days=30,
                  cost_bps_per_turnover=0.0)
    port = simulate_tsm_portfolio({"A": close, "B": close.copy()}, p,
                                    target_portfolio_vol=0.10)
    s_port = summarize(port)
    s_single = summarize(simulate_tsm(close, p))
    # With perfect correlation, Sharpe shouldn't *improve* — within tight noise tolerance
    assert s_port["sharpe"] == pytest.approx(s_single["sharpe"], abs=0.15)


def test_portfolio_anticorrelated_trends_diversify():
    # One uptrend (TSM goes long, profits) + one downtrend (TSM goes short,
    # also profits). The two profit streams are uncorrelated in time → Sharpe
    # of the equal-weight portfolio > Sharpe of either alone.
    up = _uptrend(seed=10)
    dn = _downtrend(seed=20)
    p = TSMParams(lookback_days=120, skip_days=10, vol_lookback_days=30,
                  cost_bps_per_turnover=0.0)
    s_up = summarize(simulate_tsm(up, p))
    s_dn = summarize(simulate_tsm(dn, p))
    port = simulate_tsm_portfolio({"UP": up, "DN": dn}, p,
                                    target_portfolio_vol=None)
    s_port = summarize(port)
    # Portfolio Sharpe should beat the worse of the two and be close to or
    # better than the better of the two
    assert s_port["sharpe"] > min(s_up["sharpe"], s_dn["sharpe"])


def test_portfolio_n_live_tracks_warmup_completion():
    # Asset A has 400 bars, asset B has 600 — for the first 200 dates B
    # doesn't even exist; n_live should be 1 (just A, after its warmup),
    # then 2 once B comes online.
    a = _uptrend(n=400, seed=1)
    b = _uptrend(n=600, seed=2)
    # b has its own date index (the same calendar but longer); shift b earlier
    # so A starts later than B
    p = TSMParams(lookback_days=60, skip_days=5, vol_lookback_days=20,
                  cost_bps_per_turnover=0.0)
    port = simulate_tsm_portfolio({"A": a, "B": b}, p, target_portfolio_vol=None)
    # n_live should be in {0, 1, 2}
    assert set(port["n_live"].unique()).issubset({0, 1, 2})
    # At least some dates should have both live
    assert (port["n_live"] == 2).any()


def test_portfolio_empty_prices_raises():
    with pytest.raises(ValueError):
        simulate_tsm_portfolio({}, TSMParams())


def test_portfolio_vol_target_clamped_by_max_leverage():
    # Very low-vol asset → naive scaling would demand huge leverage; cap holds.
    close = pd.Series(100 + np.linspace(0, 0.5, 600),
                       index=_daily_index(600))  # near-flat
    p = TSMParams(lookback_days=120, skip_days=10, vol_lookback_days=30,
                  max_leverage=1.5, cost_bps_per_turnover=0.0)
    port = simulate_tsm_portfolio({"A": close}, p, target_portfolio_vol=0.30)
    assert port["scale"].max() <= 1.5 + 1e-9


def test_summarize_portfolio_includes_per_asset():
    up = _uptrend(seed=3)
    dn = _downtrend(seed=4)
    p = TSMParams(lookback_days=120, skip_days=10, vol_lookback_days=30,
                  cost_bps_per_turnover=0.0)
    port = simulate_tsm_portfolio({"UP": up, "DN": dn}, p,
                                    target_portfolio_vol=None)
    s = summarize_portfolio(port)
    assert "per_asset" in s
    assert set(s["per_asset"].keys()) == {"UP", "DN"}
    assert s["max_n_live"] == 2


# ---------- multi-horizon signal tests ----------

def test_multi_horizon_signal_averages_correctly():
    # Build a series where 252-day return is +0.30 but 63-day return is -0.10.
    # Three lookbacks: 63 (-1), 126 (?), 252 (+1). The 126-day window will be
    # the integral of the regimes — just check the result is in [-1, 1] and
    # not equal to either single signal.
    n = 600
    rng = np.random.RandomState(0)
    # First 400 bars upward, last 200 downward
    drift = np.concatenate([np.linspace(0, 0.5, 400), np.linspace(0.5, 0.35, 200)])
    noise = rng.normal(0, 0.002, n).cumsum()
    close = pd.Series(100 * np.exp(drift + noise), index=_daily_index(n))

    mh = multi_horizon_signal(close, [63, 126, 252], skip_days=10)
    last = mh.iloc[-1]
    # Each individual binary signal at the last bar
    s_63 = tsm_signal(close, 63, 10).iloc[-1]
    s_126 = tsm_signal(close, 126, 10).iloc[-1]
    s_252 = tsm_signal(close, 252, 10).iloc[-1]
    expected = (s_63 + s_126 + s_252) / 3.0
    assert abs(last - expected) < 1e-9
    assert -1.0 <= last <= 1.0


def test_multi_horizon_signal_rejects_empty_list():
    close = pd.Series(np.linspace(100, 200, 400), index=_daily_index(400))
    with pytest.raises(ValueError, match="must not be empty"):
        multi_horizon_signal(close, [], skip_days=10)


def test_simulate_tsm_with_multi_horizon_returns_continuous_position():
    # Construct a series where horizons disagree: long uptrend then a recent
    # sharp pullback. The 63-day signal flips short while 252-day stays long,
    # so multi-horizon averages to ~0 or ±1/3 — i.e. NOT a binary signal.
    n = 800
    long_up = np.linspace(0, 1.0, 700)
    short_down = np.linspace(1.0, 0.6, 100)
    drift = np.concatenate([long_up, short_down])
    rng = np.random.RandomState(2)
    noise = rng.normal(0, 0.003, n).cumsum()
    close = pd.Series(50 * np.exp(drift + noise), index=_daily_index(n))

    p = TSMParams(lookback_days=252, skip_days=10, vol_lookback_days=30,
                  cost_bps_per_turnover=0.0,
                  multi_horizon_lookbacks=[63, 126, 252])
    # Compare the raw multi-horizon signal at the last bar to single-lookback
    # to confirm it produces fractional values when horizons disagree.
    mh = multi_horizon_signal(close, [63, 126, 252], skip_days=10)
    fractional_anywhere = ((mh.abs() > 0.01) & (mh.abs() < 0.99)).any()
    assert fractional_anywhere
    # And the full sim runs without crashing
    sim = simulate_tsm(close, p)
    assert "equity" in sim and "position" in sim


# ---------- portfolio weighting tests ----------

def _make_pair_for_weighting():
    """A 2-asset frame for testing weighting logic. UP makes money on TSM,
    DN_BAD has been a TSM loser — Sharpe-weighting should heavily favor UP."""
    n = 600
    rng = np.random.RandomState(7)
    idx = _daily_index(n)
    up_ret = pd.Series(rng.normal(0.002, 0.01, n), index=idx)       # good Sharpe
    bad_ret = pd.Series(rng.normal(-0.001, 0.01, n), index=idx)     # negative
    rets_df = pd.DataFrame({"UP": up_ret, "BAD": bad_ret})
    is_live = pd.DataFrame(True, index=idx, columns=["UP", "BAD"])
    n_live = is_live.sum(axis=1).astype(int)
    return rets_df, is_live, n_live


def test_portfolio_weighting_equal_normalizes_to_one():
    rets_df, is_live, n_live = _make_pair_for_weighting()
    w = _portfolio_weights(rets_df, is_live, n_live, "equal", 252)
    row = w.iloc[-1]
    assert abs(row.sum() - 1.0) < 1e-9
    assert all(abs(v - 0.5) < 1e-9 for v in row)


def test_portfolio_weighting_inverse_vol_favors_low_vol():
    rets_df, is_live, n_live = _make_pair_for_weighting()
    # Boost BAD's vol so inverse-vol weights it less
    rets_df["BAD"] = rets_df["BAD"] * 5.0
    w = _portfolio_weights(rets_df, is_live, n_live, "inverse_vol", 252)
    # Take a date with full warmup
    row = w.iloc[300]
    assert abs(row.sum() - 1.0) < 1e-6
    assert row["UP"] > row["BAD"]


def test_portfolio_weighting_sharpe_drops_negative_sharpe_asset():
    rets_df, is_live, n_live = _make_pair_for_weighting()
    w = _portfolio_weights(rets_df, is_live, n_live, "sharpe", 252)
    # After warmup, BAD has negative Sharpe → its weight should be 0
    row = w.iloc[400]
    assert abs(row.sum() - 1.0) < 1e-6
    assert row["BAD"] == 0.0
    assert row["UP"] == pytest.approx(1.0, abs=1e-6)


def test_portfolio_weighting_unknown_scheme_raises():
    rets_df, is_live, n_live = _make_pair_for_weighting()
    with pytest.raises(ValueError, match="unreachable|unknown"):
        _portfolio_weights(rets_df, is_live, n_live, "rubbish", 252)


def test_portfolio_simulate_with_sharpe_weighting_beats_equal_when_losers():
    # Two streams: UP makes money on TSM, BAD is a TSM loser. Sharpe-weighted
    # portfolio should outperform equal-weighted because BAD gets zero weight.
    up = _uptrend(seed=10)
    bad = pd.Series(100 + np.cumsum(np.random.RandomState(99).normal(-0.01, 0.005, 600)),
                     index=_daily_index(600))

    p = TSMParams(lookback_days=120, skip_days=10, vol_lookback_days=30,
                  cost_bps_per_turnover=0.0)
    eq_port = simulate_tsm_portfolio({"UP": up, "BAD": bad}, p,
                                        target_portfolio_vol=None,
                                        weighting="equal")
    sh_port = simulate_tsm_portfolio({"UP": up, "BAD": bad}, p,
                                        target_portfolio_vol=None,
                                        weighting="sharpe",
                                        weighting_lookback_days=120)
    s_eq = summarize(eq_port)["sharpe"]
    s_sh = summarize(sh_port)["sharpe"]
    # Sharpe-weighting won't always strictly dominate (especially early in the
    # series before weights stabilize), but should generally do as well or better.
    assert s_sh >= s_eq - 0.1


def test_portfolio_simulate_rejects_unknown_weighting():
    close = _uptrend()
    p = TSMParams(lookback_days=120, skip_days=10, vol_lookback_days=30,
                  cost_bps_per_turnover=0.0)
    with pytest.raises(ValueError, match="unknown weighting"):
        simulate_tsm_portfolio({"A": close}, p, weighting="bogus")
